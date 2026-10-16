"""Full-song audio from ACE-Step 1.5, as a service the broker starts and stops.

The composer writes scores. This is the other half: rendered audio with
vocals. It is a side feature — the MusicXML remains the thing that is
*correct*, and this is a performance of it.

**Why a separate process.** ACE-Step cannot live in this virtualenv. It pins
``torch==2.10.0+cu128`` and ``transformers<4.58``; we run torch 2.13.0+cu130
and transformers 5.14. Installing it here would downgrade torch, CUDA and
transformers, which breaks the image and video pipelines and most likely
CrewAI. So it gets its own venv (``uv sync`` in its own checkout) and we talk
to its REST API, exactly as we already talk to Ollama.

**Why the broker owns its lifecycle.** ACE-Step has no unload endpoint —
``offload`` is an init-time option only. A long-lived server would therefore
hold VRAM with no way for anything else to reclaim it, and on one card that
means the tutor and the researcher get pushed onto the CPU. So the server is
started when a song is actually wanted and stopped when the card is needed
for something else. A short idle window keeps consecutive songs fast, which
is the same bargain Ollama's ``keep_alive`` makes.

**The score is the input.** ACE-Step conditions on audio, not notation, which
at first looks like it makes this a parallel generator rather than a renderer.
But the composer already turns MusicXML into MIDI and MIDI into audio through
fluidsynth, so that render is the source audio for a ``cover`` task:

    MusicXML -> MIDI -> fluidsynth WAV -> ACE-Step cover -> produced audio

``cover`` keeps the structure and changes the timbre and production, and it
skips ACE-Step's language model entirely, so no LM needs loading for this
path. Metadata maps straight across: the composer session already holds
tempo, key signature and time signature.
"""

from __future__ import annotations

import os
import time
import shutil
import re as _re
import asyncio
import logging
import pathlib
import subprocess

logger = logging.getLogger(__name__)

# Where ACE-Step is checked out, with its own .venv created by `uv sync`.
ACESTEP_HOME = pathlib.Path(
    os.getenv("ACESTEP_HOME", "/home/bogdan/ai/ACE-Step-1.5"))
ACESTEP_PORT = int(os.getenv("ACESTEP_PORT", "8001"))
ACESTEP_BASE = f"http://127.0.0.1:{ACESTEP_PORT}"

# 2B turbo: the README's own recommendation for the 8-16GB tier, and it holds
# ~4.7GB of weights rather than the XL's ~9GB.
ACESTEP_MODEL = os.getenv("ACESTEP_MODEL", "acestep-v15-turbo")

# How long the server may sit idle before it is stopped and the card handed
# back. Long enough that a second song in the same sitting is fast, short
# enough that nobody waits on it.
IDLE_SECONDS = float(os.getenv("ACESTEP_IDLE_SECONDS", "300"))

# Startup is fast because of --no-init: the server answers /health in a few
# seconds and loads weights on the first request instead.
STARTUP_TIMEOUT = float(os.getenv("ACESTEP_STARTUP_TIMEOUT", "180"))

# A song itself takes seconds on this card. The *first* one also downloads
# roughly 6 GB of weights, measured here at under 2 MB/s, which is comfortably
# over an hour — so the default has to cover that or the first attempt always
# fails. Progress is reported from the server's own log so the wait is legible
# rather than silent.
GENERATE_TIMEOUT = float(os.getenv("ACESTEP_GENERATE_TIMEOUT", "7200"))

# The server's own output. A file rather than a pipe, deliberately — see
# ensure_running.
LOG_PATH = pathlib.Path(
    os.getenv("ACESTEP_LOG",
              "/home/bogdan/ai/agent/logs/acestep.log"))

_proc: subprocess.Popen | None = None
_last_used: float = 0.0
_lock = asyncio.Lock()


def is_installed() -> bool:
    """True when the checkout and its virtualenv both exist.

    Checked before a request runs so the UI can say what is missing instead
    of failing with a connection error.
    """
    return (ACESTEP_HOME / "acestep").is_dir() and _python().exists()


def _python() -> pathlib.Path:
    return ACESTEP_HOME / ".venv" / "bin" / "python"


def install_hint() -> str:
    """What to do when it is not installed, in one line."""
    return (f"ACE-Step is not installed at {ACESTEP_HOME}. Run: "
            f"git clone https://github.com/ACE-Step/ACE-Step-1.5.git "
            f"{ACESTEP_HOME} && cd {ACESTEP_HOME} && uv sync")


# -- the server ------------------------------------------------------------

def _healthy(timeout: float = 3.0) -> bool:
    import requests

    try:
        r = requests.get(f"{ACESTEP_BASE}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def running() -> bool:
    """True when a server is alive, ours or one left by a previous reload."""
    if _proc is not None and _proc.poll() is None:
        return True
    return bool(_orphan_pids())


async def ensure_running(on_progress=None) -> None:
    """Start the server if it is not already answering, and wait for it.

    *on_progress* is called with a short status string, so a streaming
    endpoint can explain a two-minute model load rather than appearing hung.
    """
    global _proc, _last_used

    async with _lock:
        if _healthy():
            _last_used = time.monotonic()
            return
        if not is_installed():
            raise RuntimeError(install_hint())

        await _say(on_progress, "Starting the song engine…")
        # Start from a scrubbed environment. ACE-Step runs a different torch
        # (2.10+cu128) against a different CUDA than ours (2.13+cu130), and
        # both virtualenvs ship their own nvidia/cudnn. Handing it our
        # PYTHONPATH, VIRTUAL_ENV or library paths invites it to load a mix of
        # the two, which surfaces as
        # CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH deep inside a convolution.
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME",
                            "LD_LIBRARY_PATH", "LD_PRELOAD")}
        env["ACESTEP_CONFIG_PATH"] = ACESTEP_MODEL
        env["ACESTEP_API_PORT"] = str(ACESTEP_PORT)
        # Offload, quantization and batch limits are applied by ACE-Step
        # itself from its own VRAM tier detection — its .env.example says so
        # explicitly ("ALWAYS applied") and there are no env vars to set.

        logger.info("[song] starting ACE-Step api_server on %d -> %s",
                    ACESTEP_PORT, LOG_PATH)
        # Output goes to a file, never to a pipe. A pipe has to be drained by
        # whoever holds the read end, and this server outlives the request
        # that started it: once that reader is gone every write fails with
        # EPIPE. Measured consequence, with stdout=PIPE: "Failed to download
        # main model: Both HuggingFace and ModelScope downloads failed.
        # HF: [Errno 32] Broken pipe, MS: [Errno 32] Broken pipe". The
        # download was killed by its own progress output.
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log = open(LOG_PATH, "a", buffering=1, errors="replace")
        log.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"model={ACESTEP_MODEL} =====\n")
        _proc = subprocess.Popen(
            # --no-init: come up without loading weights, so /health answers in
            # seconds and the load happens on the first request — which is
            # inside the GPU reservation, where it belongs. No --init-llm
            # either: `cover` skips ACE-Step's language model entirely, and
            # text2music with thinking off does not need it, so the 1.7B LM is
            # VRAM we never have to find.
            [str(_python()), "-m", "acestep.api_server",
             "--host", "127.0.0.1", "--port", str(ACESTEP_PORT), "--no-init"],
            cwd=str(ACESTEP_HOME), env=env,
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.monotonic() + STARTUP_TIMEOUT
        told = 0.0
        while time.monotonic() < deadline:
            if _proc.poll() is not None:
                _proc = None
                raise RuntimeError(
                    f"ACE-Step server exited at startup. Last lines of "
                    f"{LOG_PATH}:\n{log_tail()}")
            # gpu-queue-exempt: a health probe over HTTP, no GPU work here.
            if await asyncio.get_running_loop().run_in_executor(None, _healthy):
                _last_used = time.monotonic()
                logger.info("[song] ACE-Step ready")
                await _say(on_progress, "Song engine ready")
                return
            waited = time.monotonic() - (deadline - STARTUP_TIMEOUT)
            if waited - told > 15:
                told = waited
                await _say(on_progress,
                           f"Loading the song model… {waited:.0f}s "
                           f"(first run also downloads it)")
            await asyncio.sleep(2)

        stop()
        raise RuntimeError(
            f"ACE-Step did not become ready within {STARTUP_TIMEOUT:.0f}s")


def stop() -> None:
    """Stop the server and give the VRAM back.

    This is the only way to reclaim it: there is no unload endpoint, so the
    process has to go. Called by the broker when the card is needed for
    anything else.

    It also stops a server this process did not start. The web app runs under
    ``uvicorn --reload``, so it restarts on every code change and loses its
    handle; an ACE-Step left over from before the restart would otherwise hold
    ten gigabytes with nothing able to reclaim it.
    """
    global _proc
    if _proc is not None:
        logger.info("[song] stopping ACE-Step to free the GPU")
        try:
            _proc.terminate()
            try:
                _proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                _proc.kill()
                _proc.wait(timeout=10)
        except Exception as e:
            logger.warning("[song] could not stop ACE-Step cleanly: %s", e)
        finally:
            _proc = None

    _stop_orphans()


def _stop_orphans() -> None:
    """Kill any acestep api_server we did not start, by command line.

    Matched narrowly on ``acestep.api_server`` and our own port so this cannot
    reach anything else the person happens to be running.
    """
    import signal

    for pid in _orphan_pids():
        logger.info("[song] stopping orphaned ACE-Step pid %d", pid)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            except PermissionError:
                logger.warning("[song] not allowed to stop pid %d", pid)
                break
            for _ in range(20):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            else:
                continue
            break


def _orphan_pids() -> list[int]:
    """PIDs of acestep api_server processes bound to our port."""
    pids = []
    me = os.getpid()
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid == me:
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace")
        except (OSError, PermissionError):
            continue
        if "acestep.api_server" in cmdline and str(ACESTEP_PORT) in cmdline:
            pids.append(pid)
    return pids


_DOWNLOAD_RE = _re.compile(
    r"([\w.\-/]+(?:\.safetensors|\.bin|\.pt|\.ckpt))[^\d]*?"
    r"(\d+)%.*?([\d.]+[KMG])/([\d.]+[KMG])")


def progress_note() -> str:
    """What the engine is actually doing, from its own log.

    The first song downloads several gigabytes, and "Composing the audio…"
    for an hour is indistinguishable from a hang. ACE-Step writes its download
    progress to the log, so the real state is there to be read.
    """
    tail = log_tail(40)
    match = None
    for m in _DOWNLOAD_RE.finditer(tail):
        match = m          # the last one is the current file
    if match:
        name = match.group(1).split("/")[-1]
        return (f"Downloading {name} — {match.group(2)}% "
                f"({match.group(3)}/{match.group(4)}), one-time setup")
    if "Loading" in tail or "loading" in tail:
        return "Loading the song model"
    return "Composing the audio"


def log_tail(lines: int = 25) -> str:
    """The last lines of the server's log, for explaining a failure."""
    try:
        return "\n".join(
            LOG_PATH.read_text(errors="replace").splitlines()[-lines:])
    except Exception:
        return "(no log)"


def idle_seconds() -> float:
    """How long since the server was last used. 0 when it is not running."""
    return 0.0 if not running() else max(0.0, time.monotonic() - _last_used)


def stop_if_idle() -> bool:
    """Stop the server when it has been unused past the idle window."""
    if running() and idle_seconds() > IDLE_SECONDS:
        stop()
        return True
    return False


# -- generating ------------------------------------------------------------

async def generate(
    *,
    prompt: str,
    lyrics: str = "",
    bpm: int | None = None,
    key_scale: str = "",
    time_signature: str = "",
    duration: float | None = None,
    vocal_language: str = "en",
    src_audio: bytes | None = None,
    src_name: str = "score.wav",
    cover_strength: float = 0.75,
    audio_format: str = "mp3",
    on_progress=None,
) -> tuple[bytes, dict]:
    """Generate one song. Returns (audio bytes, metadata).

    With *src_audio* this is a ``cover`` of that audio — the composer's own
    score, rendered — which keeps the structure and changes the production.
    Without it, a plain text-to-music generation from *prompt*.
    """
    import requests

    await ensure_running(on_progress)

    fields: dict[str, str] = {
        "prompt": prompt,
        "lyrics": lyrics or "",
        "audio_format": audio_format,
        "vocal_language": vocal_language or "en",
    }
    if bpm:
        fields["bpm"] = str(int(bpm))
    if key_scale:
        fields["key_scale"] = key_scale
    if time_signature:
        fields["time_signature"] = time_signature
    if duration and duration > 0:
        fields["audio_duration"] = str(float(duration))

    if src_audio:
        fields["task_type"] = "cover"
        fields["audio_cover_strength"] = str(float(cover_strength))
        # cover skips ACE-Step's language model entirely, so nothing extra is
        # loaded for this path.
        files = {"src_audio": (src_name, src_audio, "audio/wav")}
    else:
        fields["task_type"] = "text2music"
        files = None

    loop = asyncio.get_running_loop()

    def _submit():
        url = f"{ACESTEP_BASE}/release_task"
        if files:
            return requests.post(url, data=fields, files=files, timeout=120)
        return requests.post(url, json=fields, timeout=120)

    await _say(on_progress, "Composing the audio…")
    # gpu-queue-exempt: submits over HTTP to the ACE-Step process. The GPU
    # reservation is held by the /composer/song route for the whole call, and
    # test_song.py fails if it is not.
    resp = await loop.run_in_executor(None, _submit)
    resp.raise_for_status()
    body = resp.json()
    task_id = _task_id(body)
    if not task_id:
        raise RuntimeError(f"ACE-Step did not return a task id: {body}")

    result = await _await_task(task_id, on_progress)
    # gpu-queue-exempt: downloads the finished file over HTTP.
    audio = await loop.run_in_executor(None, _fetch_audio, result)
    global _last_used
    _last_used = time.monotonic()
    return audio, result


def _task_id(body: dict):
    data = body.get("data") if isinstance(body, dict) else None
    for src in (data, body):
        if isinstance(src, dict):
            for key in ("task_id", "taskId", "id"):
                if src.get(key):
                    return src[key]
    return None


async def _await_task(task_id, on_progress) -> dict:
    """Poll until the task finishes, and return its parsed result.

    Taken from docs/en/API.md rather than guessed. Three details here each
    cost a wasted run:

    - the request field is ``task_id_list``, not ``task_ids``; the wrong name
      returns ``{"data": []}`` with HTTP 200, which polls for ever;
    - ``data`` is empty until the task is registered, which means "still
      queued", not "failed";
    - ``result`` is a **JSON string**, not an object, and the audio reference
      lives at ``result[0]["file"]``.
    """
    import requests

    loop = asyncio.get_running_loop()
    deadline = time.monotonic() + GENERATE_TIMEOUT
    told = 0.0

    def _query():
        r = requests.post(f"{ACESTEP_BASE}/query_result",
                          json={"task_id_list": [task_id]}, timeout=30)
        r.raise_for_status()
        return r.json()

    while time.monotonic() < deadline:
        # gpu-queue-exempt: polls task status over HTTP.
        body = await loop.run_in_executor(None, _query)
        entries = body.get("data") if isinstance(body, dict) else None
        entry = None
        if isinstance(entries, list):
            entry = next((e for e in entries
                          if isinstance(e, dict)
                          and str(e.get("task_id")) == str(task_id)), None)

        status = _status_of(entry) if entry else None
        if status == 1:
            return _parse_result(entry)
        if status == 2:
            raise RuntimeError(_error_of(entry)
                               or "ACE-Step reported the generation failed")

        waited = time.monotonic() - (deadline - GENERATE_TIMEOUT)
        if waited - told > 10:
            told = waited
            await _say(on_progress, f"{progress_note()} — {waited:.0f}s")
        await asyncio.sleep(2)

    raise RuntimeError(
        f"ACE-Step did not finish within {GENERATE_TIMEOUT:.0f}s")


def _status_of(entry: dict):
    """Status as an int. Submit replies with a string, queries with an int."""
    raw = entry.get("status")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw or "").strip().lower()
    return {"queued": 0, "running": 0, "pending": 0,
            "succeeded": 1, "success": 1, "done": 1,
            "failed": 2, "error": 2}.get(text)


def _error_of(entry: dict) -> str:
    """The failure message, which is buried inside the JSON-string result.

    The top-level entry carries only ``status: 2``; the reason is in
    ``result[0]["error"]``. Reading only the top level produced "ACE-Step
    reported the generation failed" and nothing else, which is useless.
    """
    for key in ("error", "message", "detail"):
        if entry.get(key):
            return str(entry[key])
    try:
        inner = _parse_result(entry)
    except Exception:
        return ""
    for key in ("error", "message", "detail", "stage"):
        if inner.get(key):
            return str(inner[key])
    return ""


def _parse_result(entry: dict) -> dict:
    """Unpack the JSON-string ``result`` into the first audio's dict."""
    import json

    raw = entry.get("result")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"ACE-Step returned unreadable result: {e}")
    if isinstance(raw, list):
        # An empty list used to fall through as {}, which then failed later
        # with the vaguer "returned no audio". Fail here, where we know why.
        if not raw:
            raise RuntimeError(
                "ACE-Step reported success but returned an empty result")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise RuntimeError("ACE-Step returned no result for a finished task")
    return raw


def _fetch_audio(result: dict) -> bytes:
    """Read the finished audio: a local path when visible, else over HTTP."""
    import requests

    refs = [result.get(k) for k in ("file", "path", "audio_path", "url")]
    for ref in [r for r in refs if r]:
        ref = str(ref)
        # A bare filesystem path is often readable directly: ACE-Step runs on
        # this machine.
        if ref.startswith("/") and "?" not in ref:
            local = pathlib.Path(ref)
            if local.is_file():
                return local.read_bytes()
        url = ACESTEP_BASE + ref if ref.startswith("/") else ref
        try:
            r = requests.get(url, timeout=180)
            if r.ok and r.content:
                return r.content
        except Exception as e:
            logger.debug("[song] could not fetch %s: %s", url, e)
    raise RuntimeError("ACE-Step finished but returned no audio")


async def _say(on_progress, text: str) -> None:
    """Report progress. A failure here must never fail the generation."""
    if on_progress is None:
        return
    try:
        got = on_progress(text)
        if asyncio.iscoroutine(got):
            await got
    except Exception as e:
        logger.debug("[song] progress callback failed: %s", e)


def have_fluidsynth() -> bool:
    """Whether a score can be rendered to audio for the cover path."""
    return bool(shutil.which("fluidsynth"))
