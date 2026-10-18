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

# The two models worth offering on this card, and the honest difference.
#
# Only `base` can do extract/lego/complete — the Model Zoo table marks those
# ❌ for turbo and sft — so stem separation, and therefore transcribing a song
# back to notation, needs base loaded. It is the same 2B size but runs 50
# diffusion steps against turbo's 8, so it is several times slower per call.
#
# The model belongs to the engine *process*: one slot, chosen at startup. That
# is why switching restarts it, which costs about as much as the first song of
# a sitting and nothing at all if you do not switch.
MODELS: dict[str, dict] = {
    "acestep-v15-turbo": {
        "label": "Fast",
        "steps": 8,
        "guidance": None,        # turbo bakes guidance in and forces 1.0
        "stems": False,
        "note": "turbo, 8 steps. ACE-Step's own day-to-day choice.",
    },
    "acestep-v15-base": {
        "label": "Detailed",
        "steps": 50,
        "guidance": 7.0,         # base is the one that actually uses CFG
        "stems": True,
        "note": "base, 50 steps, follows the description more literally.",
    },
}

# ACE-Step caps a batch at 8 without its LM; four takes is plenty to choose
# from and keeps a render under a minute warm.
MAX_TAKES = 4


def model_choices() -> list[dict]:
    """The models offerable, for the UI to render."""
    return [{"id": k, **v} for k, v in MODELS.items()]


def resolve_model(name: str | None) -> str:
    """A requested model name, or the default. Never anything else.

    An allowlist rather than passing a string through: the name becomes an
    environment variable for a subprocess and a path for a download.
    """
    if name and name in MODELS:
        return name
    return ACESTEP_MODEL if ACESTEP_MODEL in MODELS else "acestep-v15-turbo"

# How long the server may sit idle before it is stopped and the card handed
# back. Long enough that a second song in the same sitting is fast, short
# enough that nobody waits on it.
IDLE_SECONDS = float(os.getenv("ACESTEP_IDLE_SECONDS", "300"))

# Timestep shift. The REST API's own default is 3.0; it is sent explicitly so
# a change of server default cannot silently change every song. See generate().
TURBO_SHIFT = float(os.getenv("ACESTEP_TURBO_SHIFT", "3.0"))


# Startup is fast because of --no-init: the server answers /health in a few
# seconds and loads weights on the first request instead.
STARTUP_TIMEOUT = float(os.getenv("ACESTEP_STARTUP_TIMEOUT", "180"))

# A song itself takes seconds on this card. The *first* one also downloads
# roughly 6 GB of weights, measured here at under 2 MB/s, which is comfortably
# over an hour — so the default has to cover that or the first attempt always
# fails. Progress is reported from the server's own log so the wait is legible
# rather than silent.
GENERATE_TIMEOUT = float(os.getenv("ACESTEP_GENERATE_TIMEOUT", "7200"))

# The server's own output, when this module starts it. A file rather than a
# pipe, deliberately — see _start_subprocess.
LOG_PATH = pathlib.Path(
    os.getenv("ACESTEP_LOG",
              str(pathlib.Path(__file__).resolve().parents[2] / "logs" / "acestep.log")))

_proc: subprocess.Popen | None = None
_last_used: float = 0.0
_lock = asyncio.Lock()


def loaded_model() -> str:
    """The DiT model the engine currently holds, or "" when it is stopped."""
    h = _health()
    return (h.get("loaded_model") or "") if h else ""


def holding_gpu() -> bool:
    """True when the engine has weights loaded, and so is occupying VRAM.

    Not the same as running. A systemd-managed engine is always running, but
    with lazy loading it holds no VRAM until the first request, and ~10 GB
    after it — measured on this card at 9,984 MiB after a 20-second song, and
    it does not let go on its own.
    """
    h = _health()
    return bool(h and h.get("models_initialized"))


# -- who owns the process ----------------------------------------------------
#
# Two ways to run ACE-Step, and they must not fight. Bogdan runs it as a
# systemd user service; installs without one get a subprocess this module
# starts. Before this was explicit, the broker found the engine by command
# line and killed it to free the GPU — including the systemd service, which
# exits cleanly on SIGTERM, so `Restart=on-failure` never brought it back. The
# journal shows exactly that: "Shutting down" at 20:21:51 on 2026-10-17, and
# every later song ran on a stray copy this module had started instead.

SYSTEMD_UNIT = os.getenv("ACESTEP_SYSTEMD_UNIT", "ace-step")
_systemd_cache: tuple[float, bool] = (0.0, False)


def _systemd_managed() -> bool:
    """True when a systemd user unit owns ACE-Step. Cached for a minute."""
    global _systemd_cache
    now = time.monotonic()
    if now - _systemd_cache[0] < 60:
        return _systemd_cache[1]
    try:
        found = subprocess.run(
            ["systemctl", "--user", "cat", f"{SYSTEMD_UNIT}.service"],
            capture_output=True, timeout=10).returncode == 0
    except Exception:
        found = False
    _systemd_cache = (now, found)
    return found


def _systemctl(*args: str) -> bool:
    """Run a systemctl --user command. Returns whether it succeeded."""
    try:
        r = subprocess.run(["systemctl", "--user", *args,
                            f"{SYSTEMD_UNIT}.service"],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        logger.warning("[song] systemctl %s failed: %s", " ".join(args), e)
        return False
    if r.returncode != 0:
        logger.warning("[song] systemctl %s: %s", " ".join(args),
                       (r.stderr or r.stdout).strip()[:300])
    return r.returncode == 0


def _health(timeout: float = 3.0) -> dict | None:
    """The engine's /health payload, or None when it is not answering."""
    import requests

    try:
        r = requests.get(f"{ACESTEP_BASE}/health", timeout=timeout)
        if r.status_code != 200:
            return None
        body = r.json()
        return body.get("data") if isinstance(body, dict) else None
    except Exception:
        return None


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
    return _health(timeout) is not None


def running() -> bool:
    """True when an ACE-Step server is answering, whoever started it."""
    if _systemd_managed():
        return _healthy()
    if _proc is not None and _proc.poll() is None:
        return True
    return bool(_orphan_pids())


async def ensure_running(on_progress=None, model: str | None = None) -> None:
    """Make sure the engine answers, and holds *model*.

    Whoever owns the process starts it: systemd when a unit exists, otherwise
    a subprocess of ours. Either way the model is switched in place with
    ``POST /v1/init`` rather than by restarting — measured on this card at
    1.0-1.3s to turbo and 9.5s to base, against ~50s for a cold restart.

    *on_progress* is called with a short status string, so a streaming
    endpoint can explain a model load rather than appearing hung.
    """
    global _last_used

    want = resolve_model(model)
    loop = asyncio.get_running_loop()

    async with _lock:
        # gpu-queue-exempt: a health probe over HTTP, no GPU work here.
        if not await loop.run_in_executor(None, _healthy):
            managed = _systemd_managed()
            if not managed and not is_installed():
                raise RuntimeError(install_hint())
            await _say(on_progress, "Starting the song engine…")
            if managed:
                logger.info("[song] starting %s.service", SYSTEMD_UNIT)
                # gpu-queue-exempt: asks systemd to start a unit, no GPU work.
                await loop.run_in_executor(None, _systemctl, "start")
            else:
                _start_subprocess(want)
            await _wait_until_healthy(on_progress)

        # gpu-queue-exempt: reads /health over HTTP.
        loaded = await loop.run_in_executor(None, loaded_model)
        if loaded and loaded != want:
            await _say(on_progress,
                       f"Switching to the {MODELS[want]['label'].lower()} "
                       f"engine…")
            # gpu-queue-exempt: an HTTP call; the caller holds the GPU.
            await loop.run_in_executor(None, _switch_model, want)
        elif not loaded and want != _configured_model():
            # Lazily loaded: nothing is resident yet, but the first request
            # would load the configured model, so name the one we want first.
            # gpu-queue-exempt: an HTTP call; the caller holds the GPU.
            await loop.run_in_executor(None, _switch_model, want)
        _last_used = time.monotonic()


def _configured_model() -> str:
    """What the engine loads by default when nothing has been asked for."""
    h = _health() or {}
    return h.get("loaded_model") or ACESTEP_MODEL


def _switch_model(want: str) -> None:
    """Load *want* into the engine's slot without restarting it."""
    import requests

    t0 = time.monotonic()
    r = requests.post(f"{ACESTEP_BASE}/v1/init", json={"model": want},
                      timeout=STARTUP_TIMEOUT + 900)
    data = (r.json() or {}).get("data") or {} if r.ok else {}
    if not r.ok or data.get("loaded_model") != want:
        raise RuntimeError(f"ACE-Step could not switch to {want}: "
                           f"HTTP {r.status_code} {r.text[:200]}")
    logger.info("[song] engine now on %s (%.1fs)", want, time.monotonic() - t0)


def _start_subprocess(want: str) -> None:
    """Start ACE-Step ourselves, for installs with no systemd unit."""
    global _proc

    # Start from a scrubbed environment. ACE-Step runs a different torch
    # (2.10+cu128) against a different CUDA than ours (2.13+cu130), and both
    # virtualenvs ship their own nvidia/cudnn. Handing it our PYTHONPATH,
    # VIRTUAL_ENV or library paths makes it load a mix of the two, which
    # surfaces as CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH inside a
    # convolution.
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME",
                        "LD_LIBRARY_PATH", "LD_PRELOAD")}
    env["ACESTEP_CONFIG_PATH"] = want
    env["ACESTEP_API_PORT"] = str(ACESTEP_PORT)

    logger.info("[song] starting ACE-Step api_server on %d -> %s",
                ACESTEP_PORT, LOG_PATH)
    # Output goes to a file, never to a pipe. A pipe has to be drained by
    # whoever holds the read end, and this server outlives the request that
    # started it: once that reader is gone every write fails with EPIPE. With
    # stdout=PIPE the model download died of its own progress output:
    # "Both HuggingFace and ModelScope downloads failed. HF: [Errno 32]
    # Broken pipe".
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "a", buffering=1, errors="replace")
    log.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"model={want} =====\n")
    _proc = subprocess.Popen(
        # --no-init: come up without loading weights, so /health answers in
        # seconds and the load happens on the first request, inside the GPU
        # reservation. No --init-llm: neither cover nor text2music with
        # thinking off uses ACE-Step's language model.
        [str(_python()), "-m", "acestep.api_server",
         "--host", "127.0.0.1", "--port", str(ACESTEP_PORT), "--no-init"],
        cwd=str(ACESTEP_HOME), env=env,
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


async def _wait_until_healthy(on_progress) -> None:
    global _proc

    loop = asyncio.get_running_loop()
    deadline = time.monotonic() + STARTUP_TIMEOUT
    told = 0.0
    while time.monotonic() < deadline:
        if _proc is not None and _proc.poll() is not None:
            _proc = None
            raise RuntimeError(
                f"ACE-Step server exited at startup. Last lines of "
                f"{LOG_PATH}:\n{log_tail()}")
        # gpu-queue-exempt: a health probe over HTTP, no GPU work here.
        if await loop.run_in_executor(None, _healthy):
            logger.info("[song] ACE-Step ready")
            await _say(on_progress, "Song engine ready")
            return
        waited = time.monotonic() - (deadline - STARTUP_TIMEOUT)
        if waited - told > 15:
            told = waited
            await _say(on_progress, f"Starting the song engine… {waited:.0f}s")
        await asyncio.sleep(1)
    raise RuntimeError(
        f"ACE-Step did not become ready within {STARTUP_TIMEOUT:.0f}s")


def stop() -> bool:
    """Make the engine give its VRAM back. Returns whether anything was freed.

    ACE-Step has no unload endpoint and keeps its weights resident after a
    request (9,984 MiB measured), so the process has to restart. Under systemd
    that is ``systemctl --user restart``: the service comes straight back with
    lazy loading and holds no VRAM until the next song. Nothing is restarted
    when nothing is held, because that would only cost the next song a reload.

    Without systemd, the subprocess this module started is stopped, along with
    any stray copy a previous ``uvicorn --reload`` left behind.
    """
    global _proc

    if _systemd_managed():
        if not holding_gpu():
            logger.info("[song] engine holds no VRAM; nothing to free")
            return False
        logger.info("[song] restarting %s.service to free the GPU", SYSTEMD_UNIT)
        audit("engine-restart", reason="free the GPU")
        return _systemctl("restart")

    freed = False
    if _proc is not None:
        logger.info("[song] stopping ACE-Step to free the GPU")
        try:
            _proc.terminate()
            try:
                _proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                _proc.kill()
                _proc.wait(timeout=10)
            freed = True
        except Exception as e:
            logger.warning("[song] could not stop ACE-Step cleanly: %s", e)
        finally:
            _proc = None

    return _stop_orphans() or freed


def _stop_orphans() -> bool:
    """Kill any acestep api_server we did not start, by command line.

    Matched narrowly on ``acestep.api_server`` and our own port so this cannot
    reach anything else the person happens to be running — and never a
    process systemd owns, which this used to kill.
    """
    import signal

    stopped = False
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
                    stopped = True
                    break
            else:
                continue
            break
    return stopped


def _orphan_pids() -> list[int]:
    """PIDs of stray acestep api_server processes bound to our port.

    "Stray" excludes anything systemd owns. The match is on the command line,
    and the systemd service's command line (``uvicorn acestep.api_server:app
    --port 8001``) matched it perfectly — which is how the broker came to kill
    the service it should have been asking.
    """
    pids = []
    me = os.getpid()
    owned = _systemd_pids()
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid == me or pid in owned:
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        if _is_acestep_server(raw):
            pids.append(pid)
    return pids


def _is_acestep_server(raw_cmdline: bytes) -> bool:
    """Whether a /proc cmdline is an ACE-Step API server on our port.

    Parsed as an argument vector, not searched as one string. The substring
    test matched *any* process whose command line merely mentioned
    "acestep.api_server" and the port — a shell running grep, an editor, or
    the very shell that wrote this function, which it reported as an orphan
    to kill. It must be a Python interpreter, running the server module, with
    our port as its own argument.
    """
    argv = [a.decode(errors="replace") for a in raw_cmdline.split(b"\0") if a]
    if not argv or not pathlib.Path(argv[0]).name.startswith("python"):
        return False
    module = any(a == "acestep.api_server" or a.startswith("acestep.api_server:")
                 for a in argv[1:])
    port = str(ACESTEP_PORT)
    on_port = any(a == port or a == f"--port={port}" for a in argv[1:])
    return module and on_port


def _systemd_pids() -> set[int]:
    """Every PID in the systemd unit's cgroup, so they are never touched."""
    if not _systemd_managed():
        return set()
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "-p", "ControlGroup", "--value",
             f"{SYSTEMD_UNIT}.service"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if not out:
            return set()
        procs = pathlib.Path("/sys/fs/cgroup") / out.lstrip("/") / "cgroup.procs"
        return {int(x) for x in procs.read_text().split() if x.strip()}
    except Exception:
        return set()


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
    instrumental: bool | None = None,
    bpm: int | None = None,
    key_scale: str = "",
    time_signature: str = "",
    duration: float | None = None,
    vocal_language: str = "auto",
    takes: int = 2,
    seed: int | None = None,
    src_audio: bytes | None = None,
    src_name: str = "score.wav",
    cover_strength: float = 0.75,
    audio_format: str = "mp3",
    model: str | None = None,
    on_progress=None,
) -> list[dict]:
    """Generate a song. Returns one dict per take.

    Each take is ``{"audio": bytes, "seed": int | None, "meta": dict}``.

    *takes* is ACE-Step's ``batch_size``. Measured warm on this card, two
    takes cost no more than one (8.4s against 9.1s) because they run in
    parallel — and the server made two anyway when it was not asked, which
    this used to throw half of away.

    With *src_audio* this is a ``cover`` of that audio, which keeps the
    structure and changes the production. Without it, text-to-music.
    """
    import requests

    want = resolve_model(model)
    spec = MODELS[want]
    await ensure_running(on_progress, model=want)

    if instrumental is None:
        instrumental = not (lyrics or "").strip()
    language = (detect_language(lyrics) if vocal_language in ("", "auto", None)
                else vocal_language)
    fields: dict[str, str] = {
        "prompt": prompt,
        # ACE-Step's documented convention for a track with no singing, and
        # `instrumental` is a real switch rather than something to infer.
        "lyrics": "[Instrumental]" if instrumental else lyrics.strip(),
        "instrumental": "true" if instrumental else "false",
        "audio_format": audio_format,
        "vocal_language": language,
        "batch_size": str(max(1, min(int(takes or 1), MAX_TAKES))),
        # Sent explicitly, every time. Left to the server's default (8), the
        # "50 step" engine ran at 8 and sounded the same as the fast one —
        # which is exactly what was heard.
        "inference_steps": str(spec["steps"]),
        # The REST API already defaults this to 3.0 (release_task_models.py),
        # so sending it changes nothing today; it is explicit so a change of
        # server default cannot silently change our songs. Turbo does use it —
        # SHIFT_TIMESTEPS in models/turbo has a schedule per value — despite
        # API.md saying otherwise. v3.1.1 claimed earlier songs ran at 1.0;
        # that was the Python function's default, which the API overrides.
        "shift": str(TURBO_SHIFT),
    }
    if spec.get("guidance"):
        # Only the base model uses CFG; turbo forces it to 1.0 regardless.
        fields["guidance_scale"] = str(spec["guidance"])
    if seed is not None and int(seed) >= 0:
        fields["use_random_seed"] = "false"
        fields["seed"] = str(int(seed))
    if bpm:
        fields["bpm"] = str(int(bpm))
    if key_scale:
        fields["key_scale"] = key_scale
    if time_signature:
        fields["time_signature"] = str(time_signature).split("/")[0]
    if duration and duration > 0:
        fields["audio_duration"] = str(float(duration))

    if src_audio:
        fields["task_type"] = "cover"
        fields["audio_cover_strength"] = str(float(cover_strength))
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

    n = int(fields["batch_size"])
    await _say(on_progress, f"Composing {n} take{'s' if n > 1 else ''}…")

    # One retry, for one cause. The GPU queue only governs this app; another
    # Ollama client (OpenClaw here) can load a model mid-render — it did, 9.5
    # GB of devstral-small-2:24b, leaving 12 MiB free. Freeing Ollama again and
    # trying once usually succeeds; failing twice gets a message naming what
    # holds the card instead of the engine's "failed".
    for attempt in (1, 2):
        # gpu-queue-exempt: submits over HTTP to the ACE-Step process. The GPU
        # reservation is held by the /composer/song route for the whole call,
        # and test_song.py fails if it is not.
        resp = await loop.run_in_executor(None, _submit)
        resp.raise_for_status()
        body = resp.json()
        task_id = _task_id(body)
        if not task_id:
            raise RuntimeError(f"ACE-Step did not return a task id: {body}")
        try:
            results = await _await_task(task_id, on_progress)
            break
        except RuntimeError as e:
            if not _is_out_of_memory(str(e)):
                raise
            who = gpu_occupants()
            logger.warning("[song] out of GPU memory (attempt %d): %s", attempt, who)
            if attempt == 2:
                held = "; ".join(who) if who else "something outside this app"
                raise RuntimeError(
                    f"The GPU ran out of memory twice. Holding it: {held}. "
                    f"Programs that use Ollama directly bypass this app's GPU "
                    f"queue.") from e
            await _say(on_progress, "Out of GPU memory — another program loaded "
                                    "a model. Freeing it and trying again…")
            from researcher.gpu_broker import _release_ollama

            # gpu-queue-exempt: the route already holds the GPU; this makes
            # room inside that reservation.
            await loop.run_in_executor(None, _release_ollama, "song-retry")
    seeds = _seeds_of(results)
    out = []
    for i, res in enumerate(results):
        # gpu-queue-exempt: downloads the finished file over HTTP.
        audio = await loop.run_in_executor(None, _fetch_audio, res)
        out.append({"audio": audio, "seed": seeds[i] if i < len(seeds) else None,
                    "meta": {**(res.get("metas") or {}), "language": language,
                             "model": want, "steps": spec["steps"]}})
    global _last_used
    _last_used = time.monotonic()
    return out


def _seeds_of(results: list[dict]) -> list[int | None]:
    """Per-take seeds. ACE-Step reports them as one "a,b" string per entry."""
    for res in results:
        raw = str(res.get("seed_value") or "")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if parts:
            out: list[int | None] = []
            for p in parts:
                try:
                    out.append(int(p))
                except ValueError:
                    out.append(None)
            return out
    return []


def detect_language(lyrics: str, fallback: str = "en") -> str:
    """The language the lyrics are written in, as an ISO 639-1 code.

    ACE-Step uses ``vocal_language`` for pronunciation and never reads it from
    the description, so French lyrics with the field left on English sing
    against themselves. Section tags and stage directions are stripped first;
    too little text to judge falls back rather than guessing.
    """
    text = _re.sub(r"\[[^\]]*\]", " ", lyrics or "")
    words = [w for w in _re.findall(r"\w+", text) if len(w) > 1]
    if len(words) < 4:
        return fallback
    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0          # langdetect is random otherwise
        best = detect_langs(" ".join(words))[0]
    except Exception:
        return fallback
    code = best.lang.split("-")[0]
    return code if best.prob >= 0.6 else fallback


def _task_id(body: dict):
    data = body.get("data") if isinstance(body, dict) else None
    for src in (data, body):
        if isinstance(src, dict):
            for key in ("task_id", "taskId", "id"):
                if src.get(key):
                    return src[key]
    return None


async def _await_task(task_id, on_progress) -> list[dict]:
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
            return _parse_results(entry)
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
    """The failure message, wherever this version of ACE-Step put it.

    It moves around. One failure carried it as ``result[0]["error"]``; a CUDA
    out-of-memory carried only ``stage: "failed"`` inside the result, with the
    real reason in a traceback in the top-level ``progress_text``. Reading
    ``stage`` first reported just "failed", which explained nothing.
    """
    for key in ("error", "message", "detail"):
        if entry.get(key):
            return str(entry[key])
    try:
        inner = _parse_result(entry)
    except Exception:
        inner = {}
    for key in ("error", "message", "detail"):
        if inner.get(key):
            return str(inner[key])
    reason = _exception_line(entry.get("progress_text") or "")
    if reason:
        return reason
    return str(inner.get("stage") or "")


def _exception_line(text: str) -> str:
    """The last "SomeError: message" line of a traceback, if there is one."""
    found = ""
    for line in (text or "").splitlines():
        m = _re.search(r"\b(\w*(?:Error|Exception))\s*:\s*(.+)", line)
        if m:
            found = f"{m.group(1)}: {m.group(2).strip()}"
    return found[:600]


def gpu_occupants() -> list[str]:
    """Who else holds GPU memory, in words, for an out-of-memory message.

    The GPU queue only governs this app. Anything else talking to Ollama —
    OpenClaw on this machine, configured with devstral-small-2:24b — can load
    a model at any moment, and did, mid-render: 9,504 MiB, leaving 12 MiB.
    """
    out: list[str] = []
    try:
        import requests

        ps = requests.get("http://localhost:11434/api/ps", timeout=3).json()
        for m in ps.get("models", []):
            gib = (m.get("size_vram") or 0) / 2**30
            out.append(f"Ollama has {m.get('name')} loaded ({gib:.1f} GiB)")
    except Exception:
        pass
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        for line in smi.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) >= 1024 \
                    and "ollama" not in parts[2] and "ACE-Step" not in parts[2]:
                out.append(f"{pathlib.Path(parts[2]).name} (pid {parts[0]}) "
                           f"holds {int(parts[1]) / 1024:.1f} GiB")
    except Exception:
        pass
    return out


def _is_out_of_memory(message: str) -> bool:
    return "out of memory" in (message or "").lower()


def _parse_results(entry: dict) -> list[dict]:
    """Every take in a finished task, in batch order.

    ACE-Step builds the list from ``audio_paths`` in order
    (api/http/query_result_service.py), so index 0 is the first take written.
    """
    import json

    raw = entry.get("result")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"ACE-Step returned unreadable result: {e}")
    if isinstance(raw, dict):
        raw = [raw]
    takes = [r for r in (raw or []) if isinstance(r, dict) and r.get("file")]
    if not takes:
        raise RuntimeError(
            "ACE-Step reported success but returned an empty result")
    return takes


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


# -- where songs live ------------------------------------------------------
#
# Two folders, and the separation is the safety mechanism:
#
#     scratch/  renders nobody has saved yet. Pruned after a grace period.
#     kept/     saved songs. Nothing in this module ever deletes from here.
#
# Before, everything sat together in static/generated and only a database
# lookup stood between a saved song and the cleanup. That held for the real
# server, but every throwaway test server used its own database against the
# same folder — and from its point of view nothing there was saved. Moving a
# song out of scratch when it is saved makes it unreachable by the pruner by
# construction, whichever database the process happens to have.
#
# ACESTEP_SONG_ROOT puts the whole tree elsewhere, which is what test servers
# must do.

SONG_ROOT = pathlib.Path(os.getenv(
    "ACESTEP_SONG_ROOT", str(pathlib.Path(__file__).resolve().parent / "data" / "songs")))
SCRATCH_DIR = SONG_ROOT / "scratch"
KEPT_DIR = SONG_ROOT / "kept"
SONG_URL_PREFIX = "/songs"

# Every keep, prune and migration, one line each. Deletions are rare and the
# one time a saved song went missing there was no record of why.
AUDIT_PATH = pathlib.Path(os.getenv(
    "ACESTEP_SONG_AUDIT", str(pathlib.Path(__file__).resolve().parents[2] / "logs" / "songs.log")))

# A rendered song is scratch until somebody saves it. This is the window in
# which it can still be saved — long enough to listen, think and come back.
KEEP_UNSAVED_SECONDS = float(os.getenv("ACESTEP_KEEP_UNSAVED_SECONDS", "7200"))

# Only files this module writes are ever considered for deletion.
SONG_GLOB = "song-*.mp3"
_SONG_NAME = _re.compile(r"^song-[0-9a-f]{12}\.mp3$")


def ensure_song_dirs() -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    KEPT_DIR.mkdir(parents=True, exist_ok=True)


def audit(action: str, **details) -> None:
    """Append one line to the song audit log. Never raises."""
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        extra = " ".join(f"{k}={v}" for k, v in details.items())
        with open(AUDIT_PATH, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {action} {extra}\n")
    except Exception as e:
        logger.warning("[song] could not write the audit log: %s", e)
    logger.info("[song] %s %s", action, details)


def write_scratch(audio: bytes) -> str:
    """Store a fresh render. Returns its path relative to SONG_ROOT."""
    import uuid

    ensure_song_dirs()
    name = f"song-{uuid.uuid4().hex[:12]}.mp3"
    (SCRATCH_DIR / name).write_bytes(audio)
    return f"scratch/{name}"


def url_for(rel: str) -> str:
    return f"{SONG_URL_PREFIX}/{rel}"


def resolve(rel: str) -> pathlib.Path:
    """A stored song's absolute path. Refuses anything outside the tree.

    *rel* comes from the browser on save, so it is checked rather than
    trusted: one of the two folders, and a name this module would have made.
    """
    rel = (rel or "").strip()
    if rel.startswith(SONG_URL_PREFIX + "/"):
        rel = rel[len(SONG_URL_PREFIX) + 1:]
    parts = pathlib.PurePosixPath(rel).parts
    if len(parts) != 2 or parts[0] not in ("scratch", "kept") \
            or not _SONG_NAME.match(parts[1]):
        raise ValueError(f"not a stored song: {rel!r}")
    return SONG_ROOT / parts[0] / parts[1]


def keep(rel: str) -> str:
    """Move a scratch render into kept/. Returns the new relative path.

    Already kept is fine and returns as-is, so saving the same take twice —
    or renaming it — does not need a second copy.
    """
    src = resolve(rel)
    if src.parent == KEPT_DIR:
        if not src.is_file():
            raise FileNotFoundError(rel)
        return f"kept/{src.name}"
    if not src.is_file():
        raise FileNotFoundError(rel)
    ensure_song_dirs()
    dst = KEPT_DIR / src.name
    src.replace(dst)                    # same filesystem: atomic
    audit("keep", file=src.name)
    return f"kept/{src.name}"


def prune_unsaved(song_dir: pathlib.Path | None = None, keep: set[str] = frozenset(),
                  max_age: float | None = None) -> list[str]:
    """Delete scratch renders older than the grace period. Returns the names.

    Only ever looks in scratch/ — saved songs are not there to be found.
    *keep* is a second guard: any name a composition still points at is left
    alone even in scratch, in case a row was written before its file moved.
    """
    song_dir = SCRATCH_DIR if song_dir is None else song_dir
    if max_age is None:
        max_age = KEEP_UNSAVED_SECONDS
    kept_names = {pathlib.PurePath(k).name for k in keep}
    removed: list[str] = []
    now = time.time()
    if not song_dir.is_dir():
        return removed
    for f in sorted(song_dir.glob(SONG_GLOB)):
        if f.name in kept_names:
            continue
        try:
            age = now - f.stat().st_mtime
        except OSError:
            continue
        if age < max_age:
            continue
        try:
            f.unlink()
            removed.append(f.name)
            audit("prune", file=f.name, age_hours=f"{age / 3600:.1f}")
        except OSError as e:
            logger.warning("[song] could not remove %s: %s", f.name, e)
    return removed


async def migrate_legacy(db, legacy_dir: pathlib.Path) -> int:
    """Move songs saved before v3.2 out of static/generated into kept/.

    Each file is copied, verified by checksum, the database repointed, and
    only then is the old copy removed — so at every instant at least one
    verified copy exists and the row points at a real file. Returns how many
    files moved.
    """
    import hashlib
    import shutil

    def digest(path: pathlib.Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    cur = await db.execute(
        "SELECT DISTINCT audio_path FROM compositions "
        "WHERE audio_path != '' AND audio_path NOT LIKE '%/%'")
    names = [r[0] for r in await cur.fetchall()]
    moved = 0
    for name in names:
        if not _SONG_NAME.match(name):
            logger.warning("[song] not migrating unexpected audio_path %r", name)
            continue
        src, dst = legacy_dir / name, KEPT_DIR / name
        if not dst.is_file():
            if not src.is_file():
                audit("migrate-missing", file=name)
                logger.warning("[song] saved song %s is missing from %s",
                               name, legacy_dir)
                continue
            ensure_song_dirs()
            shutil.copy2(src, dst)
        if src.is_file() and digest(src) != digest(dst):
            logger.error("[song] copy of %s does not match; leaving both", name)
            continue
        await db.execute("UPDATE compositions SET audio_path = ? "
                         "WHERE audio_path = ?", (f"kept/{name}", name))
        await db.commit()
        if src.is_file():
            src.unlink()
        audit("migrate", file=name, to="kept")
        moved += 1
    return moved
