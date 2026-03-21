import sys
import os

# Disable CrewAI interactive trace prompt that blocks the server
os.environ.setdefault('CREWAI_TRACING_ENABLED', 'false')

import io
import re
import json
import uuid
import asyncio
import queue
import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import aiosqlite

from researcher.crew import (
    ResearchCrew,
    generate_image as _generate_image_tool,
    generate_ai_image as _generate_ai_image_tool,
    preload_sd,
)

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
STATIC_DIR = Path(__file__).parent / "static"
DB_PATH = Path(__file__).parent / "data" / "sessions.db"
GENERATED_DIR = STATIC_DIR / "generated"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Strip ANSI escape codes and box-drawing decorations from captured output
_ansi_re = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')
_box_re = re.compile(r'[╭╮╰╯│─┌┐└┘├┤┬┴┼]+')


def _clean_line(text: str) -> str:
    """Clean a single line of stdout output."""
    text = _ansi_re.sub('', text)
    text = _box_re.sub('', text)
    return text.strip()


class _QueueWriter:
    """Captures writes to stdout and puts cleaned lines into a queue."""
    encoding = 'utf-8'

    def __init__(self, q: queue.Queue):
        self._q = q
        self._buf = ''

    def write(self, s):
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            cleaned = _clean_line(line)
            if cleaned:
                self._q.put(cleaned)

    def flush(self):
        if self._buf:
            cleaned = _clean_line(self._buf)
            if cleaned:
                self._q.put(cleaned)
            self._buf = ''

    def isatty(self):
        return False

    def fileno(self):
        raise OSError('not a real file')


# --- Pydantic models ---

class ChatRequest(BaseModel):
    message: str
    history: list = []

class ModelRequest(BaseModel):
    model: str

class SessionSaveRequest(BaseModel):
    name: str
    messages: list

class AskRequest(BaseModel):
    topic: str


# --- Whisper speech-to-text (lazy-loaded, CPU-only to avoid VRAM contention) ---

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                import whisper
                logger.info("Loading Whisper base model (CPU)…")
                _whisper_model = whisper.load_model("base", device="cpu")
                logger.info("Whisper model ready.")
    return _whisper_model


# --- Lifespan: initialise app.state, SQLite, SD preload ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # -- Crew / model init --
    model = os.getenv("MODEL", "ollama/qwen3.5:9b")
    app.state.current_model = model
    app.state.crew_instance = ResearchCrew(model=model).crew()
    app.state.crew_semaphore = asyncio.Semaphore(1)   # serialise crew runs

    # -- SQLite session database --
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            model       TEXT DEFAULT '',
            messages    TEXT NOT NULL DEFAULT '[]'
        )
    """)
    await db.commit()

    # Migrate any legacy JSON session files
    legacy_dir = Path(__file__).parent / "data" / "sessions"
    if legacy_dir.exists():
        migrated = 0
        for f in legacy_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                await db.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(id, name, created_at, updated_at, model, messages) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (data["id"], data["name"], data["created_at"],
                     data.get("updated_at", data["created_at"]),
                     data.get("model", ""),
                     json.dumps(data.get("messages", [])))
                )
                migrated += 1
            except (json.JSONDecodeError, KeyError):
                continue
        if migrated:
            await db.commit()
            for f in legacy_dir.glob("*.json"):
                f.unlink()
            try:
                legacy_dir.rmdir()
            except OSError:
                pass
            logger.info("Migrated %d JSON session(s) to SQLite", migrated)

    app.state.db = db

    # Warm up Stable Diffusion in background
    preload_sd()

    yield

    # -- Shutdown --
    await app.state.db.close()


app = FastAPI(lifespan=lifespan)


# --- Chat UI ---

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/info")
async def info(request: Request):
    return {"model": request.app.state.current_model}


@app.get("/models")
async def list_models(request: Request):
    """Query Ollama for available local models."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        models = [
            {
                "name": m["name"],
                "size_gb": round(m.get("size", 0) / 1e9, 1),
                "family": m.get("details", {}).get("family", ""),
                "params": m.get("details", {}).get("parameter_size", ""),
            }
            for m in data.get("models", [])
        ]
        return {"models": models, "current": request.app.state.current_model}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {e}")


@app.post("/model")
async def switch_model(req: ModelRequest, request: Request):
    """Switch the active LLM model."""
    new_model = req.model if req.model.startswith("ollama/") else f"ollama/{req.model}"
    try:
        request.app.state.crew_instance = ResearchCrew(model=new_model).crew()
        request.app.state.current_model = new_model
        return {"model": request.app.state.current_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {e}")


# --- Post-processing helpers ---

_img_md_re = re.compile(r'!\[[^\]]*\]\(/static/generated/[a-f0-9]+\.png\)')
_orphan_re = re.compile(
    r'Action:\s*(generate_?(?:ai_?)?image)\s*\n\s*Action\s*Input:\s*(\{.*\}|.+)',
    re.DOTALL | re.IGNORECASE,
)


def _postprocess(response_text: str, verbose_log: str) -> str:
    """Clean up crew output: rescue orphan tool calls, validate images, strip noise."""
    # Orphan rescue — run tool server-side if LLM described action but never executed
    orphan = _orphan_re.search(response_text)
    if orphan:
        try:
            tool_name = orphan.group(1).lower().replace(' ', '').replace('_', '')
            raw_input = orphan.group(2).strip()
            if 'ai' in tool_name:
                prompt = raw_input.strip('"\'')
                try:
                    parsed = json.loads(raw_input)
                    prompt = parsed.get('prompt', raw_input)
                except (json.JSONDecodeError, TypeError):
                    pass
                response_text = _generate_ai_image_tool.run(prompt)
            else:
                parsed = json.loads(raw_input)
                if 'instructions' in parsed:
                    iv = parsed['instructions']
                    if isinstance(iv, dict):
                        iv = json.dumps(iv)
                    response_text = _generate_image_tool.run(iv)
                else:
                    response_text = _generate_image_tool.run(raw_input)
        except Exception as exc:
            sys.stderr.write(f"[orphan-rescue] Error: {exc}\n")
            sys.stderr.flush()

    # Pull images from verbose log if response has none
    if '/static/generated/' not in response_text:
        images_in_log = _img_md_re.findall(verbose_log)
        if images_in_log:
            response_text += '\n\n' + '\n'.join(images_in_log)

    # Validate that referenced images actually exist on disk
    def _validate_img(match):
        rel = match.group(1).replace('/static/', '', 1)
        return match.group(0) if (STATIC_DIR / rel).exists() else ''

    response_text = re.sub(
        r'!\[[^\]]*\]\((/static/generated/[^)]+)\)',
        _validate_img, response_text,
    )

    response_text = re.sub(r'<result>\s*</result>', '', response_text)
    response_text = re.sub(
        r'Thought:.*?Action:.*?Action Input:.*?$',
        '', response_text, flags=re.DOTALL,
    )
    response_text = response_text.strip()

    if not response_text:
        images_in_log = _img_md_re.findall(verbose_log)
        valid = [
            img for img in images_in_log
            if (STATIC_DIR / img.split('(')[1].rstrip(')').replace('/static/', '', 1)).exists()
        ]
        if valid:
            response_text = valid[-1]
        else:
            response_text = 'The agent could not complete the request. Please try rephrasing.'

    return response_text


def _extract_usage(result) -> dict:
    if hasattr(result, 'token_usage') and result.token_usage:
        tu = result.token_usage
        return {
            "total_tokens": getattr(tu, 'total_tokens', 0),
            "prompt_tokens": getattr(tu, 'prompt_tokens', 0),
            "completion_tokens": getattr(tu, 'completion_tokens', 0),
        }
    return {}


# --- /chat  SSE streaming endpoint ---

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    # Build context from conversation history
    context_lines = []
    _img_ctx_re = re.compile(r'!\[[^\]]*\]\(/static/generated/[^)]+\)')
    for msg in req.history[-4:]:
        role = msg.get('role', 'user')
        text = msg.get('text', '')
        if role == 'user':
            context_lines.append(f"User: {text}")
        elif role == 'assistant':
            text = _img_ctx_re.sub('[image was shown]', text)
            short = text[:120] + ('...' if len(text) > 120 else '')
            context_lines.append(f"Assistant: {short}")

    if context_lines:
        topic = ("Previous conversation:\n"
                 + "\n".join(context_lines)
                 + "\n\nNew request: " + req.message)
    else:
        topic = req.message

    inputs = {'topic': topic}
    crew = request.app.state.crew_instance
    semaphore = request.app.state.crew_semaphore

    q: queue.Queue = queue.Queue()

    def _run_crew():
        writer = _QueueWriter(q)
        old_stdout, old_stdin = sys.stdout, sys.stdin
        sys.stdout = writer
        sys.stdin = io.StringIO('N\n')
        try:
            return crew.kickoff(inputs=inputs)
        finally:
            writer.flush()
            sys.stdout = old_stdout
            sys.stdin = old_stdin

    def _sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def event_stream():
        async with semaphore:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, _run_crew)
            all_lines: list[str] = []

            _skip_history = False

            def _should_show(line: str) -> bool:
                nonlocal _skip_history
                if 'Previous conversation:' in line:
                    _skip_history = True
                    return False
                if _skip_history:
                    if 'New request:' in line:
                        _skip_history = False
                    return False
                return True

            while not future.done():
                await asyncio.sleep(0.15)
                while not q.empty():
                    try:
                        line = q.get_nowait()
                        all_lines.append(line)
                        if _should_show(line):
                            yield _sse("reasoning", line)
                    except queue.Empty:
                        break

            # Drain remaining
            while not q.empty():
                try:
                    line = q.get_nowait()
                    all_lines.append(line)
                    if _should_show(line):
                        yield _sse("reasoning", line)
                except queue.Empty:
                    break

            try:
                result = future.result()
            except Exception as e:
                yield _sse("error", str(e))
                return

            response_text = _postprocess(str(result), '\n'.join(all_lines))
            usage = _extract_usage(result)

            yield _sse("done", {
                "response": response_text,
                "reasoning": all_lines,
                "token_usage": usage,
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Session management (SQLite) ---

_SESSION_ID_RE = re.compile(r'^[a-f0-9]{8}$')


def _validate_sid(session_id: str):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")


@app.get("/sessions")
async def list_sessions(request: Request):
    db = request.app.state.db
    cursor = await db.execute(
        "SELECT id, name, created_at, updated_at, model, messages "
        "FROM sessions ORDER BY updated_at DESC"
    )
    rows = await cursor.fetchall()
    sessions = []
    for row in rows:
        try:
            msgs = json.loads(row["messages"])
        except (json.JSONDecodeError, TypeError):
            msgs = []
        sessions.append({
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": len(msgs),
            "model": row["model"] or "",
        })
    return {"sessions": sessions}


@app.post("/sessions")
async def save_session(req: SessionSaveRequest, request: Request):
    db = request.app.state.db
    sid = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO sessions (id, name, created_at, updated_at, model, messages) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (sid, req.name, now, now,
         request.app.state.current_model, json.dumps(req.messages)),
    )
    await db.commit()
    return {"id": sid, "name": req.name}


@app.get("/sessions/{session_id}")
async def load_session(session_id: str, request: Request):
    _validate_sid(session_id)
    db = request.app.state.db
    cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "model": row["model"],
        "messages": json.loads(row["messages"]),
    }


@app.put("/sessions/{session_id}")
async def update_session(session_id: str, req: SessionSaveRequest, request: Request):
    _validate_sid(session_id)
    db = request.app.state.db
    cursor = await db.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if not await cursor.fetchone():
        raise HTTPException(status_code=404, detail="Session not found")
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "UPDATE sessions SET name=?, messages=?, updated_at=?, model=? WHERE id=?",
        (req.name, json.dumps(req.messages), now,
         request.app.state.current_model, session_id),
    )
    await db.commit()
    return {"id": session_id, "name": req.name}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    _validate_sid(session_id)
    db = request.app.state.db
    cursor = await db.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if not await cursor.fetchone():
        raise HTTPException(status_code=404, detail="Session not found")
    await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    await db.commit()
    return {"deleted": session_id}


# --- /ask — structured API for automation ---

@app.post("/ask")
async def ask(req: AskRequest, request: Request):
    """Programmatic API endpoint. POST JSON with {topic: "..."}. Returns structured response."""
    crew = request.app.state.crew_instance
    semaphore = request.app.state.crew_semaphore

    q: queue.Queue = queue.Queue()

    def _run():
        writer = _QueueWriter(q)
        old_stdout, old_stdin = sys.stdout, sys.stdin
        sys.stdout = writer
        sys.stdin = io.StringIO('N\n')
        try:
            return crew.kickoff(inputs={'topic': req.topic})
        finally:
            writer.flush()
            sys.stdout = old_stdout
            sys.stdin = old_stdin

    async with semaphore:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, _run)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    reasoning = []
    while not q.empty():
        try:
            reasoning.append(q.get_nowait())
        except queue.Empty:
            break

    return {
        "status": "success",
        "response": str(result),
        "reasoning": reasoning,
        "token_usage": _extract_usage(result),
        "model": request.app.state.current_model,
    }


# --- /transcribe — Whisper voice-to-text ---

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "en"):
    """Transcribe audio using Whisper (CPU). Accepts any ffmpeg-supported audio format."""
    # Validate language code (2-letter ISO 639-1)
    lang = language.strip().lower()[:2] if language else "en"
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        loop = asyncio.get_running_loop()

        def _transcribe():
            model = _get_whisper()
            return model.transcribe(tmp_path, language=lang)

        result = await loop.run_in_executor(None, _transcribe)
        return {"text": result.get("text", "").strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# --- /tts — Server-side text-to-speech via edge-tts ---

_tts_voices_cache: list[dict] | None = None

@app.get("/tts/voices")
async def tts_voices():
    """Return available edge-tts voices (cached after first call)."""
    return await _ensure_tts_voices()


async def _ensure_tts_voices() -> list[dict]:
    """Return cached voice list, fetching if needed."""
    global _tts_voices_cache
    if _tts_voices_cache is None:
        import edge_tts
        voices = await edge_tts.list_voices()
        _tts_voices_cache = [
            {"name": v["ShortName"], "gender": v.get("Gender", ""), "locale": v.get("Locale", "")}
            for v in voices
        ]
    return _tts_voices_cache


def _detect_lang(text: str) -> str:
    """Detect the language of a text segment, defaulting to 'en'."""
    from langdetect import detect, LangDetectException
    # Common false positives from langdetect on short English text
    _FALSE_POS = {'no', 'so', 'sw', 'tl', 'cy', 'af', 'da'}
    try:
        lang = detect(text)
        if lang in _FALSE_POS and len(text.split()) < 6:
            return "en"
        return lang
    except LangDetectException:
        return "en"


# Regex to find quoted or emphasised foreign phrases in LLM output.
# Matches: "phrase", 'phrase', «phrase», *phrase*, **phrase**, _phrase_
_FOREIGN_PHRASE_RE = re.compile(
    r'["\u201c\u201d«]([^"\u201c\u201d»]{2,})["\u201c\u201d»]'
    r"|'([^']{2,})'"
    r'|\*{1,2}([^*]{2,})\*{1,2}'
    r'|_([^_]{2,})_'
)


def _split_multilingual(text: str, base_lang: str = "en") -> list[tuple[str, str]]:
    """Split *text* into ``(lang, segment)`` pairs so that each segment can be
    spoken with the correct TTS voice.

    Strategy (designed to **minimise** fragmentation):

    1.  Detect the dominant language of the **whole text**.  If it differs from
        *base_lang*, return the entire text as that language — no splitting.
    2.  If the overall language equals *base_lang* (i.e. the text is mostly in
        the base language with some foreign content mixed in):
        a.  Extract quoted / emphasised foreign phrases.
        b.  Split remaining text into sentences, detect each, and label them.
        c.  **Merge** consecutive sentences that share the same language into
            single contiguous blocks so we never pass tiny fragments to the
            TTS engine.
    """
    from langdetect import detect, LangDetectException

    # ── tunables ──────────────────────────────────────────────────
    _FALSE_LANGS = {'no', 'so', 'sw', 'tl', 'cy', 'af', 'da'}
    _TRUSTED_LANGS = {
        'fr', 'de', 'es', 'it', 'pt', 'ro', 'ru', 'zh-cn', 'zh-tw',
        'ja', 'ko', 'ar', 'nl', 'pl', 'uk', 'el', 'tr', 'sv', 'cs', 'hu',
    }

    def _safe_detect(s: str) -> str:
        try:
            lang = detect(s)
            if lang in _FALSE_LANGS and len(s.split()) < 8:
                return base_lang
            return lang
        except LangDetectException:
            return base_lang

    # ── Step 1: whole-text detection ──────────────────────────────
    # If the whole text is in a single foreign language, return it as one
    # block.  But first do a quick sanity check: split into sentences and
    # see if any sentence detects as base_lang.  If so, it's mixed text —
    # fall through to the per-sentence path.
    overall = _safe_detect(text)
    if overall != base_lang and overall in _TRUSTED_LANGS:
        # Quick check: are there base-lang sentences mixed in?
        _quick_bounds = list(re.finditer(r'(?<=[.!?])\s+', text))
        _q_starts = [0] + [m.end() for m in _quick_bounds]
        _q_ends = [m.start() for m in _quick_bounds] + [len(text)]
        _q_sents = [text[s:e].strip() for s, e in zip(_q_starts, _q_ends) if text[s:e].strip()]
        has_base = any(
            _safe_detect(s) == base_lang and len(s.split()) >= 3
            for s in _q_sents
        )
        if not has_base:
            return [(overall, text)]

    # ── Step 2a: extract quoted / emphasised foreign phrases ──────
    foreign_spans: list[tuple[int, int, str]] = []
    for m in _FOREIGN_PHRASE_RE.finditer(text):
        phrase = next(g for g in m.groups() if g is not None)
        if len(phrase.split()) < 2:
            continue
        lang = _safe_detect(phrase)
        if lang != base_lang and lang not in _FALSE_LANGS:
            foreign_spans.append((m.start(), m.end(), lang))

    # ── Step 2b: sentence-level detection ─────────────────────────
    sent_boundaries = list(re.finditer(r'(?<=[.!?])\s+|(?<=[:;])\s+', text))
    sent_starts = [0] + [m.end() for m in sent_boundaries]
    sent_ends = [m.start() for m in sent_boundaries] + [len(text)]
    sentences = [
        (s, e, text[s:e])
        for s, e in zip(sent_starts, sent_ends)
        if text[s:e].strip()
    ]

    # Label every sentence with a language
    labelled: list[tuple[int, int, str, str]] = []  # (start, end, lang, text)
    for start, end, sent in sentences:
        # If the sentence overlaps a quoted foreign span, keep the quote's
        # language for the whole sentence to avoid fragmenting it.
        overlap_lang = None
        for fs, fe, fl in foreign_spans:
            if not (fe <= start or fs >= end):
                overlap_lang = fl
                break
        if overlap_lang:
            labelled.append((start, end, overlap_lang, sent))
            continue

        stripped = sent.strip()
        wc = len(stripped.split())
        lang = _safe_detect(stripped)
        if lang == base_lang or lang in _FALSE_LANGS:
            labelled.append((start, end, base_lang, sent))
        elif lang in _TRUSTED_LANGS and wc >= 2:
            labelled.append((start, end, lang, sent))
        elif wc >= 4:
            labelled.append((start, end, lang, sent))
        else:
            # Very short text — unreliable detection, default to base
            labelled.append((start, end, base_lang, sent))

    if not labelled:
        return [(base_lang, text)]

    # Short isolated foreign fragments (< 3 words) that don't match any
    # neighbour are likely misdetections.  Absorb them into the preceding
    # block's language.
    for i in range(1, len(labelled)):
        s, e, lang, stxt = labelled[i]
        if lang == base_lang:
            continue
        wc = len(stxt.strip().split())
        if wc < 3 and lang != labelled[i - 1][2]:
            labelled[i] = (s, e, labelled[i - 1][2], stxt)

    # ── Step 2c: merge consecutive same-language sentences ────────
    merged: list[tuple[str, str]] = []
    cur_lang = labelled[0][2]
    cur_start = labelled[0][0]
    cur_end = labelled[0][1]

    for start, end, lang, _ in labelled[1:]:
        if lang == cur_lang:
            cur_end = end           # extend current block
        else:
            block = text[cur_start:cur_end].strip()
            if block:
                merged.append((cur_lang, block))
            cur_lang = lang
            cur_start = start
            cur_end = end

    # flush last block
    block = text[cur_start:cur_end].strip()
    if block:
        merged.append((cur_lang, block))

    # If every block ended up as base_lang, return the whole text as one piece
    if all(lang == base_lang for lang, _ in merged):
        return [(base_lang, text)]

    return merged


def _pick_voice_for_lang(lang: str, base_voice: str, voices: list[dict]) -> str:
    """Pick the best edge-tts voice for a detected language, preserving the gender
    of the user's selected base voice."""
    # Determine gender of the base voice
    base_gender = "Female"
    for v in voices:
        if v["name"] == base_voice:
            base_gender = v["gender"]
            break

    lang_prefix = lang.lower()[:2]

    # If the base voice already matches the language, keep it
    for v in voices:
        if v["name"] == base_voice and v["locale"].lower().startswith(lang_prefix):
            return base_voice

    candidates = [v for v in voices if v["locale"].lower().startswith(lang_prefix)]
    if not candidates:
        return base_voice  # no voice for this language; fall back

    # Prefer same gender, then Neural voices (higher quality)
    gender_match = [v for v in candidates if v["gender"] == base_gender]
    pool = gender_match or candidates
    neural = [v for v in pool if "Neural" in v["name"] and "Multilingual" not in v["name"]]
    return (neural or pool)[0]["name"]


class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-AriaNeural"


@app.post("/tts/speak")
async def tts_speak(req: TTSRequest):
    """Generate speech audio from text using edge-tts.

    Automatically detects language per sentence/phrase and switches to a
    matching voice so foreign words are pronounced correctly.
    """
    import edge_tts

    clean = req.text.strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Empty text")

    voices = await _ensure_tts_voices()

    # Determine the base language from the user's selected voice
    base_lang = "en"
    for v in voices:
        if v["name"] == req.voice:
            base_lang = v["locale"][:2].lower()
            break

    segments = _split_multilingual(clean, base_lang)
    logger.warning("[TTS] req.voice=%s base_lang=%s segments=%d text=%.80s",
                   req.voice, base_lang, len(segments), clean)

    buf = io.BytesIO()
    for lang, text in segments:
        voice = _pick_voice_for_lang(lang, req.voice, voices)
        logger.warning("[TTS] segment lang=%s voice=%s text=%.60s", lang, voice, text)
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

    if buf.tell() == 0:
        raise HTTPException(status_code=500, detail="TTS produced no audio")

    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


# --- Static files & server entry point ---

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    # 120-second timeout for 5070 Ti to process long searches
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)
