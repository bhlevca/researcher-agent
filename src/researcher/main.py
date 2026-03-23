import sys
import os

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

from researcher.upsonicai import (
    ResearchCrew,
    generate_image as _generate_image_tool,
    generate_ai_image as _generate_ai_image_tool,
    preload_sd,
    UpsonicResult,
)
from researcher.auth import (
    register_auth_routes,
    init_users_table,
    migrate_sessions_table,
    get_current_user,
    get_optional_user,
)
from researcher.ingestion import (
    register_ingestion_routes,
    init_files_table,
    get_file_context,
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


_think_re = re.compile(r'<think>[\s\S]*?</think>\s*')
_think_open_re = re.compile(r'</?think>')
_think_extract_re = re.compile(r'<think>([\s\S]*?)</think>')


def _clean_line(text: str) -> str:
    """Clean a single line of stdout output."""
    text = _ansi_re.sub('', text)
    text = _box_re.sub('', text)
    text = _think_open_re.sub('', text)
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
    file_ids: list[str] = []

class ModelRequest(BaseModel):
    model: str

class SessionSaveRequest(BaseModel):
    name: str
    messages: list

class AskRequest(BaseModel):
    topic: str
    file_ids: list[str] = []

class ContinueRequest(BaseModel):
    original_query: str
    partial_response: str
    file_ids: list[str] = []


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
    # -- Agent / model init --
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

    # Auth tables
    await init_users_table(db)
    await migrate_sessions_table(db)

    # Files table
    await init_files_table(db)

    # Warm up Stable Diffusion in background
    preload_sd()

    yield

    # -- Shutdown --
    await app.state.db.close()


app = FastAPI(lifespan=lifespan)
register_auth_routes(app)
register_ingestion_routes(app)


# --- Chat UI ---

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/info")
async def info(request: Request):
    user = await get_optional_user(request)
    if user:
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT model FROM users WHERE id = ?", (user["id"],)
        )
        row = await cursor.fetchone()
        model = (row[0] if row and row[0] else None) or request.app.state.current_model
    else:
        model = request.app.state.current_model
    return {"model": model, "user": user}


@app.get("/models")
async def list_models(request: Request):
    """Query Ollama for available local models."""
    await get_current_user(request)  # require login
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
    """Switch the active LLM model. Stores per-user preference."""
    user = await get_current_user(request)  # require login
    new_model = req.model if req.model.startswith("ollama/") else f"ollama/{req.model}"
    try:
        request.app.state.crew_instance = ResearchCrew(model=new_model).crew()
        request.app.state.current_model = new_model

        db = request.app.state.db
        await db.execute(
            "UPDATE users SET model = ? WHERE id = ?",
            (new_model, user["id"]),
        )
        await db.commit()

        return {"model": new_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {e}")


# --- Post-processing helpers ---

_img_md_re = re.compile(r'!\[[^\]]*\]\(/static/generated/[a-f0-9]+\.png\)')
_orphan_re = re.compile(
    r'Action:\s*(generate_?(?:ai_?)?image)\s*\n\s*Action\s*Input:\s*(\{.*\}|.+)',
    re.DOTALL | re.IGNORECASE,
)


def _postprocess(response_text: str, verbose_log: str) -> str:
    """Clean up agent output: rescue orphan tool calls, validate images, strip noise."""
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
                response_text = _generate_ai_image_tool(prompt)
            else:
                parsed = json.loads(raw_input)
                if 'instructions' in parsed:
                    iv = parsed['instructions']
                    if isinstance(iv, dict):
                        iv = json.dumps(iv)
                    response_text = _generate_image_tool(iv)
                else:
                    response_text = _generate_image_tool(raw_input)
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
    # Strip LLM thinking tags (qwen3, deepseek-r1, etc.)
    response_text = re.sub(r'<think>[\s\S]*?</think>\s*', '', response_text)
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


# Detect incomplete agent output (hit max iterations or malformed finish)
_INCOMPLETE_MARKERS = [
    'Maximum iterations reached',
    'Invalid response from LLM call',
    'Please try rephrasing',
]


def _is_incomplete(response_text: str, verbose_log: str) -> bool:
    """Return True if the agent output looks like it was cut short."""
    for marker in _INCOMPLETE_MARKERS:
        if marker in response_text or marker in verbose_log:
            return True
    # Response ends with an orphaned Action/Action Input (tool never executed)
    if re.search(r'Action\s*Input\s*:\s*\{[^}]*\}\s*$', response_text):
        return True
    return False


def _extract_thinking(raw_text: str) -> list[str]:
    """Pull reasoning from <think> blocks before they get stripped."""
    lines = []
    for m in _think_extract_re.finditer(raw_text):
        thought = m.group(1).strip()
        if thought:
            for line in thought.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
    return lines


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
    user = await get_current_user(request)  # require login

    # Prepend attached file contents
    file_context = ""
    if req.file_ids:
        db = request.app.state.db
        file_context = await get_file_context(db, user["id"], req.file_ids)

    # Build context from conversation history.
    # Include ALL messages so the LLM never "forgets" earlier requests.
    # User messages are always kept in full; assistant messages are
    # progressively trimmed for older entries to stay within context budget.
    context_lines = []
    _img_ctx_re = re.compile(r'!\[[^\]]*\]\(/static/generated/[^)]+\)')
    history = req.history or []
    n = len(history)
    for idx, msg in enumerate(history):
        role = msg.get('role', 'user')
        text = msg.get('text', '')
        if role == 'user':
            # Always include user messages in full
            context_lines.append(f"User: {text}")
        elif role == 'assistant':
            text = _img_ctx_re.sub('[image was shown]', text)
            # Recent assistant messages get more space; older ones less.
            # Last 4 messages: 2000 chars, next 6: 600 chars, older: 200 chars.
            if idx >= n - 4:
                limit = 2000
            elif idx >= n - 10:
                limit = 600
            else:
                limit = 200
            short = text[:limit] + ('...' if len(text) > limit else '')
            context_lines.append(f"Assistant: {short}")

    if context_lines:
        topic = (file_context
                 + "Previous conversation:\n"
                 + "\n".join(context_lines)
                 + "\n\nNew request: " + req.message)
    else:
        topic = file_context + req.message

    inputs = {'topic': topic}
    crew = request.app.state.crew_instance
    semaphore = request.app.state.crew_semaphore

    # Queue for streaming events from the blocking generator
    eq: queue.Queue = queue.Queue()

    def _run_stream():
        """Run stream_kickoff in a worker thread, push events into eq."""
        try:
            for kind, data in crew.stream_kickoff(inputs=inputs):
                eq.put((kind, data))
        except Exception as exc:
            eq.put(('error', str(exc)))

    def _sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def event_stream():
        async with semaphore:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, _run_stream)
            all_reasoning: list[str] = []
            result = None
            # Buffer for accumulating thinking token deltas into full lines
            think_buf = ""

            def _flush_thinking():
                """Flush complete lines from the thinking buffer."""
                nonlocal think_buf
                lines_out = []
                while '\n' in think_buf:
                    line, think_buf = think_buf.split('\n', 1)
                    line = line.strip()
                    if line:
                        lines_out.append(line)
                return lines_out

            def _flush_thinking_all():
                """Flush everything remaining in the buffer (partial line too)."""
                nonlocal think_buf
                lines_out = _flush_thinking()
                remainder = think_buf.strip()
                if remainder:
                    lines_out.append(remainder)
                    think_buf = ""
                return lines_out

            while True:
                await asyncio.sleep(0.05)
                # Drain all queued events
                while not eq.empty():
                    try:
                        kind, data = eq.get_nowait()
                    except queue.Empty:
                        break

                    if kind == 'thinking':
                        # Accumulate token deltas; emit only complete lines
                        think_buf += str(data)
                        for line in _flush_thinking():
                            all_reasoning.append(line)
                            yield _sse("reasoning", line)
                    elif kind in ('tool_call', 'tool_result', 'step', 'done', 'error'):
                        # Flush any pending thinking text before non-thinking events
                        for line in _flush_thinking_all():
                            all_reasoning.append(line)
                            yield _sse("reasoning", line)
                        if kind == 'tool_call':
                            line = f"🔧 Tool: {data}"
                            all_reasoning.append(line)
                            yield _sse("reasoning", line)
                        elif kind == 'tool_result':
                            line = f"   ↳ {data}"
                            all_reasoning.append(line)
                            yield _sse("reasoning", line)
                        elif kind == 'step':
                            line = f"▶ {data}"
                            all_reasoning.append(line)
                            yield _sse("reasoning", line)
                        elif kind == 'error':
                            yield _sse("error", data)
                            return
                        elif kind == 'done':
                            result = data  # UpsonicResult
                    # text_delta — don't send as reasoning; it's the final answer text

                if result is not None:
                    break
                if future.done():
                    # Flush remaining thinking buffer
                    for line in _flush_thinking_all():
                        all_reasoning.append(line)
                        yield _sse("reasoning", line)
                    # If thread finished but we didn't get a 'done' event, check for exceptions
                    try:
                        future.result()
                    except Exception as e:
                        yield _sse("error", str(e))
                        return
                    break

            if result is None:
                yield _sse("error", "Agent stream ended without producing a result")
                return

            raw = str(result)
            response_text = _postprocess(raw, '\n'.join(all_reasoning))
            usage = _extract_usage(result)
            incomplete = _is_incomplete(response_text, '\n'.join(all_reasoning))

            yield _sse("done", {
                "response": response_text,
                "reasoning": all_reasoning,
                "token_usage": usage,
                "incomplete": incomplete,
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- /chat/continue  LLM-direct continuation (no agent tools) ---

@app.post("/chat/continue")
async def chat_continue(req: ContinueRequest, request: Request):
    """Continue an incomplete agent response by calling the LLM directly."""
    import litellm

    user = await get_current_user(request)  # require login

    # Optional file context
    file_context = ""
    if req.file_ids:
        db = request.app.state.db
        file_context = await get_file_context(db, user["id"], req.file_ids)

    model = request.app.state.current_model
    # Strip "ollama/" prefix for litellm's ollama_chat provider
    if model.startswith("ollama/"):
        litellm_model = "ollama_chat/" + model[len("ollama/"):]
    else:
        litellm_model = model

    system_prompt = (
        "You are a knowledgeable research assistant. "
        "The user asked a question and a previous agent produced a partial answer "
        "but ran out of processing steps before finishing. "
        "Your job is to CONTINUE and COMPLETE the answer from where it left off. "
        "Do NOT repeat content that was already produced — only produce the REMAINING parts. "
        "Continue seamlessly from the last line of the partial answer. "
        "Use the same formatting style (markdown tables, headings, etc.) as the partial answer."
    )
    user_prompt = (
        file_context
        + "Original user request:\n" + req.original_query
        + "\n\n--- PARTIAL ANSWER (already shown to user) ---\n"
        + req.partial_response
        + "\n--- END OF PARTIAL ANSWER ---\n\n"
        "Continue from where the partial answer left off. "
        "Only output the NEW content that completes the answer. "
        "Do NOT repeat any of the partial answer above."
    )

    loop = asyncio.get_running_loop()
    def _call_llm():
        return litellm.completion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            api_base="http://localhost:11434",
            num_retries=0,
            temperature=0.1,
        )

    try:
        logger.info("[chat/continue] Calling LLM model=%s, query_len=%d, partial_len=%d",
                     litellm_model, len(req.original_query), len(req.partial_response))
        response = await loop.run_in_executor(None, _call_llm)
        text = response.choices[0].message.content or ""
        logger.info("[chat/continue] LLM returned %d chars", len(text))
        # Strip thinking tags
        text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text).strip()
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "total_tokens": getattr(response.usage, 'total_tokens', 0),
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
            }
        return {
            "response": text,
            "reasoning": ["Continued via direct LLM call (bypassed agent tools)"],
            "token_usage": usage,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Session management (SQLite) ---

_SESSION_ID_RE = re.compile(r'^[a-f0-9]{8}$')


def _validate_sid(session_id: str):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")


@app.get("/sessions")
async def list_sessions(request: Request):
    user = await get_current_user(request)  # require login
    db = request.app.state.db
    cursor = await db.execute(
        "SELECT id, name, created_at, updated_at, model, messages "
        "FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
        (user["id"],),
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
    user = await get_current_user(request)  # require login
    db = request.app.state.db
    sid = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO sessions (id, name, created_at, updated_at, model, messages, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sid, req.name, now, now,
         request.app.state.current_model, json.dumps(req.messages), user["id"]),
    )
    await db.commit()
    return {"id": sid, "name": req.name}


@app.get("/sessions/{session_id}")
async def load_session(session_id: str, request: Request):
    _validate_sid(session_id)
    user = await get_current_user(request)  # require login
    db = request.app.state.db
    cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    owner = row["user_id"] if "user_id" in row.keys() else ""
    if owner and (not user or user["id"] != owner):
        raise HTTPException(status_code=403, detail="Not your session")
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
    user = await get_current_user(request)  # require login
    db = request.app.state.db
    cursor = await db.execute("SELECT id, user_id FROM sessions WHERE id = ?", (session_id,))
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    owner = row[1] or ""
    if owner and (not user or user["id"] != owner):
        raise HTTPException(status_code=403, detail="Not your session")
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
    user = await get_current_user(request)  # require login
    db = request.app.state.db
    cursor = await db.execute("SELECT id, user_id FROM sessions WHERE id = ?", (session_id,))
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    owner = row[1] or ""
    if owner and (not user or user["id"] != owner):
        raise HTTPException(status_code=403, detail="Not your session")
    await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    await db.commit()
    return {"deleted": session_id}


# --- /ask — structured API for automation ---

@app.post("/ask")
async def ask(req: AskRequest, request: Request):
    """Programmatic API endpoint. POST JSON with {topic: "..."}. Returns structured response."""
    user = await get_current_user(request)  # require login

    # Prepend attached file contents
    file_context = ""
    if req.file_ids:
        db = request.app.state.db
        file_context = await get_file_context(db, user["id"], req.file_ids)

    topic = file_context + req.topic
    crew = request.app.state.crew_instance
    semaphore = request.app.state.crew_semaphore

    q: queue.Queue = queue.Queue()

    def _run():
        writer = _QueueWriter(q)
        old_stdout = sys.stdout
        sys.stdout = writer
        try:
            return crew.kickoff(inputs={'topic': topic})
        finally:
            writer.flush()
            sys.stdout = old_stdout

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

    raw = str(result)
    # Extract reasoning from AgentRunOutput thinking parts
    if hasattr(result, 'reasoning_lines'):
        agent_reasoning = result.reasoning_lines
        if agent_reasoning:
            reasoning.extend(agent_reasoning)
    # Extract <think> reasoning from raw text as fallback
    think_lines = _extract_thinking(raw)
    if think_lines:
        reasoning.extend(think_lines)
    response_text = _postprocess(raw, '\n'.join(reasoning))

    return {
        "status": "success",
        "response": response_text,
        "reasoning": reasoning,
        "token_usage": _extract_usage(result),
        "model": request.app.state.current_model,
    }


# --- /transcribe — Whisper voice-to-text ---

@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...), language: str = "en"):
    """Transcribe audio using Whisper (CPU). Accepts any ffmpeg-supported audio format."""
    await get_current_user(request)  # require login
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


# --- /tts — Server-side text-to-speech (see researcher.tts) ---

from researcher.tts import (                        # noqa: E402
    _detect_lang,
    _split_multilingual,
    _pick_voice_for_lang,
    _FOREIGN_PHRASE_RE,
    register_tts_routes,
)

register_tts_routes(app)


# --- Static files & server entry point ---

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    # 120-second timeout for 5070 Ti to process long searches
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)
