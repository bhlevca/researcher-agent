import sys
import os

# Never write .pyc bytecode files — prevents stale cache bugs after edits
sys.dont_write_bytecode = True

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
import researcher.crew as _crew_mod
from crewai.agents.parser import AgentAction, AgentFinish
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.reasoning_events import (
    AgentReasoningStartedEvent,
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
)
from crewai.events.types.observation_events import (
    StepObservationCompletedEvent,
    PlanRefinementEvent,
    GoalAchievedEarlyEvent,
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
    # -- Crew / model init --
    model = os.getenv("MODEL", "ollama/qwen3.5:9b")
    app.state.current_model = model
    app.state.research_crew = ResearchCrew(model=model)
    app.state.crew_semaphore = asyncio.Semaphore(1)   # serialise crew runs

    # Wipe stale memory from previous server runs so old data cannot
    # contaminate new requests (users save sessions explicitly).
    app.state.research_crew.reset_memory()

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
        request.app.state.research_crew = ResearchCrew(model=new_model)
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


# --- Memory depth toggle: shallow (fast) / deep (thorough) ---

@app.get("/memory-depth")
async def get_memory_depth():
    """Return the current memory recall depth."""
    return {"depth": _crew_mod.memory_depth}


@app.post("/memory-depth")
async def set_memory_depth(request: Request):
    """Switch memory recall depth between 'shallow' and 'deep'."""
    body = await request.json()
    depth = body.get("depth", "shallow")
    if depth not in ("shallow", "deep"):
        raise HTTPException(status_code=400, detail="depth must be 'shallow' or 'deep'")
    _crew_mod.memory_depth = depth
    logger.info("Memory depth set to: %s", depth)
    return {"depth": depth}


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


def _extract_usage(result) -> dict:
    if hasattr(result, 'token_usage') and result.token_usage:
        tu = result.token_usage
        return {
            "total_tokens": getattr(tu, 'total_tokens', 0),
            "prompt_tokens": getattr(tu, 'prompt_tokens', 0),
            "completion_tokens": getattr(tu, 'completion_tokens', 0),
        }
    return {}


# --- Conversation context builder (3-tier) ---

_IMG_CTX_RE = re.compile(r'!\[[^\]]*\]\(/static/generated/[^)]+\)')


def _clean_assistant_text(text: str) -> str:
    """Strip image markdown and thinking tags from assistant text."""
    text = _IMG_CTX_RE.sub('[image]', text)
    text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
    return text.strip()


def _heuristic_shorten(text: str, max_len: int = 400) -> str:
    """Shorten text to max_len, keeping start and end for context."""
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 10
    return text[:half] + " [...] " + text[-half:]


def _summarize_with_llm(messages: list[dict], model: str) -> str:
    """Use litellm to summarize older conversation messages."""
    import litellm
    litellm_model = ("ollama_chat/" + model[len("ollama/"):]
                     if model.startswith("ollama/") else model)
    conversation_text = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: "
        f"{_clean_assistant_text(m.get('text', ''))[:500]}"
        for m in messages
    )
    try:
        resp = litellm.completion(
            model=litellm_model,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation in 3-5 bullet points. "
                    "Focus on: topics discussed, decisions made, "
                    "pending questions. Be concise.\n\n" + conversation_text
                ),
            }],
            api_base="http://localhost:11434",
            num_retries=0,
            temperature=0.1,
        )
        summary = resp.choices[0].message.content or ""
        summary = re.sub(r'<think>[\s\S]*?</think>\s*', '', summary).strip()
        return summary
    except Exception as e:
        logger.warning("LLM summarization failed: %s", e)
        # Fallback to heuristic
        return "\n".join(
            f"- {'User' if m.get('role') == 'user' else 'Assistant'}: "
            f"{_heuristic_shorten(m.get('text', ''), 200)}"
            for m in messages[-6:]
        )


def _build_conversation_context(history: list[dict], model: str) -> str:
    """Build conversation context with tiered strategy.

    Short  (≤4 messages):  full text for all messages
    Medium (5-10):         last 3 full, earlier heuristic-shortened
    Long   (>10):          last 3 full, earlier LLM-summarized
    """
    if not history:
        return ""

    n = len(history)
    lines = []

    if n <= 4:
        # Short: include everything
        for msg in history:
            role = msg.get('role', 'user')
            text = msg.get('text', '')
            if role == 'assistant':
                text = _clean_assistant_text(text)
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")

    elif n <= 10:
        # Medium: heuristic shorten older, keep last 3 full
        older = history[:-3]
        recent = history[-3:]
        lines.append("[Earlier in conversation]")
        for msg in older:
            role = msg.get('role', 'user')
            text = msg.get('text', '')
            if role == 'assistant':
                text = _clean_assistant_text(text)
            lines.append(
                f"{'User' if role == 'user' else 'Assistant'}: "
                f"{_heuristic_shorten(text)}"
            )
        lines.append("[Recent messages]")
        for msg in recent:
            role = msg.get('role', 'user')
            text = msg.get('text', '')
            if role == 'assistant':
                text = _clean_assistant_text(text)
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")

    else:
        # Long: LLM-summarize older, keep last 3 full
        older = history[:-3]
        recent = history[-3:]
        summary = _summarize_with_llm(older, model)
        lines.append("[Conversation summary]")
        lines.append(summary)
        lines.append("[Recent messages]")
        for msg in recent:
            role = msg.get('role', 'user')
            text = msg.get('text', '')
            if role == 'assistant':
                text = _clean_assistant_text(text)
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")

    return "Previous conversation:\n" + "\n".join(lines)


# --- /chat  SSE streaming endpoint ---

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    user = await get_current_user(request)  # require login

    # Prepend attached file contents
    file_context = ""
    if req.file_ids:
        db = request.app.state.db
        file_context = await get_file_context(db, user["id"], req.file_ids)

    # Build context from conversation history (3-tier strategy)
    model = request.app.state.current_model
    conv_context = _build_conversation_context(req.history, model)

    if conv_context:
        topic = file_context + conv_context + "\n\nNew request: " + req.message
    else:
        topic = file_context + req.message

    research_crew = request.app.state.research_crew
    crew = research_crew.build_crew(topic)
    semaphore = request.app.state.crew_semaphore

    q: queue.Queue = queue.Queue()

    def _make_step_callback(q_ref):
        """Return a callback that puts structured reasoning into the queue."""
        def _step_cb(step):
            sys.stderr.write(f"[STEP_CB] type={type(step).__name__}\n")
            sys.stderr.flush()
            if isinstance(step, AgentAction):
                thought = (step.thought or '').strip()
                if thought:
                    # Show the agent's actual thought process
                    for line in thought.split('\n'):
                        line = line.strip()
                        if line:
                            q_ref.put(f"💭 {line}")
                tool_name = step.tool or ''
                tool_input = step.tool_input or ''
                if tool_name:
                    q_ref.put(f"🔧 Using {tool_name}: {str(tool_input)[:200]}")
                result = str(step.result or '')[:300]
                if result:
                    q_ref.put(f"📋 Result: {result}")
            elif isinstance(step, AgentFinish):
                thought = (step.thought or '').strip()
                if thought:
                    for line in thought.split('\n'):
                        line = line.strip()
                        if line:
                            q_ref.put(f"💭 {line}")
                q_ref.put("✅ Composing final answer...")
            else:
                # Unknown step type — log and show raw
                sys.stderr.write(f"[STEP_CB] unknown step: {repr(step)[:300]}\n")
                sys.stderr.flush()
                q_ref.put(f"💭 {str(step)[:300]}")
        return _step_cb

    def _make_event_handlers(q_ref):
        """Create event bus handlers that put reasoning events into the queue."""

        @crewai_event_bus.on(AgentReasoningStartedEvent)
        def on_reasoning_started(source, event):
            q_ref.put(f"🧠 Planning (attempt {event.attempt})...")

        @crewai_event_bus.on(AgentReasoningCompletedEvent)
        def on_reasoning_completed(source, event):
            status = "✅ Ready" if event.ready else "🔄 Refining"
            q_ref.put(f"🧠 {status}")
            plan = (event.plan or '').strip()
            if plan:
                for line in plan.split('\n'):
                    line = line.strip()
                    if line:
                        q_ref.put(f"📝 {line}")

        @crewai_event_bus.on(AgentReasoningFailedEvent)
        def on_reasoning_failed(source, event):
            q_ref.put(f"⚠️ Reasoning error: {str(event.error)[:200]}")

        # NOTE: ToolUsageStarted/Finished are handled by step_callback
        # (which also includes 💭 thoughts). Only subscribe to errors here.

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_error(source, event):
            q_ref.put(f"⚠️ Tool error ({event.tool_name}): {str(event.error)[:200]}")

        @crewai_event_bus.on(StepObservationCompletedEvent)
        def on_observation(source, event):
            info = (event.key_information_learned or '').strip()
            if info:
                q_ref.put(f"👁️ Observed: {info[:300]}")

        @crewai_event_bus.on(GoalAchievedEarlyEvent)
        def on_goal_early(source, event):
            q_ref.put(f"🎯 Goal achieved early (skipping {event.steps_remaining} steps)")

        return [
            (AgentReasoningStartedEvent, on_reasoning_started),
            (AgentReasoningCompletedEvent, on_reasoning_completed),
            (AgentReasoningFailedEvent, on_reasoning_failed),
            (ToolUsageErrorEvent, on_tool_error),
            (StepObservationCompletedEvent, on_observation),
            (GoalAchievedEarlyEvent, on_goal_early),
        ]

    def _run_crew():
        # Set step_callback for structured reasoning before kickoff
        crew.step_callback = _make_step_callback(q)
        # Register event bus handlers for reasoning/planning events
        handlers = _make_event_handlers(q)
        # Still capture stdout for image paths and postprocessing
        writer = _QueueWriter(q)
        old_stdout, old_stdin = sys.stdout, sys.stdin
        sys.stdout = writer
        sys.stdin = io.StringIO('N\n')
        try:
            return crew.kickoff()
        finally:
            writer.flush()
            sys.stdout = old_stdout
            sys.stdin = old_stdin
            # Unregister event handlers to avoid leaks
            for event_type, handler in handlers:
                crewai_event_bus.off(event_type, handler)
            # Clear memory so it cannot leak into the next request
            research_crew.reset_memory()

    def _sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def event_stream():
        async with semaphore:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, _run_crew)
            all_lines: list[str] = []

            _skip_history = False

            # Verbose stdout noise patterns to suppress from reasoning panel
            _verbose_noise = re.compile(
                r'^(Agent:|Task:|Thought:|Action:|Action Input:|Observation:|'
                r'Entering new CrewAgentExecutor|Finished chain|'
                r'> |I encountered an error|Tool .* accepts these inputs|'
                r'Tool Name:|Tool Arguments:|Tool Description:|'
                r'\[1m|> Entering|> Finished|Moving on then)',
                re.IGNORECASE,
            )

            def _should_show(line: str) -> bool:
                nonlocal _skip_history
                if 'Previous conversation:' in line:
                    _skip_history = True
                    return False
                if _skip_history:
                    if 'New request:' in line:
                        _skip_history = False
                    return False
                # Step callback & event bus lines (emoji-prefixed) always show
                if line and line[0] in '💭🔧📋✅🧠📝⚠️👁️🎯':
                    return True
                # Suppress verbose framework noise
                if _verbose_noise.search(line):
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

            response_text = _postprocess(
                research_crew.postprocess(result), '\n'.join(all_lines)
            )
            usage = _extract_usage(result)
            incomplete = _is_incomplete(response_text, '\n'.join(all_lines))

            yield _sse("done", {
                "response": response_text,
                "reasoning": all_lines,
                "token_usage": usage,
                "incomplete": incomplete,
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- /chat/continue  LLM-direct continuation (no CrewAI) ---

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
    research_crew = request.app.state.research_crew
    crew = research_crew.build_crew(topic)
    semaphore = request.app.state.crew_semaphore

    q: queue.Queue = queue.Queue()

    def _run():
        writer = _QueueWriter(q)
        old_stdout, old_stdin = sys.stdout, sys.stdin
        sys.stdout = writer
        sys.stdin = io.StringIO('N\n')
        try:
            return crew.kickoff()
        finally:
            writer.flush()
            sys.stdout = old_stdout
            sys.stdin = old_stdin
            research_crew.reset_memory()

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
        "response": research_crew.postprocess(result),
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
