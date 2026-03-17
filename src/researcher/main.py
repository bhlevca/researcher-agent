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
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from researcher.crew import (
    ResearchCrew,
    generate_image as _generate_image_tool,
    generate_ai_image as _generate_ai_image_tool,
    preload_sd,
)

load_dotenv()
app = FastAPI()

# Pre-load Stable Diffusion in background so first image request is fast
preload_sd()

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Mutable state: current crew and model
_current_model = os.getenv("MODEL", "ollama/qwen3.5:9b")
_crew_instance = ResearchCrew(model=_current_model).crew()

# --- Chat UI ---

STATIC_DIR = Path(__file__).parent / "static"
SESSIONS_DIR = Path(__file__).parent / "data" / "sessions"
GENERATED_DIR = STATIC_DIR / "generated"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Strip ANSI escape codes and box-drawing decorations from captured output
_ansi_re = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')
_box_re = re.compile(r'[╭╮╰╯│─┌┐└┘├┤┬┴┼]+')

def _clean_log(text: str) -> str:
    text = _ansi_re.sub('', text)
    text = _box_re.sub('', text)
    # Collapse runs of whitespace on each line and drop empty lines
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return '\n'.join(lines)

def _clean_line(text: str) -> str:
    """Clean a single line of stdout output."""
    text = _ansi_re.sub('', text)
    text = _box_re.sub('', text)
    return text.strip()

class _QueueWriter:
    """Captures writes to stdout and puts cleaned lines into a queue."""
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

class ChatRequest(BaseModel):
    message: str
    history: list = []

class ModelRequest(BaseModel):
    model: str

class SessionSaveRequest(BaseModel):
    name: str
    messages: list

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/info")
async def info():
    return {"model": _current_model}

@app.get("/models")
async def list_models():
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
        return {"models": models, "current": _current_model}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {e}")

@app.post("/model")
async def switch_model(req: ModelRequest):
    """Switch the active LLM model."""
    global _current_model, _crew_instance
    new_model = req.model if req.model.startswith("ollama/") else f"ollama/{req.model}"
    try:
        _crew_instance = ResearchCrew(model=new_model).crew()
        _current_model = new_model
        return {"model": _current_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {e}")

@app.post("/chat")
async def chat(req: ChatRequest):
    # Build context from conversation history so follow-up requests work
    context_lines = []
    for msg in req.history[-6:]:   # last 3 exchanges max
        role = msg.get('role', 'user')
        text = msg.get('text', '')
        if role == 'user':
            context_lines.append(f"User: {text}")
        elif role == 'assistant':
            short = text[:300] + ('...' if len(text) > 300 else '')
            context_lines.append(f"Assistant: {short}")

    if context_lines:
        topic = ("Previous conversation:\n"
                 + "\n".join(context_lines)
                 + "\n\nNew request: " + req.message)
    else:
        topic = req.message

    inputs = {'topic': topic}

    q: queue.Queue = queue.Queue()

    def _run_crew():
        writer = _QueueWriter(q)
        old_stdout = sys.stdout
        old_stdin = sys.stdin
        sys.stdout = writer
        sys.stdin = io.StringIO('N\n')
        try:
            result = _crew_instance.kickoff(inputs=inputs)
        finally:
            writer.flush()
            sys.stdout = old_stdout
            sys.stdin = old_stdin
        return result

    def _sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def event_stream():
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, _run_crew)
        all_lines: list[str] = []

        # Stream reasoning lines while crew is running
        while not future.done():
            await asyncio.sleep(0.15)
            while not q.empty():
                try:
                    line = q.get_nowait()
                    all_lines.append(line)
                    yield _sse("reasoning", line)
                except queue.Empty:
                    break

        # Drain remaining lines
        while not q.empty():
            try:
                line = q.get_nowait()
                all_lines.append(line)
                yield _sse("reasoning", line)
            except queue.Empty:
                break

        # Get result (or error)
        try:
            result = future.result()
        except Exception as e:
            yield _sse("error", str(e))
            return

        verbose_log = '\n'.join(all_lines)
        response_text = str(result)

        # --- Post-processing (image extraction, validation, cleanup) ---
        _img_md_re = re.compile(r'!\[[^\]]*\]\(/static/generated/[a-f0-9]+\.png\)')
        if '/static/generated/' not in response_text:
            images_in_log = _img_md_re.findall(verbose_log)
            if images_in_log:
                response_text += '\n\n' + '\n'.join(images_in_log)

        def _validate_img(match):
            rel = match.group(1).replace('/static/', '', 1)
            path = STATIC_DIR / rel
            return match.group(0) if path.exists() else ''
        response_text = re.sub(
            r'!\[[^\]]*\]\((/static/generated/[^)]+)\)',
            _validate_img, response_text
        )

        # Orphan rescue (run tool server-side if LLM described action in text)
        _orphan_re = re.compile(
            r'Action:\s*(generate_?(?:ai_?)?image)\s*\n\s*Action\s*Input:\s*(\{.*\}|.+)',
            re.DOTALL | re.IGNORECASE
        )
        orphan = _orphan_re.search(response_text)
        if orphan and '/static/generated/' not in response_text:
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
                    img_result = _generate_ai_image_tool.run(prompt)
                else:
                    parsed = json.loads(raw_input)
                    if 'instructions' in parsed:
                        iv = parsed['instructions']
                        if isinstance(iv, dict):
                            iv = json.dumps(iv)
                        img_result = _generate_image_tool.run(iv)
                    else:
                        img_result = _generate_image_tool.run(raw_input)
                response_text = img_result
            except Exception:
                pass

        response_text = re.sub(r'<result>\s*</result>', '', response_text)
        response_text = re.sub(
            r'Thought:.*?Action:.*?Action Input:.*?$',
            '', response_text, flags=re.DOTALL
        )
        response_text = response_text.strip()

        if not response_text:
            images_in_log = _img_md_re.findall(verbose_log)
            valid = [img for img in images_in_log
                     if (STATIC_DIR / img.split('(')[1].rstrip(')').replace('/static/', '', 1)).exists()]
            if valid:
                response_text = valid[-1]
            else:
                response_text = 'The agent could not complete the request. Please try rephrasing.'

        usage = {}
        if hasattr(result, 'token_usage') and result.token_usage:
            tu = result.token_usage
            usage = {
                "total_tokens": getattr(tu, 'total_tokens', 0),
                "prompt_tokens": getattr(tu, 'prompt_tokens', 0),
                "completion_tokens": getattr(tu, 'completion_tokens', 0),
            }

        yield _sse("done", {
            "response": response_text,
            "reasoning": all_lines,
            "token_usage": usage,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# --- Session management ---

_SESSION_ID_RE = re.compile(r'^[a-f0-9]{8}$')

def _session_path(session_id: str) -> Path:
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
    return SESSIONS_DIR / f"{session_id}.json"

@app.get("/sessions")
async def list_sessions():
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append({
                "id": data["id"],
                "name": data["name"],
                "created_at": data["created_at"],
                "updated_at": data.get("updated_at", data["created_at"]),
                "message_count": len(data.get("messages", [])),
                "model": data.get("model", ""),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return {"sessions": sessions}

@app.post("/sessions")
async def save_session(req: SessionSaveRequest):
    sid = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    session = {
        "id": sid,
        "name": req.name,
        "created_at": now,
        "updated_at": now,
        "model": _current_model,
        "messages": req.messages,
    }
    (SESSIONS_DIR / f"{sid}.json").write_text(json.dumps(session, indent=2))
    return {"id": sid, "name": req.name}

@app.get("/sessions/{session_id}")
async def load_session(session_id: str):
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return json.loads(path.read_text())

@app.put("/sessions/{session_id}")
async def update_session(session_id: str, req: SessionSaveRequest):
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    data = json.loads(path.read_text())
    data["name"] = req.name
    data["messages"] = req.messages
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["model"] = _current_model
    path.write_text(json.dumps(data, indent=2))
    return {"id": session_id, "name": req.name}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    path.unlink()
    return {"deleted": session_id}

# --- Legacy GET endpoint ---

@app.get("/ask")
async def run(q: str):
    try:
        inputs = {'topic': q}
        result = _crew_instance.kickoff(inputs=inputs)
        return {"status": "success", "response": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    # 120 second timeout for your 5070 Ti to process the search
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)