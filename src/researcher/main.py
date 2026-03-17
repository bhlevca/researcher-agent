import sys
import os
import io
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from researcher.crew import ResearchCrew

load_dotenv()
app = FastAPI()

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Mutable state: current crew and model
_current_model = os.getenv("MODEL", "ollama/qwen3.5:9b")
_crew_instance = ResearchCrew(model=_current_model).crew()

# --- Chat UI ---

STATIC_DIR = Path(__file__).parent / "static"

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

class ChatRequest(BaseModel):
    message: str

class ModelRequest(BaseModel):
    model: str

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
    try:
        inputs = {'topic': req.message}

        # Capture CrewAI's verbose stdout (reasoning steps)
        capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = capture
        try:
            result = _crew_instance.kickoff(inputs=inputs)
        finally:
            sys.stdout = old_stdout

        verbose_log = _clean_log(capture.getvalue())

        # Build reasoning steps from captured output
        steps = [line for line in verbose_log.splitlines() if line.strip()]

        # Token usage if available
        usage = {}
        if hasattr(result, 'token_usage') and result.token_usage:
            tu = result.token_usage
            usage = {
                "total_tokens": getattr(tu, 'total_tokens', 0),
                "prompt_tokens": getattr(tu, 'prompt_tokens', 0),
                "completion_tokens": getattr(tu, 'completion_tokens', 0),
            }

        return {
            "response": str(result),
            "reasoning": steps,
            "token_usage": usage,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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