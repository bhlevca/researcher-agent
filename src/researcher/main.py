import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from researcher.crew import ResearchCrew

load_dotenv()
app = FastAPI()

# Initialize crew once at startup, not per request
_crew_instance = ResearchCrew().crew()

# --- Chat UI ---

STATIC_DIR = Path(__file__).parent / "static"

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/info")
async def info():
    return {"model": os.getenv("MODEL", "ollama/qwen3.5:9b")}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        inputs = {'topic': req.message}
        result = _crew_instance.kickoff(inputs=inputs)
        return {"response": str(result)}
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