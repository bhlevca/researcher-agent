import sys
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from researcher.crew import ResearchCrew

load_dotenv()
app = FastAPI()

# Initialize crew once at startup, not per request
_crew_instance = ResearchCrew().crew()

@app.get("/ask")
async def run(q: str):
    try:
        inputs = {'topic': q}
        result = _crew_instance.kickoff(inputs=inputs)
        return {"status": "success", "response": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # 120 second timeout for your 5070 Ti to process the search
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)