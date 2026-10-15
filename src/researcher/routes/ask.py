"""Programmatic /ask API endpoint."""

import sys
import io
import queue
import asyncio
import logging
import json

from fastapi import HTTPException, Request

from researcher.config import AskRequest, _QueueWriter, _maybe_unload_ollama
from researcher.gpu_broker import GPU, Need
from researcher.user_crews import get_crew
from researcher.postprocess import _extract_usage
from researcher.auth import get_current_user
from researcher.ingestion import get_file_context
from researcher.image import load_image_params_from_json

logger = logging.getLogger(__name__)


def register_ask_routes(app):
    """Attach /ask endpoint to the FastAPI app."""

    @app.post("/ask")
    async def ask(req: AskRequest, request: Request):
        """Programmatic API endpoint. POST JSON with {topic: "..."}. Returns structured response."""
        user = await get_current_user(request)
        uid = user["id"]

        # The model this person chose, which is held in the database rather
        # than in app.state.current_model — that global is reset from .env on
        # every restart and is shared between everyone.
        db = request.app.state.db
        research_crew = await get_crew(request, uid, "research")

        if uid not in request.app.state.crew_semaphores:
            request.app.state.crew_semaphores[uid] = asyncio.Semaphore(1)

        file_context = ""
        if req.file_ids:
            file_context = await get_file_context(db, uid, req.file_ids)

        topic = file_context + req.topic

        # Ensure image params from DB are applied before generation
        cursor = await db.execute(
            "SELECT image_params FROM users WHERE id = ?", (uid,)
        )
        row = await cursor.fetchone()
        import sys as _sys, json as _json
        _raw = row[0] if row and row[0] else None
        _parsed = _json.loads(_raw) if _raw else {}
        _sys.stderr.write(f"[ASKDBG] image_params from DB: realesrgan_enabled={_parsed.get('realesrgan_enabled', 'MISSING')}\n")
        load_image_params_from_json(_raw)

        crew = research_crew.build_crew(topic, request_text=req.topic)
        semaphore = request.app.state.crew_semaphores[uid]

        q: queue.Queue = queue.Queue()

        def _run():
            writer = _QueueWriter(q)
            old_stdout, old_stdin = sys.stdout, sys.stdin
            sys.stdout = writer
            sys.stdin = io.StringIO("N\n")
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
                _need = Need("ollama",
                             research_crew.model_name.replace("ollama/", ""))
                async with GPU.exclusive(_need, "ask", user=uid):
                    result = await loop.run_in_executor(None, _run)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Agent error: {e}")

        reasoning = []
        while not q.empty():
            try:
                reasoning.append(q.get_nowait())
            except queue.Empty:
                break

        _maybe_unload_ollama()

        return {
            "status": "success",
            "response": research_crew.postprocess(result),
            "reasoning": reasoning,
            "token_usage": _extract_usage(result),
            "model": research_crew.model_name,
        }
