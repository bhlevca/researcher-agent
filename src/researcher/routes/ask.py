"""Programmatic /ask API endpoint."""

import sys
import io
import queue
import asyncio
import logging
import json

from fastapi import HTTPException, Request

from researcher.config import AskRequest, _QueueWriter, _maybe_unload_ollama
from researcher.postprocess import _extract_usage
from researcher.auth import get_current_user
from researcher.ingestion import get_file_context
from researcher.image import load_image_params_from_json
from researcher.crew import ResearchCrew

logger = logging.getLogger(__name__)


def register_ask_routes(app):
    """Attach /ask endpoint to the FastAPI app."""

    @app.post("/ask")
    async def ask(req: AskRequest, request: Request):
        """Programmatic API endpoint. POST JSON with {topic: "..."}. Returns structured response."""
        user = await get_current_user(request)

        # Ensure runtime model matches the user's saved model.
        db = request.app.state.db
        cursor_model = await db.execute(
            "SELECT model, llm_params FROM users WHERE id = ?", (user["id"],)
        )
        row_model = await cursor_model.fetchone()
        saved_model = row_model[0] if row_model and row_model[0] else None
        effective_model = saved_model or request.app.state.current_model
        if effective_model != request.app.state.current_model:
            new_crew = ResearchCrew(model=effective_model)
            if row_model and row_model[1]:
                try:
                    new_crew.update_llm_params(json.loads(row_model[1]))
                except (json.JSONDecodeError, TypeError):
                    pass
            request.app.state.research_crew = new_crew
            request.app.state.current_model = effective_model

        file_context = ""
        if req.file_ids:
            db = request.app.state.db
            file_context = await get_file_context(db, user["id"], req.file_ids)

        topic = file_context + req.topic
        research_crew = request.app.state.research_crew

        # Ensure image params from DB are applied before generation
        cursor = await db.execute(
            "SELECT image_params FROM users WHERE id = ?", (user["id"],)
        )
        row = await cursor.fetchone()
        load_image_params_from_json(row[0] if row and row[0] else None)

        crew = research_crew.build_crew(topic)
        semaphore = request.app.state.crew_semaphore

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
            "model": request.app.state.current_model,
        }
