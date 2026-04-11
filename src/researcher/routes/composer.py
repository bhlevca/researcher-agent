"""Music Composer API routes.

Manages composer sessions, chat-based composition, score generation,
harmonization, analysis, and composition saving/exporting.
"""

import sys
import io
import re
import json
import queue
import asyncio
import logging

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse, Response

from researcher.auth import get_current_user
from researcher.config import _QueueWriter, _SESSION_ID_RE

from researcher.composer.models import (
    ComposerSessionCreate,
    ComposerSessionUpdate,
    ComposeChatRequest,
    ComposeScoreRequest,
    ComposeHarmonizeRequest,
    ComposeAnalyzeRequest,
)
from researcher.composer.storage import (
    create_composer_session,
    list_composer_sessions,
    get_composer_session,
    update_composer_session_messages,
    delete_composer_session,
    save_composition,
    list_compositions,
    get_composition,
    delete_composition,
)

logger = logging.getLogger(__name__)


def _validate_sid(session_id: str):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")


def _build_composer_context(messages: list, max_messages: int = 10) -> str:
    if not messages:
        return "No previous conversation."
    recent = messages[-max_messages:]
    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        text = msg.get("content", "")[:500]
        lines.append(f"{role.capitalize()}: {text}")
    return "\n".join(lines)


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _run_composer_crew(crew, q: queue.Queue):
    """Run a composer crew in a thread, capturing stdout."""
    from researcher.routes.chat import _make_step_callback, _make_event_handlers
    from crewai.events.event_bus import crewai_event_bus

    crew.step_callback = _make_step_callback(q)
    handlers = _make_event_handlers(q)
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
        for event_type, handler in handlers:
            crewai_event_bus.off(event_type, handler)


async def _stream_composer_crew(crew, request: Request):
    """Generic SSE streamer for any composer crew run."""
    semaphore = request.app.state.crew_semaphore
    q: queue.Queue = queue.Queue()

    _verbose_noise = re.compile(
        r"^(Agent:|Task:|Thought:|Action:|Action Input:|Observation:|"
        r"Entering new CrewAgentExecutor|Finished chain|"
        r"> |I encountered an error|Tool .* accepts these inputs|"
        r"Tool Name:|Tool Arguments:|Tool Description:|"
        r"\[1m|> Entering|> Finished|Moving on then)",
        re.IGNORECASE,
    )

    def _should_show(line: str) -> bool:
        if line and line[0] in "💭🔧📋✅🧠📝⚠️👁️🎯🎵🎶":
            return True
        if _verbose_noise.search(line):
            return False
        return True

    async def event_stream():
        cancel = request.app.state.cancel_event
        cancel.clear()
        async with semaphore:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, lambda: _run_composer_crew(crew, q))
            all_lines: list[str] = []

            while not future.done():
                if cancel.is_set():
                    yield _sse("cancelled", "Request cancelled by user")
                    return
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

            response_text = result.raw if hasattr(result, "raw") else str(result)
            response_text = re.sub(r"<think>[\s\S]*?</think>\s*", "", response_text).strip()

            # Detect if response contains a score (JSON or XML)
            has_score = False
            if '<score-partwise' in response_text or '```xml' in response_text:
                has_score = True
            elif '```json' in response_text and '"parts"' in response_text:
                has_score = True
            elif '"parts"' in response_text and '"measures"' in response_text:
                has_score = True

            yield _sse(
                "done",
                {
                    "response": response_text,
                    "reasoning": all_lines,
                    "has_score": has_score,
                },
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def register_composer_routes(app):
    """Attach all /composer/* endpoints to the FastAPI app."""

    # ------------------------------------------------------------------
    # Composer Sessions CRUD
    # ------------------------------------------------------------------

    @app.get("/composer/sessions")
    async def composer_list_sessions(request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        sessions = await list_composer_sessions(db, user["id"])
        return {"sessions": sessions}

    @app.post("/composer/sessions")
    async def composer_create_session(req: ComposerSessionCreate, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        session = await create_composer_session(
            db, user["id"], req.name, req.genre,
            req.key_signature, req.time_signature, req.tempo,
        )
        return session

    @app.get("/composer/sessions/{session_id}")
    async def composer_get_session(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        session = await get_composer_session(db, session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Composer session not found")
        return session

    @app.put("/composer/sessions/{session_id}")
    async def composer_update_session(
        session_id: str, req: ComposerSessionUpdate, request: Request
    ):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        session = await get_composer_session(db, session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Composer session not found")
        await update_composer_session_messages(db, session_id, req.messages)
        return {"id": session_id, "name": req.name}

    @app.delete("/composer/sessions/{session_id}")
    async def composer_delete_session(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        deleted = await delete_composer_session(db, session_id, user["id"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Composer session not found")
        return {"deleted": session_id}

    # ------------------------------------------------------------------
    # Chat / Compose (SSE streaming)
    # ------------------------------------------------------------------

    @app.post("/composer/chat")
    async def composer_chat(req: ComposeChatRequest, request: Request):
        """Conversational composition — SSE streaming."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db

        session = await get_composer_session(db, req.session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Composer session not found")

        composer_crew = request.app.state.composer_crew
        context = _build_composer_context(session["messages"])

        crew = composer_crew.build_chat_crew(
            message=req.message,
            context=context,
            genre=session["genre"],
            key_signature=session["key_signature"],
            time_signature=session["time_signature"],
            tempo=session["tempo"],
        )

        return await _stream_composer_crew(crew, request)

    # ------------------------------------------------------------------
    # Score Generation (SSE streaming)
    # ------------------------------------------------------------------

    @app.post("/composer/score")
    async def composer_generate_score(req: ComposeScoreRequest, request: Request):
        """Generate a full score — SSE streaming."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db

        session = await get_composer_session(db, req.session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Composer session not found")

        composer_crew = request.app.state.composer_crew

        crew = composer_crew.build_score_crew(
            description=req.description,
            instruments=req.instruments,
            measures=req.measures,
            style=req.style,
            genre=session["genre"],
            key_signature=session["key_signature"],
            time_signature=session["time_signature"],
            tempo=session["tempo"],
        )

        return await _stream_composer_crew(crew, request)

    # ------------------------------------------------------------------
    # Harmonize (SSE streaming)
    # ------------------------------------------------------------------

    @app.post("/composer/harmonize")
    async def composer_harmonize(req: ComposeHarmonizeRequest, request: Request):
        """Harmonize a melody — SSE streaming."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db

        session = await get_composer_session(db, req.session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Composer session not found")

        composer_crew = request.app.state.composer_crew

        crew = composer_crew.build_harmonize_crew(
            melody=req.melody,
            style=req.style,
            genre=session["genre"],
            key_signature=session["key_signature"],
            time_signature=session["time_signature"],
            tempo=session["tempo"],
        )

        return await _stream_composer_crew(crew, request)

    # ------------------------------------------------------------------
    # Analyze (SSE streaming)
    # ------------------------------------------------------------------

    @app.post("/composer/analyze")
    async def composer_analyze(req: ComposeAnalyzeRequest, request: Request):
        """Analyze musical content — SSE streaming."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db

        session = await get_composer_session(db, req.session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Composer session not found")

        composer_crew = request.app.state.composer_crew

        crew = composer_crew.build_analyze_crew(
            content=req.content,
            genre=session["genre"],
            key_signature=session["key_signature"],
            time_signature=session["time_signature"],
            tempo=session["tempo"],
        )

        return await _stream_composer_crew(crew, request)

    # ------------------------------------------------------------------
    # Compositions CRUD
    # ------------------------------------------------------------------

    @app.post("/composer/convert-musicxml")
    async def composer_convert_musicxml(request: Request):
        """Convert a JSON score response to MusicXML on the fly (for download)."""
        user = await get_current_user(request)
        body = await request.json()

        session_id = body.get("session_id", "")
        response_text = body.get("response", "")
        if not response_text:
            raise HTTPException(status_code=400, detail="No response text")

        # Try to get session params (key/time/tempo) if session provided
        key_sig, time_sig, tempo = "C major", "4/4", 120
        if session_id and _SESSION_ID_RE.match(session_id):
            db = request.app.state.db
            session = await get_composer_session(db, session_id, user["id"])
            if session:
                key_sig = session.get("key_signature", key_sig)
                time_sig = session.get("time_signature", time_sig)
                tempo = session.get("tempo", tempo)

        composer_crew = request.app.state.composer_crew
        musicxml = composer_crew.extract_musicxml(
            response_text, key=key_sig, time_signature=time_sig, tempo=tempo,
        )
        if not musicxml:
            raise HTTPException(status_code=422, detail="Could not extract score")

        return Response(
            content=musicxml,
            media_type="application/vnd.recordare.musicxml+xml",
            headers={
                "Content-Disposition": 'attachment; filename="composition.musicxml"'
            },
        )

    @app.post("/composer/compositions/save")
    async def composer_save_composition(request: Request):
        """Save a generated composition (MusicXML) to the database."""
        user = await get_current_user(request)
        db = request.app.state.db
        body = await request.json()

        session_id = body.get("session_id", "")
        _validate_sid(session_id)
        session = await get_composer_session(db, session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Composer session not found")

        composer_crew = request.app.state.composer_crew
        response_text = body.get("response", "")
        musicxml = composer_crew.extract_musicxml(
            response_text,
            key=session["key_signature"],
            time_signature=session["time_signature"],
            tempo=session["tempo"],
        )

        comp = await save_composition(
            db,
            user_id=user["id"],
            session_id=session_id,
            title=body.get("title", "Untitled"),
            description=body.get("description", ""),
            instruments=body.get("instruments", []),
            genre=session["genre"],
            key_signature=session["key_signature"],
            time_signature=session["time_signature"],
            tempo=session["tempo"],
            musicxml=musicxml or "",
        )
        comp["has_musicxml"] = bool(musicxml)
        return comp

    @app.get("/composer/sessions/{session_id}/compositions")
    async def composer_list_compositions(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        comps = await list_compositions(db, user["id"], session_id)
        return {"compositions": comps}

    @app.get("/composer/compositions/{composition_id}")
    async def composer_get_composition(composition_id: str, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        comp = await get_composition(db, composition_id, user["id"])
        if not comp:
            raise HTTPException(status_code=404, detail="Composition not found")
        return comp

    @app.get("/composer/compositions/{composition_id}/musicxml")
    async def composer_download_musicxml(composition_id: str, request: Request):
        """Download the MusicXML file for a composition."""
        user = await get_current_user(request)
        db = request.app.state.db
        comp = await get_composition(db, composition_id, user["id"])
        if not comp:
            raise HTTPException(status_code=404, detail="Composition not found")
        if not comp.get("musicxml"):
            raise HTTPException(status_code=404, detail="No MusicXML data")
        filename = re.sub(r'[^\w\s-]', '', comp["title"]).strip().replace(' ', '_')
        return Response(
            content=comp["musicxml"],
            media_type="application/vnd.recordare.musicxml+xml",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}.musicxml"'
            },
        )

    @app.delete("/composer/compositions/{composition_id}")
    async def composer_delete_composition(composition_id: str, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        deleted = await delete_composition(db, composition_id, user["id"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Composition not found")
        return {"deleted": composition_id}
