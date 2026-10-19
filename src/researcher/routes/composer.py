"""Music Composer API routes.

Manages composer sessions, chat-based composition, score generation,
harmonization, analysis, and composition saving/exporting.
"""

import sys
import io
import re
import json
import pathlib
import queue
import asyncio
import threading
import logging
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse, Response

from researcher.auth import get_current_user
from researcher.config import _QueueWriter, _SESSION_ID_RE
from researcher import song
from researcher.composer.crew import ComposerCrew
from researcher.downloads import content_disposition
from researcher.gpu_broker import GPU, Need, queue_message
from researcher.user_crews import get_crew, resolve_model

from researcher.composer.models import (
    ComposeSongRequest,
    CompositionRename,
    SongSaveRequest,
    ComposerSessionCreate,
    ComposerSessionUpdate,
    ComposeChatRequest,
    ComposeScoreRequest,
    ComposeHarmonizeRequest,
    ComposeAnalyzeRequest,
)
from researcher.composer.storage import (
    saved_audio_paths,
    find_composition_by_audio,
    rename_composition,
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




async def _get_composer_crew(request: Request, uid: str) -> ComposerCrew:
    """The per-user ComposerCrew, built with the model that person chose."""
    return await get_crew(request, uid, "composer")


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


async def _stream_composer_crew(crew, request: Request, uid: str, timeout_sec: int | None = None):
    """Generic SSE streamer for any composer crew run."""
    # Lazy-init per-user semaphore and cancel event
    if uid not in request.app.state.crew_semaphores:
        request.app.state.crew_semaphores[uid] = asyncio.Semaphore(1)
    if uid not in request.app.state.cancel_events:
        request.app.state.cancel_events[uid] = threading.Event()

    semaphore = request.app.state.crew_semaphores[uid]
    if timeout_sec is None:
        timeout_sec = request.app.state.crew_timeout
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
        cancel = request.app.state.cancel_events[uid]
        cancel.clear()
        async with semaphore:
            # One card, one queue: the composer takes its turn beside the
            # researcher and the tutor rather than loading a second model
            # alongside them.
            _chosen, _ = await resolve_model(request, uid)
            _need = Need("ollama", str(_chosen).replace("ollama/", ""))
            async with GPU.reserve(_need, "composer", user=uid) as _res:
                async for _notice in _res.wait():
                    yield _sse("queue", queue_message(
                        _notice["position"], _notice["estimate"],
                        _notice["ticket"]))
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, lambda: _run_composer_crew(crew, q))
                all_lines: list[str] = []
                elapsed = 0.0

                early_exit: str | None = None  # "cancelled" | "timeout"

                while not future.done():
                    if cancel.is_set() and early_exit is None:
                        early_exit = "cancelled"
                        yield _sse("cancelled", "Request cancelled by user")
                    if elapsed >= timeout_sec and early_exit is None:
                        early_exit = "timeout"
                        yield _sse("error", f"Request timed out after {timeout_sec}s")
                    await asyncio.sleep(0.15)
                    elapsed += 0.15
                    while not q.empty():
                        try:
                            line = q.get_nowait()
                            all_lines.append(line)
                            if early_exit is None and _should_show(line):
                                yield _sse("reasoning", line)
                        except queue.Empty:
                            break

                if early_exit is not None:
                    # Drain queue silently; semaphore releases after thread finished
                    while not q.empty():
                        q.get_nowait()
                    return

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

        uid = user["id"]
        composer_crew = await _get_composer_crew(request, uid)
        context = _build_composer_context(session["messages"])

        crew = composer_crew.build_chat_crew(
            message=req.message,
            context=context,
            genre=session["genre"],
            key_signature=session["key_signature"],
            time_signature=session["time_signature"],
            tempo=session["tempo"],
        )

        return await _stream_composer_crew(crew, request, uid, timeout_sec=composer_crew.get_llm_params().get("crew_timeout_sec"))

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

        uid = user["id"]
        composer_crew = await _get_composer_crew(request, uid)

        # ── Specialist path ──────────────────────────────────────────────
        # A model pre-trained on ABC writes idiomatic music; a general chat
        # model writes exercises. When one is configured, call it directly:
        # it has no tools and needs no planning loop, so routing it through
        # CrewAI would only add minutes and a format it does not speak.
        from researcher.composer import specialist

        if specialist.COMPOSER_MODEL:
            from researcher.composer.knowledge.loader import get_genre_block

            style_guide = get_genre_block(req.style or session["genre"]) or ""

            def _sse(event, data):
                return f"event: {event}\ndata: {json.dumps(data)}\n\n"

            async def specialist_stream():
                yield _sse("reasoning",
                           f"🎼 Composing with {specialist.COMPOSER_MODEL}")

                # Long form is assembled, not requested in one go: a single
                # model call holds together for 8-16 bars and then repeats.
                if req.minutes and req.minutes >= 0.5:
                    yield _sse("reasoning",
                               f"🏗️ Assembling ~{req.minutes:g} min from recurring "
                               f"sections with extended harmony")
                    loop = asyncio.get_running_loop()
                    # The specialist runs its own crew, so it queues too.
                    _chosen, _ = await resolve_model(request, uid)
                    async with GPU.exclusive(
                            Need("ollama", str(_chosen).replace("ollama/", "")),
                            "composer", user=uid):
                        score, info = await loop.run_in_executor(
                            None,
                            lambda: specialist.compose_long(
                                description=req.description,
                                genre=session["genre"],
                                key_signature=session["key_signature"],
                                time_signature=session["time_signature"],
                                tempo=session["tempo"],
                                minutes=req.minutes,
                                style_guide=style_guide,
                                instruments=req.instruments,
                            ),
                        )
                    if score is None:
                        yield _sse("error", "Could not assemble the piece.")
                        return
                    seeded = ("seeded by the specialist" if info.get("used_specialist")
                              else "using a default motif")
                    yield _sse("reasoning",
                               f"📐 {info['bars']} bars, {info['seconds']/60:.1f} min, "
                               f"{info['sections']} sections, {seeded}")
                    import tempfile as _tf
                    with _tf.NamedTemporaryFile(suffix=".musicxml", delete=False) as t:
                        xml_path = t.name
                    score.write("musicxml", fp=xml_path)
                    xml = pathlib.Path(xml_path).read_text()
                    pathlib.Path(xml_path).unlink(missing_ok=True)
                    yield _sse("done", {
                        "response": (
                            f"Composed a {info['seconds']/60:.1f}-minute piece: "
                            f"{info['bars']} bars in {info['sections']} sections.\n\n"
                            f"```musicxml\n{xml}\n```"
                        ),
                        "reasoning": [f"Assembled by realize.py, {seeded}"],
                        "token_usage": {}, "incomplete": False,
                    })
                    return
                loop = asyncio.get_running_loop()
                # The specialist runs its own crew, so it queues too.
                _chosen, _ = await resolve_model(request, uid)
                async with GPU.exclusive(
                        Need("ollama", str(_chosen).replace("ollama/", "")),
                        "composer", user=uid):
                    abc = await loop.run_in_executor(
                        None,
                        lambda: specialist.compose(
                            description=req.description,
                            genre=session["genre"],
                            key_signature=session["key_signature"],
                            time_signature=session["time_signature"],
                            tempo=session["tempo"],
                            bars=req.measures,
                            style_guide=style_guide,
                        ),
                    )
                if not abc:
                    yield _sse("error", "The composer model returned nothing.")
                    return
                abc, repairs = specialist.sanitize_abc(abc)
                if repairs:
                    yield _sse("reasoning", "🧹 Fixed notation: " + ", ".join(repairs))
                report = composer_crew.validate_abc(abc, session["time_signature"])
                yield _sse("reasoning",
                           f"📐 {report['measures']} bars, "
                           f"{report['stepwise_pct']:.0f}% stepwise, "
                           f"longest run {report['longest_run']}")
                if not report["ok"] and report.get("short_bars"):
                    fixed, bars = composer_crew.repair_bar_lengths(
                        abc, session["time_signature"])
                    if fixed:
                        yield _sse("reasoning",
                                   f"🔧 Re-barred {report['short_bars']} short bar(s)")
                        abc = fixed

                # The specialist supplies idiomatic material; this enforces what
                # it reliably gets wrong — scale runs, and a bare single line.
                try:
                    from researcher.composer import realize

                    polished, prep = realize.polish(
                        abc, genre=session["genre"],
                        key_name=session["key_signature"],
                        tempo=session["tempo"])
                    if polished is not None:
                        notes = []
                        if prep["scale_runs_fixed"]:
                            notes.append(f"broke {prep['scale_runs_fixed']} scale run(s)")
                        if prep["bass_added"]:
                            notes.append("added a bass line from the chord symbols")
                        if notes:
                            yield _sse("reasoning", "🎹 Polished: " + ", ".join(notes))
                except Exception as e:
                    logger.warning("[Composer] polish skipped: %s", e)
                yield _sse("done", {
                    "response": f"```abc\n{abc}\n```",
                    "reasoning": [f"Composed by {specialist.COMPOSER_MODEL}"],
                    "token_usage": {},
                    "incomplete": False,
                })

            return StreamingResponse(specialist_stream(),
                                     media_type="text/event-stream")

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

        return await _stream_composer_crew(crew, request, uid, timeout_sec=composer_crew.get_llm_params().get("crew_timeout_sec"))

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

        uid = user["id"]
        composer_crew = await _get_composer_crew(request, uid)

        crew = composer_crew.build_harmonize_crew(
            melody=req.melody,
            style=req.style,
            genre=session["genre"],
            key_signature=session["key_signature"],
            time_signature=session["time_signature"],
            tempo=session["tempo"],
        )

        return await _stream_composer_crew(crew, request, uid, timeout_sec=composer_crew.get_llm_params().get("crew_timeout_sec"))

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

        uid = user["id"]
        composer_crew = await _get_composer_crew(request, uid)

        crew = composer_crew.build_analyze_crew(
            content=req.content,
            genre=session["genre"],
            key_signature=session["key_signature"],
            time_signature=session["time_signature"],
            tempo=session["tempo"],
        )

        return await _stream_composer_crew(crew, request, uid, timeout_sec=composer_crew.get_llm_params().get("crew_timeout_sec"))

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

        composer_crew = await _get_composer_crew(request, user["id"])
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

        composer_crew = await _get_composer_crew(request, user["id"])
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
        for c in comps:
            stored = c.get("audio_path") or ""
            c["audio_url"] = song.url_for(stored) if "/" in stored else ""
            c["has_musicxml"] = bool(c.get("has_musicxml"))
        return {"compositions": comps}

    @app.get("/composer/compositions/{composition_id}")
    async def composer_get_composition(composition_id: str, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        comp = await get_composition(db, composition_id, user["id"])
        if not comp:
            raise HTTPException(status_code=404, detail="Composition not found")
        stored = comp.get("audio_path") or ""
        comp["audio_url"] = song.url_for(stored) if "/" in stored else ""
        return comp

    @app.post("/composer/convert-midi")
    async def composer_convert_midi(request: Request):
        """Render a response's score to MIDI so it can be played in the browser."""
        user = await get_current_user(request)
        body = await request.json()
        response_text = body.get("response", "")
        if not response_text:
            raise HTTPException(status_code=400, detail="No response text")

        key_sig, time_sig, tempo = "C major", "4/4", 120
        session_id = body.get("session_id", "")
        if session_id:
            session = await get_composer_session(request.app.state.db, session_id, user["id"])
            if session:
                key_sig = session["key_signature"]
                time_sig = session["time_signature"]
                tempo = session["tempo"]

        crew = await _get_composer_crew(request, user["id"])
        musicxml = crew.extract_musicxml(response_text, key_sig, time_sig, tempo)
        if not musicxml:
            raise HTTPException(status_code=422, detail="No score found in that response")
        midi = crew.musicxml_to_midi(musicxml, tempo)
        if not midi:
            raise HTTPException(status_code=500, detail="Could not render MIDI")
        return Response(
            content=midi,
            media_type="audio/midi",
            headers={"Content-Disposition": 'attachment; filename="composition.mid"'},
        )

    @app.post("/composer/convert-audio")
    async def composer_convert_audio(request: Request):
        """Render a response's score to WAV so it can be played in the browser.

        Falls back to 422 when no synth is available, so the UI can offer the
        MIDI download instead of failing silently.
        """
        user = await get_current_user(request)
        body = await request.json()
        response_text = body.get("response", "")
        if not response_text:
            raise HTTPException(status_code=400, detail="No response text")

        key_sig, time_sig, tempo = "C major", "4/4", 120
        session_id = body.get("session_id", "")
        if session_id:
            session = await get_composer_session(request.app.state.db, session_id, user["id"])
            if session:
                key_sig = session["key_signature"]
                time_sig = session["time_signature"]
                tempo = session["tempo"]

        crew = await _get_composer_crew(request, user["id"])
        musicxml = crew.extract_musicxml(response_text, key_sig, time_sig, tempo)
        if not musicxml:
            raise HTTPException(status_code=422, detail="No score found in that response")
        midi = crew.musicxml_to_midi(musicxml, tempo)
        if not midi:
            raise HTTPException(status_code=500, detail="Could not render MIDI")

        loop = asyncio.get_running_loop()
        # gpu-queue-exempt: fluidsynth renders MIDI on the CPU.
        audio = await loop.run_in_executor(None, crew.midi_to_audio, midi)
        if not audio:
            raise HTTPException(
                status_code=422,
                detail="No audio synth available on the server; download the MIDI instead",
            )
        return Response(content=audio, media_type="audio/wav")

    @app.get("/composer/compositions/{composition_id}/audio")
    async def composer_download_audio(composition_id: str, request: Request):
        """Download a saved song under its own name.

        The file on disk is ``song-<hash>.mp3`` because the name has to be
        unique and safe; what you get when you download it is the title you
        gave the composition.
        """
        user = await get_current_user(request)
        db = request.app.state.db
        comp = await get_composition(db, composition_id, user["id"])
        if not comp:
            raise HTTPException(status_code=404, detail="Composition not found")
        stored = comp.get("audio_path") or ""
        if not stored:
            raise HTTPException(status_code=404,
                                detail="No audio saved for this composition")
        try:
            path = song.resolve(stored)
        except ValueError:
            raise HTTPException(status_code=404, detail="The audio file is gone")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="The audio file is gone")
        return Response(
            content=path.read_bytes(),
            media_type="audio/mpeg",
            headers=content_disposition(f"{comp['title']}.mp3"),
        )

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
        comp = await get_composition(db, composition_id, user["id"])
        deleted = await delete_composition(db, composition_id, user["id"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Composition not found")
        # A saved song's audio lives in kept/, which nothing else ever cleans,
        # so deleting the entry has to delete the file — but only once no other
        # entry still points at it. Two rows sharing one file already exist
        # from before saving became an upsert.
        stored = (comp or {}).get("audio_path") or ""
        if stored.startswith("kept/") and stored not in await saved_audio_paths(db):
            try:
                path = song.resolve(stored)
                if path.is_file():
                    path.unlink()
                    song.audit("delete", file=stored, user=user["id"],
                               title=repr((comp or {}).get("title", "")))
            except (ValueError, OSError) as e:
                logger.warning("[song] could not remove %s: %s", stored, e)
        return {"deleted": composition_id}

    # ------------------------------------------------------------------
    # Song audio (ACE-Step)
    #
    # Text to audio. The Song tab used to offer to perform a score and to write
    # one; the scores it produced had nothing to do with what ACE-Step made,
    # which was worse than none, so notation lives in the Score tab only.
    # ------------------------------------------------------------------

    @app.post("/composer/song")
    async def composer_song(req: ComposeSongRequest, request: Request):
        """Render takes of a song — SSE streaming."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        uid = user["id"]
        db = request.app.state.db

        session = await get_composer_session(db, req.session_id, uid)
        if not session:
            raise HTTPException(status_code=404, detail="Composer session not found")
        if not song.is_installed() and not song._systemd_managed():
            raise HTTPException(status_code=503, detail=song.install_hint())

        prompt = (req.prompt or "").strip()
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="Describe the song first: genre, instruments, mood, voice.")

        takes = max(1, min(int(req.takes or 1), song.MAX_TAKES))
        model = song.resolve_model(req.model)

        async def event_stream():
            progress: queue.Queue = queue.Queue()
            lines: list[str] = []

            # One card, one queue. ACE-Step has no unload endpoint, so the
            # broker restarts it when anything else needs the GPU.
            async with GPU.reserve(Need("song", "acestep"), "song",
                                   user=uid) as res:
                async for notice in res.wait():
                    yield _sse("queue", queue_message(
                        notice["position"], notice["estimate"],
                        notice["ticket"]))

                task = asyncio.create_task(song.generate(
                    prompt=prompt,
                    lyrics=req.lyrics,
                    instrumental=req.instrumental or not (req.lyrics or "").strip(),
                    bpm=req.bpm or session["tempo"],
                    key_scale=(req.key_scale or session["key_signature"]),
                    time_signature=(req.time_signature or session["time_signature"]),
                    duration=req.duration,
                    vocal_language=req.vocal_language,
                    takes=takes,
                    seed=req.seed,
                    model=model,
                    on_progress=progress.put_nowait,
                ))
                while not task.done():
                    try:
                        line = progress.get_nowait()
                        lines.append(line)
                        yield _sse("reasoning", "⏳ " + line)
                    except queue.Empty:
                        await asyncio.sleep(0.4)
                try:
                    results = task.result()
                except Exception as e:
                    logger.exception("[song] generation failed")
                    yield _sse("error", str(e))
                    return

            rendered = []
            for i, take in enumerate(results, 1):
                rel = song.write_scratch(take["audio"])
                meta = take.get("meta") or {}
                rendered.append({
                    "take": i,
                    "file": rel,
                    "url": song.url_for(rel),
                    "seed": take.get("seed"),
                    "duration": meta.get("duration"),
                    "bytes": len(take["audio"]),
                })
            meta0 = (results[0].get("meta") or {}) if results else {}
            logger.info("[song] %d take(s) on %s for %s", len(rendered), model, uid)

            # Old scratch renders go; saved songs live in kept/ and are not
            # there to be found. The database list is a second guard.
            kept = await saved_audio_paths(db)
            # gpu-queue-exempt: deleting files, no GPU involved.
            await asyncio.get_running_loop().run_in_executor(
                None, song.prune_unsaved, None, kept)

            yield _sse("done", {
                "takes": rendered,
                "language": meta0.get("language", ""),
                "model": model,
                "engine": song.MODELS[model]["label"],
                "steps": song.MODELS[model]["steps"],
                "keep_hours": round(song.KEEP_UNSAVED_SECONDS / 3600, 1),
                "reasoning": lines,
            })

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/composer/song/save")
    async def composer_song_save(req: SongSaveRequest, request: Request):
        """Keep a take in the library under a name.

        Moves it from scratch/ to kept/, where the cleanup never looks.
        Saving the same take again updates its name rather than adding a
        second entry.
        """
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        uid = user["id"]
        db = request.app.state.db
        session = await get_composer_session(db, req.session_id, uid)
        if not session:
            raise HTTPException(status_code=404, detail="Composer session not found")

        title = (req.title or "").strip() or "Untitled song"
        try:
            rel = song.keep(req.file)
        except ValueError:
            raise HTTPException(status_code=400, detail="That is not a rendered song")
        except FileNotFoundError:
            hours = round(song.KEEP_UNSAVED_SECONDS / 3600, 1)
            raise HTTPException(
                status_code=404,
                detail=f"That render is gone — unsaved renders are kept for "
                       f"{hours:g} hours.")

        existing = await find_composition_by_audio(db, uid, rel)
        if existing:
            await rename_composition(db, existing["id"], uid, title)
            existing["title"] = title
            comp = existing
        else:
            comp = await save_composition(
                db,
                user_id=uid,
                session_id=req.session_id,
                title=title,
                description=(req.prompt or "").strip(),
                instruments=[],
                genre=session["genre"],
                key_signature=session["key_signature"],
                time_signature=session["time_signature"],
                tempo=session["tempo"],
                musicxml="",
                audio_path=rel,
            )
        song.audit("save", file=rel, title=repr(title), user=uid)
        return {"composition": comp, "url": song.url_for(rel), "file": rel}

    @app.put("/composer/compositions/{composition_id}")
    async def composer_rename_composition(
            composition_id: str, req: CompositionRename, request: Request):
        """Rename a composition in place."""
        user = await get_current_user(request)
        title = (req.title or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="A name is required")
        if not await rename_composition(request.app.state.db, composition_id,
                                        user["id"], title):
            raise HTTPException(status_code=404, detail="Composition not found")
        return {"id": composition_id, "title": title}

    @app.post("/composer/song/stop")
    async def composer_song_stop(request: Request):
        """Make the engine give the GPU back.

        Refused while a song is rendering: under systemd this is a restart,
        and restarting mid-render throws the render away.
        """
        await get_current_user(request)
        running = (GPU.snapshot() or {}).get("running") or {}
        if running.get("label") == "song":
            raise HTTPException(status_code=409,
                                detail="A song is rendering — wait for it to finish.")
        # gpu-queue-exempt: asks the engine to release, no GPU work queued.
        freed = await asyncio.get_running_loop().run_in_executor(None, song.stop)
        GPU.forget_residency()
        return {"stopped": bool(freed)}

    @app.get("/composer/song/status")
    async def composer_song_status(request: Request):
        """What the engine is, who owns it, and whether it holds the GPU."""
        await get_current_user(request)
        managed = song._systemd_managed()
        installed = managed or song.is_installed()
        loop = asyncio.get_running_loop()
        # gpu-queue-exempt: reads /health over HTTP.
        running = await loop.run_in_executor(None, song.running)
        # gpu-queue-exempt: reads /health over HTTP.
        holding = running and await loop.run_in_executor(None, song.holding_gpu)
        # gpu-queue-exempt: reads /health over HTTP.
        loaded = await loop.run_in_executor(None, song.loaded_model) if running else ""
        return {
            "installed": installed,
            "managed_by": "systemd" if managed else "app",
            "unit": f"{song.SYSTEMD_UNIT}.service" if managed else "",
            "running": running,
            "holding_gpu": holding,
            "loaded_model": loaded,
            "models": song.model_choices(),
            "max_takes": song.MAX_TAKES,
            "keep_hours": round(song.KEEP_UNSAVED_SECONDS / 3600, 1),
            "hint": "" if installed else song.install_hint(),
        }
