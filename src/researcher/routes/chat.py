"""SSE streaming /chat, /chat/cancel, /chat/continue endpoints."""

import sys
import io
import re
import json
import asyncio
import queue
import threading
import logging

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from researcher.config import (
    ChatRequest,
    ContinueRequest,
    _QueueWriter,
    _maybe_unload_ollama,
)
from researcher.postprocess import (
    _postprocess,
    _is_incomplete,
    _extract_usage,
    _build_conversation_context,
)
from researcher.auth import get_current_user
from researcher.ingestion import (
    get_file_context,
    get_file_texts,
    is_extraction_request,
)
from researcher.image import load_image_params_from_json
from researcher.crew import ResearchCrew

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
    GoalAchievedEarlyEvent,
)

logger = logging.getLogger(__name__)


def _make_step_callback(q_ref):
    """Return a callback that puts structured reasoning into the queue."""

    def _step_cb(step):
        sys.stderr.write(f"[STEP_CB] type={type(step).__name__}\n")
        sys.stderr.flush()
        if isinstance(step, AgentAction):
            thought = (step.thought or "").strip()
            if thought:
                for line in thought.split("\n"):
                    line = line.strip()
                    if line:
                        q_ref.put(f"💭 {line}")
            tool_name = step.tool or ""
            tool_input = step.tool_input or ""
            if tool_name:
                q_ref.put(f"🔧 Using {tool_name}: {str(tool_input)[:200]}")
            result = str(step.result or "")[:300]
            if result:
                q_ref.put(f"📋 Result: {result}")
        elif isinstance(step, AgentFinish):
            thought = (step.thought or "").strip()
            if thought:
                for line in thought.split("\n"):
                    line = line.strip()
                    if line:
                        q_ref.put(f"💭 {line}")
            q_ref.put("✅ Composing final answer...")
        else:
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
        plan = (event.plan or "").strip()
        if plan:
            for line in plan.split("\n"):
                line = line.strip()
                if line:
                    q_ref.put(f"📝 {line}")

    @crewai_event_bus.on(AgentReasoningFailedEvent)
    def on_reasoning_failed(source, event):
        q_ref.put(f"⚠️ Reasoning error: {str(event.error)[:200]}")

    @crewai_event_bus.on(ToolUsageErrorEvent)
    def on_tool_error(source, event):
        q_ref.put(f"⚠️ Tool error ({event.tool_name}): {str(event.error)[:200]}")

    @crewai_event_bus.on(StepObservationCompletedEvent)
    def on_observation(source, event):
        info = (event.key_information_learned or "").strip()
        if info:
            q_ref.put(f"👁️ Observed: {info[:300]}")

    @crewai_event_bus.on(GoalAchievedEarlyEvent)
    def on_goal_early(source, event):
        q_ref.put(
            f"🎯 Goal achieved early (skipping {event.steps_remaining} steps)"
        )

    return [
        (AgentReasoningStartedEvent, on_reasoning_started),
        (AgentReasoningCompletedEvent, on_reasoning_completed),
        (AgentReasoningFailedEvent, on_reasoning_failed),
        (ToolUsageErrorEvent, on_tool_error),
        (StepObservationCompletedEvent, on_observation),
        (GoalAchievedEarlyEvent, on_goal_early),
    ]


# Explicit media prefixes. Deliberately narrow — an accidental match would send
# a real question to a diffusion model instead of the agent.
_MEDIA_PREFIX_RE = re.compile(
    r"^\s*(video|clip|animation|image|img|picture|photo|draw)\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_VIDEO_WORDS = {"video", "clip", "animation"}


def _match_media_prefix(message: str) -> tuple[str, str] | None:
    """Return ("video"|"image", prompt) when the message opens with a prefix."""
    m = _MEDIA_PREFIX_RE.match(message or "")
    if not m:
        return None
    word, prompt = m.group(1).lower(), m.group(2).strip()
    if not prompt:
        return None
    return ("video" if word in _VIDEO_WORDS else "image"), prompt


def register_chat_routes(app):
    """Attach /chat, /chat/cancel, /chat/continue to the FastAPI app."""

    @app.post("/chat")
    async def chat(req: ChatRequest, request: Request):
        user = await get_current_user(request)

        uid = user["id"]
        db = request.app.state.db

        # ── Fast path: transcription request + attachment ────────────────────
        # "OCR this", "extract the text", "give me the text" over an attached
        # scan is a conversion job, not a conversation. The text was already
        # produced at upload, so return it directly. Routing it through the crew
        # costs a decomposition call, a planning call and a full ReAct loop —
        # minutes — just to have an LLM retype text we already hold, with the
        # added risk that it paraphrases a document whose exact wording matters.
        if req.file_ids and is_extraction_request(req.message):
            files = await get_file_texts(db, uid, req.file_ids)
            have_text = [f for f in files if f["text"]]

            def _sse_direct(event: str, data) -> str:
                return f"event: {event}\ndata: {json.dumps(data)}\n\n"

            async def direct_stream():
                if not have_text:
                    names = ", ".join(f["filename"] for f in files) or "the attachment"
                    yield _sse_direct("done", {
                        "response": (
                            f"No text could be extracted from {names}. If it is a "
                            "scanned PDF or photo, check that OCR is enabled and "
                            "the OCR model is pulled, then re-upload."
                        ),
                        "reasoning": ["Direct extraction — no agent involved"],
                        "token_usage": {},
                        "incomplete": False,
                    })
                    return
                chunks = []
                for f in have_text:
                    chunks.append(
                        f"### {f['filename']}\n\n{f['text']}\n\n"
                        f"[⬇ Download as DOCX](/files/{f['id']}/export?format=docx) · "
                        f"[⬇ ODT](/files/{f['id']}/export?format=odt)"
                    )
                yield _sse_direct("done", {
                    "response": "\n\n---\n\n".join(chunks),
                    "reasoning": [
                        f"Returned stored text for {len(have_text)} file(s) directly — "
                        "no LLM call, no agent run."
                    ],
                    "token_usage": {},
                    "incomplete": False,
                })

            logger.info(
                "[chat] Extraction fast path: %d file(s), bypassing crew", len(files)
            )
            return StreamingResponse(direct_stream(), media_type="text/event-stream")

        # ── Fast path: explicit media prefix ─────────────────────────────────
        # "video: <prompt>" or "image: <prompt>" sends the text STRAIGHT to the
        # generator, skipping planning and the agent's prompt rewriting. Opt-in
        # by design: without a prefix the normal agent path is untouched, so
        # nothing that works today changes.
        #
        # This buys two things — no ~30s planning round trip, and the prompt
        # reaches the model verbatim, which matters when iterating on wording.
        # Note the image path still self-critiques: the refine loop lives inside
        # the tool, not in the agent.
        media = _match_media_prefix(req.message)
        if media:
            kind, direct_prompt = media

            def _sse_media(event: str, data) -> str:
                return f"event: {event}\ndata: {json.dumps(data)}\n\n"

            if uid not in request.app.state.crew_semaphores:
                request.app.state.crew_semaphores[uid] = asyncio.Semaphore(1)
            semaphore = request.app.state.crew_semaphores[uid]

            async def media_stream():
                # Hold the per-user semaphore: this is GPU work, and letting it
                # run alongside a crew would put two models on one card.
                async with semaphore:
                    yield _sse_media(
                        "reasoning",
                        f"🎬 Direct {kind} generation (agent bypassed): {direct_prompt[:80]}",
                    )
                    loop = asyncio.get_running_loop()
                    if kind == "video":
                        from researcher.video import _generate_video_once as _gen
                    else:
                        from researcher.image import generate_ai_image as _tool
                        _gen = getattr(_tool, "func", _tool)
                    try:
                        out = await loop.run_in_executor(None, _gen, direct_prompt)
                    except Exception as e:
                        yield _sse_media("error", f"{kind} generation failed: {e}")
                        return
                    yield _sse_media("done", {
                        "response": out,
                        "reasoning": [f"Direct {kind} generation — no agent, no planning"],
                        "token_usage": {},
                        "incomplete": False,
                    })

            logger.info("[chat] Media fast path: kind=%s", kind)
            return StreamingResponse(media_stream(), media_type="text/event-stream")

        # Lazy-init per-user semaphore and cancel event
        if uid not in request.app.state.crew_semaphores:
            request.app.state.crew_semaphores[uid] = asyncio.Semaphore(1)
        if uid not in request.app.state.cancel_events:
            request.app.state.cancel_events[uid] = threading.Event()

        # Ensure runtime model matches the user's saved model.
        cursor_model = await db.execute(
            "SELECT model, llm_params FROM users WHERE id = ?", (uid,)
        )
        row_model = await cursor_model.fetchone()
        saved_model = row_model[0] if row_model and row_model[0] else None
        effective_model = saved_model or request.app.state.current_model

        existing_crew = request.app.state.research_crews.get(uid)
        if existing_crew is None or effective_model != request.app.state.current_model:
            new_crew = ResearchCrew(model=effective_model)
            if row_model and row_model[1]:
                try:
                    new_crew.update_llm_params(json.loads(row_model[1]))
                except (json.JSONDecodeError, TypeError):
                    pass
            request.app.state.research_crews[uid] = new_crew
            request.app.state.current_model = effective_model

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

        research_crew = request.app.state.research_crews[uid]

        # Ensure image params from DB are applied before generation
        cursor = await db.execute(
            "SELECT image_params FROM users WHERE id = ?", (uid,)
        )
        row = await cursor.fetchone()
        load_image_params_from_json(row[0] if row and row[0] else None)

        # Pass the raw message so the split heuristic sizes the *request*, not
        # the attachments and history bolted onto it.
        crew = research_crew.build_crew(topic, request_text=req.message)
        semaphore = request.app.state.crew_semaphores[uid]

        q: queue.Queue = queue.Queue()

        def _run_crew():
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
                research_crew.reset_memory()

        def _sse(event: str, data) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        async def event_stream():
            cancel = request.app.state.cancel_events[uid]
            cancel.clear()
            timeout_sec = research_crew.get_llm_params().get(
                "crew_timeout_sec", request.app.state.crew_timeout
            )
            async with semaphore:
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, _run_crew)
                all_lines: list[str] = []

                _skip_history = False

                _verbose_noise = re.compile(
                    r"^(Agent:|Task:|Thought:|Action:|Action Input:|Observation:|"
                    r"Entering new CrewAgentExecutor|Finished chain|"
                    r"> |I encountered an error|Tool .* accepts these inputs|"
                    r"Tool Name:|Tool Arguments:|Tool Description:|"
                    r"\[1m|> Entering|> Finished|Moving on then)",
                    re.IGNORECASE,
                )

                def _should_show(line: str) -> bool:
                    nonlocal _skip_history
                    if "Previous conversation:" in line:
                        _skip_history = True
                        return False
                    if _skip_history:
                        if "New request:" in line:
                            _skip_history = False
                        return False
                    if line and line[0] in "💭🔧📋✅🧠📝⚠️👁️🎯":
                        return True
                    if _verbose_noise.search(line):
                        return False
                    return True

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

                response_text = _postprocess(
                    research_crew.postprocess(result), "\n".join(all_lines)
                )
                usage = _extract_usage(result)
                incomplete = _is_incomplete(response_text, "\n".join(all_lines))

                yield _sse(
                    "done",
                    {
                        "response": response_text,
                        "reasoning": all_lines,
                        "token_usage": usage,
                        "incomplete": incomplete,
                    },
                )

                _maybe_unload_ollama()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/chat/cancel")
    async def chat_cancel(request: Request):
        """Signal the requesting user's running crew to stop."""
        user = await get_current_user(request)
        uid = user["id"]
        cancel = request.app.state.cancel_events.get(uid)
        if cancel:
            cancel.set()
        return {"status": "cancelled"}

    @app.post("/chat/continue")
    async def chat_continue(req: ContinueRequest, request: Request):
        """Continue an incomplete agent response by calling the LLM directly."""
        import litellm

        user = await get_current_user(request)

        file_context = ""
        if req.file_ids:
            db = request.app.state.db
            file_context = await get_file_context(db, user["id"], req.file_ids)

        model = request.app.state.current_model
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
            + "Original user request:\n"
            + req.original_query
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
            logger.info(
                "[chat/continue] Calling LLM model=%s, query_len=%d, partial_len=%d",
                litellm_model,
                len(req.original_query),
                len(req.partial_response),
            )
            response = await loop.run_in_executor(None, _call_llm)
            text = response.choices[0].message.content or ""
            logger.info("[chat/continue] LLM returned %d chars", len(text))
            text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                }
            return {
                "response": text,
                "reasoning": ["Continued via direct LLM call (bypassed agent tools)"],
                "token_usage": usage,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            _maybe_unload_ollama()
