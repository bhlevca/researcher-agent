"""SSE streaming /chat, /chat/cancel, /chat/continue endpoints."""

import sys
import io
import re
import json
import asyncio
import queue
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
from researcher.ingestion import get_file_context
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


def register_chat_routes(app):
    """Attach /chat, /chat/cancel, /chat/continue to the FastAPI app."""

    @app.post("/chat")
    async def chat(req: ChatRequest, request: Request):
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

        # Ensure image params from DB are applied before generation
        cursor = await db.execute(
            "SELECT image_params FROM users WHERE id = ?", (user["id"],)
        )
        row = await cursor.fetchone()
        load_image_params_from_json(row[0] if row and row[0] else None)

        crew = research_crew.build_crew(topic)
        semaphore = request.app.state.crew_semaphore

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
            cancel = request.app.state.cancel_event
            cancel.clear()
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
        """Signal the running crew to stop. The SSE stream will close gracefully."""
        request.app.state.cancel_event.set()
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
