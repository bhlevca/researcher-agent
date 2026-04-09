"""Language Tutor API routes.

Separate endpoints from the regular chat/session routes.
Manages tutor sessions, conversations, lessons, vocabulary, quizzes,
and student appraisal.
"""

import sys
import io
import re
import json
import queue
import asyncio
import logging

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from researcher.auth import get_current_user
from researcher.config import _QueueWriter, _SESSION_ID_RE

from researcher.tutor.models import (
    TutorSessionCreate,
    TutorSessionUpdate,
    TutorChatRequest,
    LessonPlanRequest,
    QuizRequest,
    QuizSubmitRequest,
    VocabularyAddRequest,
    AppraisalRequest,
)
from researcher.tutor.storage import (
    create_tutor_session,
    list_tutor_sessions,
    get_tutor_session,
    update_tutor_session_messages,
    delete_tutor_session,
    add_vocabulary,
    add_vocabulary_batch,
    list_vocabulary,
    list_vocabulary_by_lang,
    delete_vocabulary,
    save_lesson_plan,
    list_lesson_plans,
    get_lesson_plan,
    save_quiz,
    get_quiz,
    save_quiz_results,
    list_quiz_results,
    update_vocabulary_mastery,
    get_student_stats,
    get_cross_session_stats,
)

logger = logging.getLogger(__name__)


def _validate_sid(session_id: str):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")


def _build_tutor_context(messages: list, max_messages: int = 10) -> str:
    """Build conversation context string from recent tutor messages."""
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


def _run_tutor_crew(crew, q: queue.Queue):
    """Run a tutor crew in a thread, capturing stdout."""
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


async def _stream_tutor_crew(crew, request: Request):
    """Generic SSE streamer for any tutor crew run."""
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
        if line and line[0] in "💭🔧📋✅🧠📝⚠️👁️🎯":
            return True
        if _verbose_noise.search(line):
            return False
        return True

    async def event_stream():
        cancel = request.app.state.cancel_event
        cancel.clear()
        async with semaphore:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, lambda: _run_tutor_crew(crew, q))
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
            # Strip thinking tags
            response_text = re.sub(r"<think>[\s\S]*?</think>\s*", "", response_text).strip()

            yield _sse(
                "done",
                {
                    "response": response_text,
                    "reasoning": all_lines,
                },
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def register_tutor_routes(app):
    """Attach all /tutor/* endpoints to the FastAPI app."""

    # ------------------------------------------------------------------
    # Tutor Sessions CRUD
    # ------------------------------------------------------------------

    @app.get("/tutor/sessions")
    async def tutor_list_sessions(request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        sessions = await list_tutor_sessions(db, user["id"])
        return {"sessions": sessions}

    @app.post("/tutor/sessions")
    async def tutor_create_session(req: TutorSessionCreate, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        session = await create_tutor_session(
            db, user["id"], req.name, req.target_lang, req.native_lang, req.level
        )
        return session

    @app.get("/tutor/sessions/{session_id}")
    async def tutor_get_session(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        session = await get_tutor_session(db, session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Tutor session not found")
        return session

    @app.put("/tutor/sessions/{session_id}")
    async def tutor_update_session(
        session_id: str, req: TutorSessionUpdate, request: Request
    ):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        session = await get_tutor_session(db, session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Tutor session not found")
        await update_tutor_session_messages(db, session_id, req.messages)
        return {"id": session_id, "name": req.name}

    @app.delete("/tutor/sessions/{session_id}")
    async def tutor_delete_session(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        deleted = await delete_tutor_session(db, session_id, user["id"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Tutor session not found")
        return {"deleted": session_id}

    # ------------------------------------------------------------------
    # Conversation (SSE streaming)
    # ------------------------------------------------------------------

    @app.post("/tutor/chat")
    async def tutor_chat(req: TutorChatRequest, request: Request):
        """Conversational exchange with the tutor agent — SSE streaming."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db

        session = await get_tutor_session(db, req.session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Tutor session not found")

        tutor_crew = request.app.state.tutor_crew
        context = _build_tutor_context(session["messages"])

        crew = tutor_crew.build_conversation_crew(
            message=req.message,
            context=context,
            target_lang=session["target_lang"],
            native_lang=session["native_lang"],
            level=session["level"],
        )

        return await _stream_tutor_crew(crew, request)

    # ------------------------------------------------------------------
    # Lesson Plans
    # ------------------------------------------------------------------

    @app.post("/tutor/lessons")
    async def tutor_create_lesson(req: LessonPlanRequest, request: Request):
        """Generate a lesson plan — SSE streaming."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db

        session = await get_tutor_session(db, req.session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Tutor session not found")

        tutor_crew = request.app.state.tutor_crew

        crew = tutor_crew.build_lesson_crew(
            topic=req.topic,
            lesson_type=req.lesson_type,
            target_lang=session["target_lang"],
            native_lang=session["native_lang"],
            level=session["level"],
        )

        return await _stream_tutor_crew(crew, request)

    @app.post("/tutor/lessons/save")
    async def tutor_save_lesson(request: Request):
        """Save a generated lesson plan to the database."""
        user = await get_current_user(request)
        db = request.app.state.db
        body = await request.json()

        session_id = body.get("session_id", "")
        _validate_sid(session_id)
        session = await get_tutor_session(db, session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Tutor session not found")

        tutor_crew = request.app.state.tutor_crew

        # Extract vocabulary from the lesson content and save it
        content_text = body.get("content", "")
        vocab_entries = tutor_crew.extract_vocabulary_from_response(content_text)
        if vocab_entries:
            await add_vocabulary_batch(
                db, user["id"], session_id, vocab_entries,
                target_lang=session["target_lang"],
            )

        lesson = await save_lesson_plan(
            db,
            user_id=user["id"],
            session_id=session_id,
            title=body.get("title", "Untitled Lesson"),
            topic=body.get("topic", ""),
            lesson_type=body.get("lesson_type", "mixed"),
            target_lang=session["target_lang"],
            level=session["level"],
            content={"markdown": content_text},
        )
        lesson["vocabulary_saved"] = len(vocab_entries)
        return lesson

    @app.get("/tutor/sessions/{session_id}/lessons")
    async def tutor_list_lessons(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        lessons = await list_lesson_plans(db, user["id"], session_id)
        return {"lessons": lessons}

    @app.get("/tutor/lessons/{lesson_id}")
    async def tutor_get_lesson(lesson_id: str, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        lesson = await get_lesson_plan(db, lesson_id, user["id"])
        if not lesson:
            raise HTTPException(status_code=404, detail="Lesson not found")
        return lesson

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    @app.get("/tutor/sessions/{session_id}/vocabulary")
    async def tutor_list_vocabulary(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        vocab = await list_vocabulary(db, user["id"], session_id)
        return {"vocabulary": vocab}

    @app.get("/tutor/vocabulary")
    async def tutor_list_vocabulary_by_lang(request: Request, lang: str = ""):
        """List vocabulary for a target language (cross-session)."""
        user = await get_current_user(request)
        db = request.app.state.db
        if not lang:
            raise HTTPException(status_code=400, detail="lang query parameter required")
        vocab = await list_vocabulary_by_lang(db, user["id"], lang)
        return {"vocabulary": vocab}

    @app.post("/tutor/vocabulary")
    async def tutor_add_vocabulary(req: VocabularyAddRequest, request: Request):
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        session = await get_tutor_session(db, req.session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Tutor session not found")
        target_lang = req.target_lang or session["target_lang"]
        entry = await add_vocabulary(
            db, user["id"], req.session_id,
            req.word, req.translation, req.context, req.phonetic, req.part_of_speech,
            target_lang=target_lang,
        )
        return entry

    @app.delete("/tutor/vocabulary/{vocab_id}")
    async def tutor_delete_vocabulary(vocab_id: str, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        deleted = await delete_vocabulary(db, vocab_id, user["id"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Vocabulary entry not found")
        return {"deleted": vocab_id}

    # ------------------------------------------------------------------
    # Quizzes
    # ------------------------------------------------------------------

    @app.post("/tutor/quiz/generate")
    async def tutor_generate_quiz(req: QuizRequest, request: Request):
        """Generate a quiz — SSE streaming."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db

        session = await get_tutor_session(db, req.session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Tutor session not found")

        tutor_crew = request.app.state.tutor_crew

        # Build vocabulary context from the student's saved words
        vocab = await list_vocabulary(db, user["id"], req.session_id)
        vocab_ctx = ""
        if vocab:
            vocab_lines = [f"- {v['word']} = {v['translation']}" for v in vocab[:50]]
            vocab_ctx = (
                "The student has learned these vocabulary words.\n"
                "Include some of them in the quiz:\n" + "\n".join(vocab_lines)
            )

        # Build lesson context if scoped to a lesson
        lesson_ctx = ""
        if req.lesson_id:
            lesson = await get_lesson_plan(db, req.lesson_id, user["id"])
            if lesson and lesson.get("content"):
                content = lesson["content"]
                md = content.get("markdown", "")[:2000] if isinstance(content, dict) else str(content)[:2000]
                lesson_ctx = f"Base the quiz on this lesson:\n{md}"

        crew = tutor_crew.build_quiz_crew(
            quiz_type=req.quiz_type,
            num_questions=req.num_questions,
            target_lang=session["target_lang"],
            native_lang=session["native_lang"],
            level=session["level"],
            vocabulary_context=vocab_ctx,
            lesson_context=lesson_ctx,
        )

        return await _stream_tutor_crew(crew, request)

    @app.post("/tutor/quiz/save")
    async def tutor_save_quiz(request: Request):
        """Save a generated quiz (the JSON questions) for later answering."""
        user = await get_current_user(request)
        db = request.app.state.db
        body = await request.json()

        session_id = body.get("session_id", "")
        _validate_sid(session_id)

        tutor_crew = request.app.state.tutor_crew
        response_text = body.get("response", "")
        questions = tutor_crew.extract_quiz_json(response_text)
        if not questions:
            raise HTTPException(status_code=400, detail="Could not parse quiz JSON from response")

        quiz = await save_quiz(
            db,
            user_id=user["id"],
            session_id=session_id,
            quiz_type=body.get("quiz_type", "mixed"),
            questions=questions,
            lesson_id=body.get("lesson_id"),
        )
        quiz["questions"] = questions
        return quiz

    @app.post("/tutor/quiz/submit")
    async def tutor_submit_quiz(req: QuizSubmitRequest, request: Request):
        """Submit answers for a quiz, grade them, and return feedback."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db

        quiz = await get_quiz(db, req.quiz_id, user["id"])
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")

        questions = quiz["questions"]
        score = 0
        graded = []

        for ans in req.answers:
            qid = ans.get("question_id", -1)
            student_answer = ans.get("answer", "").strip()
            if 0 <= qid < len(questions):
                q = questions[qid]
                correct = q.get("correct_answer", "").strip()
                is_correct = student_answer.lower() == correct.lower()
                if is_correct:
                    score += 1
                graded.append({
                    "question_id": qid,
                    "student_answer": student_answer,
                    "correct_answer": correct,
                    "is_correct": is_correct,
                })

        total = len(questions)
        pct = round(score / total * 100, 1) if total > 0 else 0
        feedback = f"Score: {score}/{total} ({pct}%)"

        await save_quiz_results(db, req.quiz_id, user["id"], graded, score, feedback)

        # Update vocabulary mastery based on quiz answers
        vocab = await list_vocabulary(db, user["id"], req.session_id)
        vocab_map = {v["word"].lower(): v["id"] for v in vocab}
        for g in graded:
            qid = g["question_id"]
            if 0 <= qid < len(questions):
                q = questions[qid]
                # Try to find the word in vocabulary to update mastery
                q_text = q.get("question", "").lower()
                for word, vid in vocab_map.items():
                    if word in q_text or word in g.get("correct_answer", "").lower():
                        await update_vocabulary_mastery(db, vid, g["is_correct"])
                        break

        return {
            "quiz_id": req.quiz_id,
            "score": score,
            "total": total,
            "percentage": pct,
            "feedback": feedback,
            "details": graded,
        }

    @app.get("/tutor/sessions/{session_id}/quizzes")
    async def tutor_list_quizzes(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        quizzes = await list_quiz_results(db, user["id"], session_id)
        return {"quizzes": quizzes}

    @app.get("/tutor/quiz/{quiz_id}")
    async def tutor_get_quiz(quiz_id: str, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        quiz = await get_quiz(db, quiz_id, user["id"])
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")
        return quiz

    # ------------------------------------------------------------------
    # Appraisal
    # ------------------------------------------------------------------

    @app.post("/tutor/appraisal")
    async def tutor_appraisal(req: AppraisalRequest, request: Request):
        """Generate a student progress appraisal — SSE streaming."""
        _validate_sid(req.session_id)
        user = await get_current_user(request)
        db = request.app.state.db

        session = await get_tutor_session(db, req.session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Tutor session not found")

        # Current session stats
        stats = await get_student_stats(db, user["id"], req.session_id)

        # Cross-session stats for same target language
        cross = await get_cross_session_stats(
            db, user["id"], session["target_lang"]
        )
        stats["cross_session"] = cross

        stats_str = json.dumps(stats, indent=2)

        # Get recent messages for context
        recent = _build_tutor_context(session["messages"], max_messages=20)

        tutor_crew = request.app.state.tutor_crew

        crew = tutor_crew.build_appraisal_crew(
            target_lang=session["target_lang"],
            native_lang=session["native_lang"],
            level=session["level"],
            stats=stats_str,
            recent_messages=recent,
        )

        return await _stream_tutor_crew(crew, request)

    # ------------------------------------------------------------------
    # Student Stats
    # ------------------------------------------------------------------

    @app.get("/tutor/sessions/{session_id}/stats")
    async def tutor_stats(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        session = await get_tutor_session(db, session_id, user["id"])
        if not session:
            raise HTTPException(status_code=404, detail="Tutor session not found")
        stats = await get_student_stats(db, user["id"], session_id)
        stats["session"] = {
            "target_lang": session["target_lang"],
            "native_lang": session["native_lang"],
            "level": session["level"],
        }
        return stats

    # ------------------------------------------------------------------
    # Translator
    # ------------------------------------------------------------------

    @app.post("/tutor/translate")
    async def tutor_translate(request: Request):
        """Translate text between target and native language using the LLM."""
        await get_current_user(request)
        body = await request.json()
        text = (body.get("text") or "").strip()
        source_lang = body.get("source_lang", "").strip()
        target_lang = body.get("target_lang", "").strip()

        if not text:
            raise HTTPException(status_code=400, detail="Empty text")
        if not source_lang or not target_lang:
            raise HTTPException(status_code=400, detail="source_lang and target_lang required")

        tutor_crew = request.app.state.tutor_crew

        prompt = (
            f"Translate the following text from {source_lang} to {target_lang}.\n"
            f"Return ONLY the translation, nothing else — no explanations, "
            f"no quotation marks, no prefixes.\n\n"
            f"{text}"
        )

        loop = asyncio.get_running_loop()

        def _call_llm():
            from litellm import completion
            resp = completion(
                model=tutor_crew._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                api_base="http://localhost:11434",
            )
            return resp.choices[0].message.content.strip()

        try:
            translation = await loop.run_in_executor(None, _call_llm)
            # Strip thinking tags if present (qwen3/deepseek-r1)
            translation = re.sub(r'<think>.*?</think>', '', translation, flags=re.DOTALL).strip()
            return {"translation": translation, "source_lang": source_lang, "target_lang": target_lang}
        except Exception as e:
            logger.error("Translation failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Translation failed: {e}")
