"""Printable sheets for the language tutor: homework, tests, exams, challenges.

Generation is synchronous and fast — the exercises come from language packs, not
from a model — so these are plain endpoints rather than the streaming ones the
tutor's chat needs. Nothing here touches the conversational routes, the voice
path or pronunciation scoring; this is additive.

Sheets are stored as a recipe and rebuilt on demand, so a seed is a sheet.
"""

import asyncio
import random
import logging

from fastapi import HTTPException, Request
from fastapi.responses import Response

from researcher.auth import get_current_user
from researcher.tutor import exercises as X
from researcher.tutor import printing
from researcher.tutor import story as story_mod
from researcher.tutor.crew import TutorCrew
from researcher.tutor.models import (
    SheetRequest,
    SheetSaveRequest,
    StoryRequest,
    StorySheetRequest,
)
from researcher.tutor.storage import (
    save_sheet,
    list_sheets,
    get_sheet,
    delete_sheet,
)

logger = logging.getLogger(__name__)

# A sheet has to fit in a school bag and a marking session.
_MAX_PER_SECTION = 40
_MAX_SECTIONS = 8

# A sensible instruction per skill, so a sheet is usable without the teacher
# having to write one. Overridable per section.
_INSTRUCTION = {
    "conjugation": "Write the correct form of the verb in brackets.",
    "tense_transform": "Rewrite each sentence in the tense shown.",
    "gender_article": "Complete with the correct article.",
    "plural": "Write the plural.",
    "adjective_agreement": "Write the adjective in its correct form.",
    "vocabulary": "Translate into the target language.",
    "german_case": "Insert the article in the case shown.",
}


def _build(req: SheetRequest) -> tuple[list, int]:
    """Build the sections for a request. Returns (sections, seed)."""
    if not X.pack_code(req.language):
        raise HTTPException(
            status_code=400,
            detail=f"No exercise pack for '{req.language}'. Printable sheets "
                   f"exist for {', '.join(sorted(X.SUPPORTED))}.")
    if not req.sections:
        raise HTTPException(status_code=400, detail="Choose at least one skill")
    if len(req.sections) > _MAX_SECTIONS:
        raise HTTPException(status_code=400,
                            detail=f"At most {_MAX_SECTIONS} sections")

    seed = req.seed if req.seed is not None else random.randrange(1, 10**9)
    names = {s["id"]: s["name"] for s in X.available_skills(req.language)}

    out = []
    for i, sec in enumerate(req.sections):
        count = max(1, min(int(sec.count), _MAX_PER_SECTION))
        try:
            items = X.generate(sec.skill, req.language, req.level, count,
                               seed=seed + i)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not items:
            continue
        out.append(printing.Section(
            title=sec.title or names.get(sec.skill, sec.skill),
            instruction=sec.instruction or _INSTRUCTION.get(sec.skill, ""),
            exercises=items,
            marks_each=max(1, int(sec.marks_each))))
    if not out:
        raise HTTPException(status_code=500, detail="No exercises generated")
    return out, seed


def _subtitle(req: SheetRequest) -> str:
    return f"Level {req.level} · {req.language}"


def _title(req: SheetRequest) -> str:
    if req.title:
        return req.title
    return f"{req.language} — {req.kind.title()}"


def _tutor_crew(request: Request, uid: str) -> TutorCrew:
    """The per-user TutorCrew, shared with the conversational routes."""
    crews = getattr(request.app.state, "tutor_crews", None)
    if crews is None:
        crews = request.app.state.tutor_crews = {}
    if uid not in crews:
        crews[uid] = TutorCrew(model=request.app.state.current_model)
    return crews[uid]


def _pdf_response(pdf: bytes | None, filename: str) -> Response:
    if not pdf:
        raise HTTPException(
            status_code=503,
            detail="PDF generation unavailable — pdflatex is not installed. "
                   "The on-screen preview still works.")
    return Response(
        content=pdf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def register_tutor_sheet_routes(app):
    """Attach ``/tutor/sheets/*`` endpoints to *app*."""

    @app.get("/tutor/sheets/skills")
    async def skills(request: Request, language: str = "French"):
        """Which languages have packs, and which skills each offers."""
        await get_current_user(request)
        return {
            "languages": sorted(X.SUPPORTED),
            "language": language,
            "skills": X.available_skills(language),
            "levels": X._CEFR,
            "kinds": ["homework", "test", "exam", "challenge"],
            "pdf_available": printing.latex_available(),
            # Said plainly, because it is the reason to trust the answer key.
            "deterministic": bool(X.pack_code(language)),
        }

    @app.post("/tutor/sheets/generate")
    async def generate_sheet(req: SheetRequest, request: Request):
        """Exercises plus an HTML preview. Nothing is stored."""
        await get_current_user(request)
        sections, seed = _build(req)
        return {
            "seed": seed,
            "language": req.language,
            "level": req.level,
            "kind": req.kind,
            "title": _title(req),
            "subtitle": _subtitle(req),
            "total_marks": sum(s.marks for s in sections),
            "sections": [
                {"title": s.title, "instruction": s.instruction,
                 "marks_each": s.marks_each,
                 "exercises": [e.as_dict() for e in s.exercises]}
                for s in sections],
            "html": printing.render_html(sections, title=_title(req),
                                         subtitle=_subtitle(req),
                                         kind=req.kind, answers=False),
            "answers_html": printing.render_html(sections, title=_title(req),
                                                 subtitle=_subtitle(req),
                                                 kind=req.kind, answers=True),
        }

    @app.post("/tutor/sheets/pdf")
    async def sheet_pdf(req: SheetRequest, request: Request):
        await get_current_user(request)
        sections, seed = _build(req)
        pdf = printing.render_pdf(sections, title=_title(req),
                                  subtitle=_subtitle(req), kind=req.kind,
                                  answers=False, student=req.student)
        return _pdf_response(pdf, f"{req.language}-{req.kind}-{seed}.pdf")

    @app.post("/tutor/sheets/answers")
    async def answers_pdf(req: SheetRequest, request: Request):
        """The key for the same sheet. Same seed, same exercises."""
        await get_current_user(request)
        sections, seed = _build(req)
        pdf = printing.render_pdf(sections, title=_title(req),
                                  subtitle=_subtitle(req), kind=req.kind,
                                  answers=True)
        return _pdf_response(pdf, f"{req.language}-{req.kind}-answers-{seed}.pdf")

    # -- Story sheets -------------------------------------------------------
    #
    # The one part of the language tutor that genuinely needs a model: a story
    # cannot be derived from a table. Two consequences, both deliberate — the
    # key prints "Model answer", and the generated text is carried back by the
    # client rather than regenerated, because a second call writes a different
    # story and the reprint would no longer match the paper.

    @app.post("/tutor/sheets/story")
    async def make_story(req: StoryRequest, request: Request):
        """Generate a story, comprehension questions and a summary task."""
        user = await get_current_user(request)
        tutor = _tutor_crew(request, user["id"])

        crew = tutor.build_story_sheet_crew(
            topic=req.topic.strip() or "everyday life",
            target_lang=req.language,
            native_lang="English",
            level=req.level,
            num_questions=max(1, min(int(req.num_questions), 12)),
            word_count=max(60, min(int(req.word_count), 600)),
            summary_words=max(20, min(int(req.summary_words), 150)))

        timeout = (tutor.get_llm_params() or {}).get("crew_timeout_sec") or 600
        loop = asyncio.get_running_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: str(crew.kickoff())),
                timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504,
                                detail=f"The model did not answer within "
                                       f"{timeout}s")
        except Exception as e:
            logger.warning("[tutor] story generation failed: %s", e)
            raise HTTPException(status_code=502,
                                detail=f"Story generation failed: {e}")

        try:
            content = story_mod.parse_story(raw)
        except ValueError as e:
            # Better to say so than to print a sheet with empty questions.
            raise HTTPException(status_code=502, detail=str(e))
        return {"content": content, "language": req.language,
                "level": req.level}

    def _story_sections(req: StorySheetRequest):
        try:
            content = story_mod._clean(req.content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad story: {e}")
        if not content.get("questions") and not content.get("story"):
            raise HTTPException(status_code=400, detail="Empty story")
        code = X.pack_code(req.language) or req.language[:2].lower()
        return story_mod.to_sections(content, level=req.level, lang=code)

    def _story_titles(req: StorySheetRequest) -> tuple[str, str]:
        title = req.title or req.content.get("title") or f"{req.language} — Reading"
        return title, f"Level {req.level} · {req.language}"

    @app.post("/tutor/sheets/story/pdf")
    async def story_pdf(req: StorySheetRequest, request: Request):
        await get_current_user(request)
        sections = _story_sections(req)
        title, subtitle = _story_titles(req)
        pdf = printing.render_pdf(sections, title=title, subtitle=subtitle,
                                  kind=req.kind, answers=False,
                                  student=req.student)
        return _pdf_response(pdf, f"{req.language}-story.pdf")

    @app.post("/tutor/sheets/story/answers")
    async def story_answers(req: StorySheetRequest, request: Request):
        await get_current_user(request)
        sections = _story_sections(req)
        title, subtitle = _story_titles(req)
        pdf = printing.render_pdf(sections, title=title, subtitle=subtitle,
                                  kind=req.kind, answers=True)
        return _pdf_response(pdf, f"{req.language}-story-answers.pdf")

    @app.post("/tutor/sheets/story/save")
    async def story_save(req: StorySheetRequest, request: Request):
        """Store the story verbatim, so a reprint is the same paper."""
        user = await get_current_user(request)
        _story_sections(req)                  # validate before storing
        title, _ = _story_titles(req)
        return await save_sheet(
            request.app.state.db, user["id"], title=title,
            language=req.language, level=req.level, kind=req.kind,
            sections=[], seed=0, student=req.student, notes=req.notes,
            content=story_mod._clean(req.content))

    # -- Saved sheets -------------------------------------------------------

    @app.post("/tutor/sheets")
    async def save(req: SheetSaveRequest, request: Request):
        user = await get_current_user(request)
        _, seed = _build(req)                 # validate before storing
        return await save_sheet(
            request.app.state.db, user["id"], title=_title(req),
            language=req.language, level=req.level, kind=req.kind,
            sections=[s.model_dump() for s in req.sections], seed=seed,
            student=req.student, notes=req.notes)

    @app.get("/tutor/sheets")
    async def listing(request: Request):
        user = await get_current_user(request)
        return {"sheets": await list_sheets(request.app.state.db, user["id"])}

    def _req_from(rec: dict) -> SheetRequest:
        return SheetRequest(language=rec["language"], level=rec["level"],
                            kind=rec["kind"], title=rec["title"],
                            student=rec["student"], seed=rec["seed"],
                            sections=rec["sections"])

    @app.get("/tutor/sheets/{sid}")
    async def one(sid: str, request: Request):
        user = await get_current_user(request)
        rec = await get_sheet(request.app.state.db, user["id"], sid)
        if not rec:
            raise HTTPException(status_code=404, detail="Sheet not found")
        if rec.get("content"):
            return rec                       # a story sheet carries its own text
        req = _req_from(rec)
        sections, _ = _build(req)
        rec["exercises"] = [
            {"title": s.title, "exercises": [e.as_dict() for e in s.exercises]}
            for s in sections]
        rec["html"] = printing.render_html(sections, title=rec["title"],
                                           subtitle=_subtitle(req),
                                           kind=rec["kind"])
        return rec

    @app.get("/tutor/sheets/{sid}/pdf")
    async def saved_pdf(sid: str, request: Request, answers: bool = False):
        """Reprint a saved sheet, or its key, from the stored recipe."""
        user = await get_current_user(request)
        rec = await get_sheet(request.app.state.db, user["id"], sid)
        if not rec:
            raise HTTPException(status_code=404, detail="Sheet not found")
        if rec.get("content"):
            # A story sheet: reprint the stored text, never regenerate it.
            code = X.pack_code(rec["language"]) or rec["language"][:2].lower()
            sections = story_mod.to_sections(rec["content"],
                                             level=rec["level"], lang=code)
            subtitle = f"Level {rec['level']} · {rec['language']}"
        else:
            req = _req_from(rec)
            sections, _ = _build(req)
            subtitle = _subtitle(req)
        pdf = printing.render_pdf(sections, title=rec["title"],
                                  subtitle=subtitle, kind=rec["kind"],
                                  answers=answers,
                                  student="" if answers else rec["student"])
        kind = "answers" if answers else "sheet"
        name = f"{rec['title']}-{kind}.pdf".replace(" ", "-")
        return _pdf_response(pdf, name)

    @app.delete("/tutor/sheets/{sid}")
    async def remove(sid: str, request: Request):
        user = await get_current_user(request)
        if not await delete_sheet(request.app.state.db, user["id"], sid):
            raise HTTPException(status_code=404, detail="Sheet not found")
        return {"deleted": sid}
