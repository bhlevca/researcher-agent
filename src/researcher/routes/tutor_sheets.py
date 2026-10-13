"""Printable sheets for the language tutor: homework, tests, exams, challenges.

Generation is synchronous and fast — the exercises come from language packs, not
from a model — so these are plain endpoints rather than the streaming ones the
tutor's chat needs. Nothing here touches the conversational routes, the voice
path or pronunciation scoring; this is additive.

Sheets are stored as a recipe and rebuilt on demand, so a seed is a sheet.
"""

import random
import logging

from fastapi import HTTPException, Request
from fastapi.responses import Response

from researcher.auth import get_current_user
from researcher.tutor import exercises as X
from researcher.tutor import printing
from researcher.tutor.models import SheetRequest, SheetSaveRequest
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
        req = _req_from(rec)
        sections, _ = _build(req)
        pdf = printing.render_pdf(sections, title=rec["title"],
                                  subtitle=_subtitle(req), kind=rec["kind"],
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
