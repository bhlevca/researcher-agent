"""Math Tutor API routes.

Generation is synchronous and fast — SymPy and matplotlib, no model call — so
these are plain endpoints rather than the streaming ones the chat routes need.
A sheet of forty questions with figures takes well under a second.

PDFs are built on demand from the stored recipe rather than kept as blobs: the
generators are deterministic, so a seed is a worksheet.
"""

import json
import random
import logging

from fastapi import HTTPException, Request
from fastapi.responses import Response

from researcher.auth import get_current_user
from researcher import mathtutor
from researcher.mathtutor import generators, worksheet
from researcher.mathtutor.models import GenerateRequest, SaveRequest, MarkRequest
from researcher.mathtutor.storage import (
    save_worksheet,
    list_worksheets,
    get_worksheet,
    delete_worksheet,
    record_results,
    topic_performance,
)

logger = logging.getLogger(__name__)

# A sheet has to fit in a school bag and a marking session.
_MAX_COUNT = 60


def _resolve_grade(grade: str) -> str:
    """Accept either naming: "grade8", "MYP3", "DP_AA_HL", or "Grade 8".

    The two schemes name the same thing, and a parent who thinks in IB codes
    should not have to translate before using the tool."""
    cur = mathtutor.curriculum()
    want = (grade or "").strip()
    if want in cur:
        return want
    folded = want.upper().replace(" ", "").replace("-", "_")
    for gid, g in cur.items():
        if folded in (gid.upper(), str(g.get("ib", "")).upper()):
            return gid
    # "Grade 8" / "8"
    digits = "".join(ch for ch in want if ch.isdigit())
    if digits and f"grade{digits}" in cur:
        return f"grade{digits}"
    raise HTTPException(status_code=400, detail=f"Unknown grade or IB level '{grade}'")


def _topic_names() -> dict[str, str]:
    """topic id -> human name, taken from the curriculum."""
    names: dict[str, str] = {}
    for grade in mathtutor.curriculum().values():
        for t in grade["topics"]:
            names.setdefault(t["id"], t["name"])
    return names


def _build(req: GenerateRequest) -> tuple[list, int]:
    """Generate the problems for a request. Returns (problems, seed)."""
    cur = mathtutor.curriculum()
    grade = _resolve_grade(req.grade)
    req.grade = grade                 # so callers get the canonical id back

    topics = [t for t in req.topics if t]
    if not topics:
        topics = [t["id"] for t in cur[grade]["topics"]]

    known = set(generators.available_topics())
    unknown = [t for t in topics if t not in known]
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"No generator for: {', '.join(unknown)}")

    count = max(1, min(int(req.count), _MAX_COUNT))
    # A seed of None would make the sheet unreproducible, so one is always
    # chosen and handed back to the caller.
    seed = req.seed if req.seed is not None else random.randrange(1, 10**9)

    problems: list = []
    if req.mix and len(topics) > 1:
        # Spread the total across topics; the remainder goes to the first few
        # so the count the user asked for is the count they get.
        per, extra = divmod(count, len(topics))
        for i, t in enumerate(topics):
            n = per + (1 if i < extra else 0)
            if n:
                problems += generators.generate(t, req.grade, n, seed=seed + i)
    else:
        for i, t in enumerate(topics):
            problems += generators.generate(t, req.grade, count, seed=seed + i)

    if not problems:
        raise HTTPException(status_code=500,
                            detail="Generation produced no problems")
    return problems, seed


def _label(grade: str) -> str:
    """Both namings, so a sheet handed to a teacher is unambiguous."""
    g = mathtutor.curriculum().get(grade)
    if not g:
        return grade
    ib = g.get("ib_label") or g.get("ib")
    return f"{g['label']} · {ib}" if ib else g["label"]


def _pdf_response(pdf: bytes | None, filename: str) -> Response:
    if not pdf:
        raise HTTPException(
            status_code=503,
            detail="PDF generation unavailable — pdflatex is not installed. "
                   "The on-screen preview still works.")
    return Response(
        content=pdf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def register_mathtutor_routes(app):
    """Attach ``/math/*`` endpoints to *app*."""

    @app.get("/math/curriculum")
    async def curriculum(request: Request):
        """Grades, their topics, and which of them are drawn."""
        await get_current_user(request)
        cur = mathtutor.curriculum()
        have = set(generators.available_topics())
        return {
            "grades": [
                {"id": gid, "label": g["label"],
                 "ib": g.get("ib", ""), "ib_label": g.get("ib_label", ""),
                 "topics": [{"id": t["id"], "name": t["name"],
                             "geometry": bool(t.get("geometry")),
                             "available": t["id"] in have}
                            for t in g["topics"]]}
                for gid, g in cur.items()
            ],
            "pdf_available": worksheet.latex_available(),
        }

    @app.post("/math/generate")
    async def generate_sheet(req: GenerateRequest, request: Request):
        """Problems plus an HTML preview. Nothing is stored."""
        await get_current_user(request)
        problems, seed = _build(req)
        return {
            "seed": seed,
            "grade": req.grade,
            "label": _label(req.grade),
            "count": len(problems),
            "problems": [p.as_dict() for p in problems],
            "html": worksheet.render_html(problems, req.title,
                                          _label(req.grade), answers=False),
            "answers_html": worksheet.render_html(problems, req.title,
                                                  _label(req.grade), answers=True),
        }

    @app.post("/math/pdf")
    async def sheet_pdf(req: GenerateRequest, request: Request):
        """The printable worksheet."""
        await get_current_user(request)
        problems, seed = _build(req)
        pdf = worksheet.render_pdf(problems, req.title, _label(req.grade),
                                   req.grade, answers=False,
                                   student=req.student)
        return _pdf_response(pdf, f"{req.grade}-worksheet-{seed}.pdf")

    @app.post("/math/answers")
    async def answers_pdf(req: GenerateRequest, request: Request):
        """The answer key for the same sheet. Same seed, same problems."""
        await get_current_user(request)
        problems, seed = _build(req)
        pdf = worksheet.render_pdf(problems, req.title, _label(req.grade),
                                   req.grade, answers=True)
        return _pdf_response(pdf, f"{req.grade}-answers-{seed}.pdf")

    # -- Saved worksheets ---------------------------------------------------

    @app.post("/math/worksheets")
    async def save(req: SaveRequest, request: Request):
        user = await get_current_user(request)
        problems, seed = _build(req)      # validate before storing
        rec = await save_worksheet(
            request.app.state.db, user["id"], title=req.title, grade=req.grade,
            topics=req.topics or [t["id"] for t in
                                  mathtutor.curriculum()[req.grade]["topics"]],
            count=req.count, seed=seed, student=req.student, notes=req.notes)
        rec["mix"] = req.mix
        return rec

    @app.get("/math/worksheets")
    async def listing(request: Request):
        user = await get_current_user(request)
        return {"worksheets": await list_worksheets(request.app.state.db,
                                                    user["id"])}

    @app.get("/math/worksheets/{wid}")
    async def one(wid: str, request: Request):
        user = await get_current_user(request)
        rec = await get_worksheet(request.app.state.db, user["id"], wid)
        if not rec:
            raise HTTPException(status_code=404, detail="Worksheet not found")
        req = GenerateRequest(grade=rec["grade"], topics=rec["topics"],
                              count=rec["count"], seed=rec["seed"],
                              title=rec["title"], student=rec["student"])
        problems, _ = _build(req)
        rec["problems"] = [p.as_dict() for p in problems]
        rec["label"] = _label(rec["grade"])
        rec["html"] = worksheet.render_html(problems, rec["title"],
                                            rec["label"], answers=False)
        rec["answers_html"] = worksheet.render_html(problems, rec["title"],
                                                    rec["label"], answers=True)
        return rec

    @app.get("/math/worksheets/{wid}/pdf")
    async def saved_pdf(wid: str, request: Request, answers: bool = False):
        """Reprint a saved sheet, or its key, from the stored recipe."""
        user = await get_current_user(request)
        rec = await get_worksheet(request.app.state.db, user["id"], wid)
        if not rec:
            raise HTTPException(status_code=404, detail="Worksheet not found")
        req = GenerateRequest(grade=rec["grade"], topics=rec["topics"],
                              count=rec["count"], seed=rec["seed"],
                              title=rec["title"], student=rec["student"])
        problems, _ = _build(req)
        pdf = worksheet.render_pdf(problems, rec["title"], _label(rec["grade"]),
                                   rec["grade"], answers=answers,
                                   student="" if answers else rec["student"])
        kind = "answers" if answers else "worksheet"
        return _pdf_response(pdf, f"{rec['title']}-{kind}.pdf".replace(" ", "-"))

    @app.delete("/math/worksheets/{wid}")
    async def remove(wid: str, request: Request):
        user = await get_current_user(request)
        if not await delete_worksheet(request.app.state.db, user["id"], wid):
            raise HTTPException(status_code=404, detail="Worksheet not found")
        return {"deleted": wid}

    # -- Marking ------------------------------------------------------------

    @app.post("/math/results")
    async def mark(req: MarkRequest, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        if not await get_worksheet(db, user["id"], req.worksheet_id):
            raise HTTPException(status_code=404, detail="Worksheet not found")
        n = await record_results(db, user["id"], req.worksheet_id,
                                 req.student, req.marks)
        return {"recorded": n}

    @app.get("/math/progress")
    async def progress(request: Request, student: str = ""):
        """Accuracy per topic, weakest first."""
        user = await get_current_user(request)
        rows = await topic_performance(request.app.state.db, user["id"], student)
        names = _topic_names()
        for r in rows:
            r["name"] = names.get(r["topic"], r["topic"])
        return {"topics": rows}
