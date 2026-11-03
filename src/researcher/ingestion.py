"""File ingestion and export.

Upload files (TXT, CSV, PDF, DOCX, XLSX, JSON, MD), extract text content
for the research agent, and export model output to various formats.

Public API
----------
- ``register_ingestion_routes(app)`` — attach ``/files/*`` and ``/export``
- ``init_files_table(db)`` — create the ``files`` table if missing
- ``get_file_context(db, user_id, file_ids)`` — merged text for chat context
"""

import io
import os
import re
import json
import uuid
import base64
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

from researcher.gpu_broker import GPU, Need, queue_message

from fastapi import HTTPException, Request, UploadFile, File
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Per-file upload cap. 10 MB was arbitrary and too small for a scanned book,
# which runs to tens or hundreds of megabytes. Raised and made configurable.
#
# Size is not what makes a book hard, though: see _extract_pdf. A scanned PDF
# is OCR'd one page at a time through a vision model, holding the GPU, inside
# this HTTP request — so a long one is bounded by OCR_MAX_PAGES and by how
# long the browser will wait, not by megabytes.
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "200")) * 1024 * 1024
MAX_TEXT_CHARS = 50_000                  # truncate extracted text
MAX_FILES_PER_USER = 50
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.bmp', '.gif'}
ALLOWED_EXTENSIONS = {
    '.txt', '.csv', '.pdf', '.docx', '.xlsx', '.md', '.json'
} | IMAGE_EXTENSIONS

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"

# --- OCR for scanned PDFs -------------------------------------------------
# Ollama vision models accept IMAGES, not documents: there is no way to hand
# them a PDF. Pages with no text layer are therefore rendered to PNG, downscaled,
# and sent as base64. Both steps matter — qwen3-vl returns an empty string when
# the image is much over ~1024px on its long edge.
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") == "1"
OCR_MODEL = os.getenv("OCR_MODEL", "deepseek-ocr:3b")
OCR_DPI = int(os.getenv("OCR_DPI", "150"))
OCR_MAX_DIM = int(os.getenv("OCR_MAX_DIM", "1024"))
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "20"))
OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "180"))
# Format offered for automatic download when a upload had to be OCR'd:
# "docx", "odt", or "none" to disable and keep it click-to-download.
OCR_AUTO_EXPORT = os.getenv("OCR_AUTO_EXPORT", "docx").lower().strip()
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_ocr_params() -> dict:
    """Current runtime OCR settings, for the Params panel."""
    return {
        "ocr_enabled": OCR_ENABLED,
        "ocr_model": OCR_MODEL,
        "ocr_dpi": OCR_DPI,
        "ocr_max_dim": OCR_MAX_DIM,
        "ocr_max_pages": OCR_MAX_PAGES,
        "ocr_timeout": OCR_TIMEOUT,
        "ocr_auto_export": OCR_AUTO_EXPORT,
    }


def update_ocr_params(params: dict) -> dict:
    """Update OCR settings at runtime. Only known keys are accepted."""
    global OCR_ENABLED, OCR_MODEL, OCR_DPI, OCR_MAX_DIM
    global OCR_MAX_PAGES, OCR_TIMEOUT, OCR_AUTO_EXPORT

    if "ocr_enabled" in params:
        OCR_ENABLED = bool(params["ocr_enabled"])
    if "ocr_model" in params:
        val = str(params["ocr_model"]).strip()
        if val:
            OCR_MODEL = val
    if "ocr_dpi" in params:
        OCR_DPI = max(72, min(400, int(params["ocr_dpi"])))
    if "ocr_max_dim" in params:
        # Deliberately capped: qwen3-vl returns an EMPTY string well above
        # ~1500px, which looks like OCR silently failing.
        OCR_MAX_DIM = max(512, min(2048, int(params["ocr_max_dim"])))
    if "ocr_max_pages" in params:
        OCR_MAX_PAGES = max(1, min(200, int(params["ocr_max_pages"])))
    if "ocr_timeout" in params:
        OCR_TIMEOUT = max(30, min(900, int(params["ocr_timeout"])))
    if "ocr_auto_export" in params:
        val = str(params["ocr_auto_export"]).lower().strip()
        if val in ("docx", "odt", "none"):
            OCR_AUTO_EXPORT = val
    return get_ocr_params()


def _ocr_prompt(model: str) -> str:
    """Return the prompt this OCR model expects.

    deepseek-ocr is a specialist that wants its own sentinel format; given a
    plain instruction it returns an empty string. General VLMs want prose.
    """
    if "deepseek-ocr" in model:
        return "<image>\nFree OCR."
    return (
        "Transcribe all text in this image exactly as it appears. "
        "Output only the transcribed text, no commentary."
    )


def _render_page_png(page) -> bytes | None:
    """Render a PDF page to a downscaled PNG suitable for a vision encoder."""
    try:
        from PIL import Image

        pix = page.get_pixmap(dpi=OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img.thumbnail((OCR_MAX_DIM, OCR_MAX_DIM), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning("[OCR] Failed to render page: %s", e)
        return None


def _ocr_image(png: bytes, model: str | None = None) -> str:
    """OCR a PNG via an Ollama vision model. Returns '' on any failure."""
    import requests

    model = model or OCR_MODEL
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": _ocr_prompt(model),
                "images": [base64.b64encode(png).decode()],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 4096},
            },
            timeout=OCR_TIMEOUT,
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
    except Exception as e:
        logger.warning("[OCR] %s failed: %s", OCR_MODEL, e)
        return ""

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------


async def init_files_table(db):
    """Create the ``files`` table if it doesn't exist."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            filename        TEXT NOT NULL,
            extension       TEXT NOT NULL,
            size            INTEGER NOT NULL,
            extracted_text  TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL
        )
    """)
    await db.commit()


# ---------------------------------------------------------------------------
# Context helper  (called by /chat and /ask)
# ---------------------------------------------------------------------------


# A request to transcribe a document is a CONVERSION job, not a conversation.
# The text already exists in the database, so routing it through the agent only
# makes an LLM slowly retype what we can return instantly — and risks rewording.
_EXTRACTION_INTENT_RE = re.compile(
    r"\b("
    r"ocr"
    r"|transcribe|transcription"
    r"|extract(?:ing)?\s+(?:the\s+|all\s+)?text"
    r"|(?:give|get|show|send)\s+(?:me\s+)?(?:the\s+|its\s+)?(?:full\s+)?text"
    r"|read\s+(?:this|the)\s+(?:document|file|scan|image|pdf|page)"
    r"|convert\s+(?:this|it|the\s+\w+)?\s*(?:to|into)\s+(?:text|docx|word|odt|document)"
    r")\b",
    re.IGNORECASE,
)


def is_extraction_request(message: str) -> bool:
    """True when the user wants the document's text, not a discussion of it."""
    return bool(_EXTRACTION_INTENT_RE.search(message or ""))


async def get_file_texts(db, user_id: str, file_ids: list[str]) -> list[dict]:
    """Return ``[{id, filename, text}]`` for the user's files, in request order."""
    out: list[dict] = []
    for fid in file_ids:
        cursor = await db.execute(
            "SELECT id, filename, extracted_text FROM files WHERE id = ? AND user_id = ?",
            (fid, user_id),
        )
        row = await cursor.fetchone()
        if row:
            out.append({"id": row[0], "filename": row[1], "text": (row[2] or "").strip()})
    return out


# How much attached text to put in the prompt. A scanned book runs to half a
# million characters — 448855-The Human Zoo came to 493,647 — against a
# num_ctx of 32768, roughly 130k characters. Sending it whole does not fail
# loudly; the tail is dropped or the call errors deep inside litellm, and
# either way the model answers from a fragment it cannot see the edges of.
#
# So a long attachment is reduced to the pages that bear on the question, and
# the block says that is what happened.
ATTACH_BUDGET_CHARS = int(os.getenv("ATTACH_BUDGET_CHARS", "24000"))

def attach_budget(num_ctx: int | None = None) -> int:
    """Characters of attachment to allow, from the model's context window.

    About a third of the window, at ~4 characters a token: the rest is the
    system prompt, the tools, the conversation and the answer. Overriding
    ATTACH_BUDGET_CHARS pins it regardless.
    """
    if os.getenv("ATTACH_BUDGET_CHARS"):
        return ATTACH_BUDGET_CHARS
    if not num_ctx or num_ctx <= 0:
        return ATTACH_BUDGET_CHARS
    return max(4000, int(num_ctx * 4 / 3))


_PAGE_MARK = re.compile(r"^\[Page (\d+)(?: — [^\]]*)?\]$", re.M)

# Words too common to tell pages apart.
_STOP = frozenset("""a an and are as at be but by for from had has have how i
if in into is it its of on or that the their then there these this to was were
what when where which who why will with you your about would could should can
do does did not no nor so than too very just me my we us our他""".split())


def _split_pages(text: str) -> list[tuple[int, str]]:
    """A scanned document's text, back into (page number, page text).

    scan_pdf_pages and _extract_pdf both write "[Page N]" headers, which is
    what makes selection possible — and what lets the model cite a page.
    """
    marks = list(_PAGE_MARK.finditer(text))
    if not marks:
        return []
    out = []
    for i, m in enumerate(marks):
        body_end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((int(m.group(1)), text[m.start():body_end].strip()))
    return out


def _keywords(question: str) -> list[str]:
    words = [w for w in re.findall(r"\w{3,}", (question or "").lower())
             if w not in _STOP]
    seen, out = set(), []
    for w in words:                    # keep order, drop repeats
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def select_pages(text: str, question: str, budget: int) -> tuple[str, dict]:
    """Reduce a long document to the pages most likely to answer *question*.

    Scores each page by how many of the question's distinctive words it
    contains, weighted by how often. Returns the chosen pages in reading order
    with their page numbers intact, and a summary for the caller to disclose.

    Deliberately simple and inspectable: no embeddings, no index to keep in
    step with the file. It is chosen pages, not a summary, so nothing is
    paraphrased and every line the model sees is the author's.
    """
    pages = _split_pages(text)
    if not pages:
        # No page structure to select on; take the opening and say so.
        return text[:budget], {"mode": "head", "pages": 0, "total": 0,
                               "chars": min(len(text), budget)}

    terms = _keywords(question)
    if terms:
        scored = []
        for num, body in pages:
            low = body.lower()
            hits = sum(low.count(t) for t in terms)
            distinct = sum(1 for t in terms if t in low)
            # Distinct terms dominate: a page mentioning three of the asked
            # words beats one repeating a single word many times.
            scored.append((distinct * 1000 + hits, num, body))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if scored[0][0] == 0:
            terms = []                 # nothing matched: fall back to the head
    if not terms:
        chosen, used = [], 0
        for num, body in pages:
            if used + len(body) > budget:
                break
            chosen.append((num, body))
            used += len(body)
        mode = "head"
    else:
        chosen, used = [], 0
        for score, num, body in scored:
            if score == 0:
                break
            if used + len(body) > budget:
                continue               # skip a page too big, keep looking
            chosen.append((num, body))
            used += len(body)
        chosen.sort()
        mode = "selected"

    if not chosen:                     # one page bigger than the whole budget
        num, body = pages[0]
        return body[:budget], {"mode": "head", "pages": 1,
                               "total": len(pages), "chars": budget}
    return ("\n\n".join(b for _, b in chosen),
            {"mode": mode, "pages": len(chosen), "total": len(pages),
             "chars": used,
             "numbers": [n for n, _ in chosen][:40]})


async def get_file_context(db, user_id: str, file_ids: list[str],
                           question: str = "",
                           budget: int | None = None) -> str:
    """Extracted text for *file_ids*, as a context block for the model.

    *question* lets a long document be reduced to the pages that bear on it;
    without one the opening pages are used. *budget* is the character ceiling
    per file.
    """
    if not file_ids:
        return ""
    budget = ATTACH_BUDGET_CHARS if budget is None else budget
    parts: list[str] = []
    empty: list[str] = []
    notes: list[str] = []
    for fid in file_ids:
        cursor = await db.execute(
            "SELECT filename, extracted_text FROM files WHERE id = ? AND user_id = ?",
            (fid, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            continue
        filename, text = row[0], (row[1] or "").strip()
        if not text:
            # Never emit an empty block under a "the text is RIGHT HERE"
            # banner: the agent can then neither read the file nor say so, and
            # burns every ReAct iteration trying to reconcile the two.
            empty.append(filename)
            continue
        if len(text) <= budget:
            parts.append(f"--- {filename} ---\n{text}")
            continue
        chunk, info = select_pages(text, question, budget)
        if info["mode"] == "selected":
            where = ", ".join(str(n) for n in info.get("numbers", []))
            notes.append(
                f"{filename}: {info['pages']} of {info['total']} pages "
                f"included (pages {where}), chosen for this question out of "
                f"{len(text):,} characters.")
            head = (f"--- {filename} (pages {where} of {info['total']}, "
                    f"selected for this question) ---")
        else:
            notes.append(
                f"{filename}: the first {info['chars']:,} of {len(text):,} "
                f"characters included — ask about a topic and the relevant "
                f"pages will be selected instead.")
            head = (f"--- {filename} (the opening, {info['chars']:,} of "
                    f"{len(text):,} characters) ---")
        parts.append(f"{head}\n{chunk}")
        logger.info("[attach] %s reduced: %s", filename, notes[-1])

    if not parts and not empty:
        return ""

    block = ""
    if parts:
        whole = not notes
        block += (
            "=== ATTACHED FILE CONTENT (already extracted — read it directly) ===\n"
            + ("The following is the FULL TEXT extracted from the user's "
               "uploaded file(s).\n" if whole else
               "The following is extracted text from the user's uploaded "
               "file(s). It is\nTOO LONG to include whole, so only part of it "
               "is here:\n" + "".join(f"  - {n}\n" for n in notes) +
               "Answer from these pages and cite their page numbers. If the "
               "answer is not in\nthem, say which pages you were given and "
               "ask for a narrower question —\ndo NOT guess at the rest of "
               "the document.\n")
            + "You MUST use this text to answer the user's questions. Do NOT "
              "claim you\ncannot read attachments — the content is RIGHT HERE.\n\n"
            + "\n\n".join(parts)
            + "\n\n=== END OF ATTACHED FILE CONTENT ===\n\n"
        )
    if empty:
        block += (
            "=== UNREADABLE ATTACHMENT(S) ===\n"
            "No text could be extracted from the following file(s): "
            + ", ".join(empty)
            + ".\nThis usually means a scanned PDF whose OCR produced nothing. "
            "Do NOT retry, do NOT guess the contents, and do NOT call any tool "
            "to open the file — you have no access to it. State plainly and "
            "immediately that the file could not be read and ask the user to "
            "supply a text-based version.\n"
            "=== END OF UNREADABLE ATTACHMENT(S) ===\n\n"
        )
    return block


# ---------------------------------------------------------------------------
# Text extractors
# ---------------------------------------------------------------------------


def _decode_bytes(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return data.decode("utf-8", errors="replace")


def _extract_txt(data: bytes) -> str:
    return _decode_bytes(data)


def _extract_csv(data: bytes) -> str:
    return _decode_bytes(data)


def _extract_md(data: bytes) -> str:
    return _decode_bytes(data)


def _extract_json(data: bytes) -> str:
    text = _decode_bytes(data)
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return text


# ---------------------------------------------------------------------------
# Choosing the model to scan with.
#
# Any vision model in Ollama can read a page, but they are not equally good at
# it, and the differences were learned the hard way:
#
#   deepseek-ocr:3b   purpose-built for pages: dense text, fastest. Measured
#                     here ~5x faster and more complete than qwen3-vl:8b on
#                     scans, and it gets its own prompt format automatically.
#   qwen3-vl:*        "VL" is vision-language, not video. A document model in
#                     its own right and generally better at layout, tables and
#                     non-Latin scripts than a plain OCR model — but slower,
#                     and measured here to return an EMPTY string above
#                     ~1024 px, which looks like OCR failing in silence rather
#                     than an error. Hence MAX_IMAGE_EDGE.
#   everything else   a general chat model that happens to accept images.
#                     Usable, but expect prose about the page instead of the
#                     text on it.
#
# Ranked rather than hardcoded so the list follows what is actually installed.
# ---------------------------------------------------------------------------

_SCAN_RANK = (
    ("deepseek-ocr", 100, "purpose-built for pages — fastest on dense text"),
    ("qwen3-vl", 70, "document model — better on tables and non-Latin scripts, slower"),
    ("qwen3.6", 55, "general vision model"),
    ("qwen3.5", 50, "general vision model"),
    ("gemma4", 45, "general vision model"),
    ("gemma3", 40, "general vision model"),
    ("mistral-small", 35, "general vision model"),
)


def _scan_score(name: str) -> tuple[int, str]:
    low = (name or "").lower()
    for tag, score, why in _SCAN_RANK:
        if tag in low:
            # A smaller variant of the same family is quicker per page, and
            # page reading does not reward size the way reasoning does.
            if any(s in low for s in (":3b", ":8b", "e4b")):
                score += 5
            return score, why
    return 10, "vision model, untested for pages"


def scan_models() -> list[dict]:
    """Installed vision models, best for scanning first.

    Asks Ollama what it has and which of those accept images, so the list
    cannot drift from the machine. An unreachable Ollama returns an empty
    list, and the caller says so rather than guessing.
    """
    import requests

    base = os.getenv("OLLAMA_BASE", "http://localhost:11434")
    try:
        tags = requests.get(f"{base}/api/tags", timeout=10).json().get("models", [])
    except Exception as e:
        logger.warning("[scan] could not list Ollama models: %s", e)
        return []

    out = []
    for m in tags:
        name = m.get("name") or ""
        try:
            caps = requests.post(f"{base}/api/show", json={"model": name},
                                 timeout=10).json().get("capabilities") or []
        except Exception:
            continue
        if "vision" not in caps:
            continue
        score, why = _scan_score(name)
        out.append({
            "name": name,
            "size_gb": round((m.get("size") or 0) / 2**30, 1),
            "recommended": False,
            "why": why,
            "_score": score,
        })
    out.sort(key=lambda m: (-m["_score"], m["size_gb"]))
    for m in out:
        m.pop("_score")
    if out:
        out[0]["recommended"] = True
    return out


def best_scan_model() -> str:
    """The model to scan with unless the person picks another."""
    models = scan_models()
    if not models:
        return OCR_MODEL
    # A configured OCR_MODEL that is actually installed wins: it is a choice
    # someone made, and the Params panel shows it.
    installed = {m["name"] for m in models}
    if OCR_MODEL in installed:
        return OCR_MODEL
    return models[0]["name"]


def _extract_pdf(data: bytes) -> tuple[str, bool]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PDF support requires PyMuPDF: pip install pymupdf",
        )
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    ocr_used = 0
    skipped = 0
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")
            continue
        # No text layer — a scanned page. PyMuPDF returns "" here, which used to
        # produce an empty attachment block that the agent could neither read nor
        # admit to. OCR it instead.
        if not OCR_ENABLED or ocr_used >= OCR_MAX_PAGES:
            skipped += 1
            continue
        png = _render_page_png(page)
        if png is None:
            continue
        ocr_text = _ocr_image(png)
        if ocr_text:
            ocr_used += 1
            pages.append(f"[Page {i + 1} — OCR]\n{ocr_text}")
            logger.info("[OCR] Page %d recovered %d chars via %s",
                        i + 1, len(ocr_text), OCR_MODEL)
    total = doc.page_count
    doc.close()
    if ocr_used:
        logger.info("[OCR] %d scanned page(s) transcribed, %d skipped",
                    ocr_used, skipped)
    if skipped:
        # Said in the text itself, because this is what the agent will read.
        # Dropping these pages in silence made a 400-page scan look like a
        # 20-page document that simply ended.
        logger.warning("[OCR] %d of %d pages were NOT read (OCR_MAX_PAGES=%d)",
                       skipped, total, OCR_MAX_PAGES)
        pages.append(
            f"[!! {skipped} of {total} pages were not read. OCR stops after "
            f"{OCR_MAX_PAGES} scanned pages (OCR_MAX_PAGES, or the Params "
            f"panel). This document is incomplete — do not treat it as the "
            f"whole text.]")
    return "\n\n".join(pages), ocr_used > 0


def _text_layer_survey(data: bytes, sample: int = 12) -> dict:
    """How much of a PDF already carries text, from a sample of its pages.

    Reported before a scan starts so the person can see whether they are
    about to re-read something that is already readable. "_text.pdf" files
    are page images with an OCR layer bolted on: 255 of The Human Zoo's 260
    pages had one, which is why a scan of it finished in seconds.
    """
    try:
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        total = doc.page_count
        step = max(1, total // sample)
        looked = with_text = 0
        for i in range(0, total, step):
            looked += 1
            if doc[i].get_text().strip():
                with_text += 1
        doc.close()
        return {"pages": total, "sampled": looked,
                "with_text_layer": with_text}
    except Exception:
        return {}


def scan_pdf_pages(data: bytes, model: str, first: int, last: int,
                   on_page=None, force_ocr: bool = False) -> tuple[str, dict]:
    """Read a PDF page by page, reporting progress as it goes.

    Unlike _extract_pdf this has no page cap: the caller chose a range and
    accepts how long it takes. *on_page* is called with
    (page_number, total, kind, chars) after each page, where kind is "text"
    for a page that had a text layer and "ocr" for one that was read by the
    vision model.

    Returns the text and a summary, so the caller can say what actually
    happened rather than leaving the person to guess from the length.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    total = doc.page_count
    first = max(1, first)
    last = total if last <= 0 else min(last, total)
    pages, stats = [], {"total": total, "first": first, "last": last,
                        "text": 0, "ocr": 0, "empty": 0, "model": model,
                        "forced": bool(force_ocr)}

    for i in range(first - 1, last):
        page = doc[i]
        text = "" if force_ocr else page.get_text()
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")
            stats["text"] += 1
            kind, chars = "text", len(text)
        else:
            png = _render_page_png(page)
            ocr_text = _ocr_image(png, model=model) if png else ""
            if ocr_text:
                pages.append(f"[Page {i + 1} — scanned]\n{ocr_text}")
                stats["ocr"] += 1
                kind, chars = "ocr", len(ocr_text)
            else:
                # Said in the text rather than dropped. A silently missing page
                # is indistinguishable from a blank one.
                pages.append(f"[Page {i + 1} — could not be read]")
                stats["empty"] += 1
                kind, chars = "empty", 0
        if on_page:
            on_page(i + 1, total, kind, chars)
    doc.close()
    return "\n\n".join(pages), stats


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="DOCX support requires python-docx: pip install python-docx",
        )
    doc = Document(io.BytesIO(data))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        rows = []
        for row in table.rows:
            rows.append("\t".join(cell.text for cell in row.cells))
        if rows:
            parts.append("\n".join(rows))
    return "\n".join(parts)


def _extract_xlsx(data: bytes) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="XLSX support requires openpyxl: pip install openpyxl",
        )
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts = []
    for name in wb.sheetnames:
        ws = wb[name]
        lines = [f"[Sheet: {name}]"]
        for row in ws.iter_rows(values_only=True):
            lines.append("\t".join(str(c) if c is not None else "" for c in row))
        parts.append("\n".join(lines))
    wb.close()
    return "\n\n".join(parts)


def _extract_image(data: bytes) -> tuple[str, bool]:
    """OCR an uploaded image. An image never has a text layer, so this is the
    only way to get anything out of it."""
    if not OCR_ENABLED:
        return "", False
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.thumbnail((OCR_MAX_DIM, OCR_MAX_DIM), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
    except Exception as e:
        logger.warning("[OCR] Could not open uploaded image: %s", e)
        return "", False

    text = _ocr_image(buf.getvalue())
    if text:
        logger.info("[OCR] Image recovered %d chars via %s", len(text), OCR_MODEL)
    return text, bool(text)


_EXTRACTORS = {
    ".txt": _extract_txt,
    ".csv": _extract_csv,
    ".md": _extract_md,
    ".json": _extract_json,
    ".docx": _extract_docx,
    ".xlsx": _extract_xlsx,
}


def _extract_text(data: bytes, ext: str) -> tuple[str, bool]:
    """Extract text from an upload.

    Returns ``(text, ocr_used)``. ``ocr_used`` is True when the content came
    from a vision model rather than an embedded text layer — that is the signal
    the caller uses to offer an automatic document download, since a scan is
    exactly the case where the user has no editable copy of their own.
    """
    if ext in IMAGE_EXTENSIONS:
        text, ocr_used = _extract_image(data)
    elif ext == ".pdf":
        text, ocr_used = _extract_pdf(data)
    else:
        func = _EXTRACTORS.get(ext)
        if not func:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        text, ocr_used = func(data), False

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"\n\n[… truncated at {MAX_TEXT_CHARS:,} characters]"
    return text, ocr_used


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _add_formatted_runs(paragraph, text):
    """Parse inline markdown (bold, italic, code) and add runs to *paragraph*."""
    from docx.shared import Pt

    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*") and part.endswith("*"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Courier New"
            run.font.size = Pt(9)
        else:
            paragraph.add_run(part)


def _export_docx(content: str, images_dir: Path | None = None) -> bytes:
    """Convert markdown-ish *content* to DOCX bytes."""
    try:
        from docx import Document
        from docx.shared import Pt, Inches
    except ImportError:
        raise HTTPException(status_code=500, detail="DOCX export requires python-docx")

    doc = Document()
    lines = content.split("\n")
    i = 0
    in_code = False
    in_table = False
    table_rows: list[list[str]] = []

    while i < len(lines):
        line = lines[i]

        # ── Code fence ──
        if line.strip().startswith("```"):
            if in_code:
                in_code = False
                i += 1
                continue
            in_code = True
            i += 1
            continue

        if in_code:
            p = doc.add_paragraph()
            run = p.add_run(line)
            run.font.name = "Courier New"
            run.font.size = Pt(9)
            pf = p.paragraph_format
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            i += 1
            continue

        # ── Table rows ──
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c.strip()) <= set("-: ") for c in cells):
                table_rows.append(cells)
            if not in_table:
                in_table = True
            i += 1
            continue
        elif in_table:
            in_table = False
            if table_rows:
                ncols = max(len(r) for r in table_rows)
                tbl = doc.add_table(rows=len(table_rows), cols=ncols)
                tbl.style = "Table Grid"
                for ri, rd in enumerate(table_rows):
                    for ci, ct in enumerate(rd):
                        if ci < ncols:
                            tbl.rows[ri].cells[ci].text = ct
                if table_rows:
                    for cell in tbl.rows[0].cells:
                        for para in cell.paragraphs:
                            for run in para.runs:
                                run.bold = True
            table_rows = []
            continue  # reprocess current line

        # ── Images ──
        img_match = re.match(
            r"!\[([^\]]*)\]\((/static/generated/[^)]+)\)", line.strip()
        )
        if img_match and images_dir:
            img_name = Path(img_match.group(2)).name
            img_full = images_dir / img_name
            if img_full.exists():
                doc.add_picture(str(img_full), width=Inches(5))
            else:
                doc.add_paragraph(f"[Image: {img_match.group(1)}]")
            i += 1
            continue

        # ── Headings ──
        if line.startswith("#### "):
            doc.add_heading(line[5:], level=4)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        # ── Bullet list ──
        elif re.match(r"^\s*[-*]\s", line):
            text = re.sub(r"^\s*[-*]\s+", "", line)
            p = doc.add_paragraph(style="List Bullet")
            _add_formatted_runs(p, text)
        # ── Numbered list ──
        elif re.match(r"^\s*\d+\.\s", line):
            text = re.sub(r"^\s*\d+\.\s+", "", line)
            p = doc.add_paragraph(style="List Number")
            _add_formatted_runs(p, text)
        # ── Blank line ──
        elif not line.strip():
            pass
        # ── Math block $$…$$ ──
        elif line.strip().startswith("$$"):
            eq_lines = [line.strip().lstrip("$")]
            i += 1
            while i < len(lines) and "$$" not in lines[i]:
                eq_lines.append(lines[i])
                i += 1
            if i < len(lines):
                eq_lines.append(lines[i].strip().rstrip("$"))
            eq_text = "\n".join(l for l in eq_lines if l.strip())
            p = doc.add_paragraph()
            p.alignment = 1  # center
            run = p.add_run(eq_text)
            run.italic = True
            run.font.size = Pt(11)
        # ── Regular paragraph ──
        else:
            p = doc.add_paragraph()
            _add_formatted_runs(p, line)

        i += 1

    # Flush remaining table
    if table_rows:
        ncols = max(len(r) for r in table_rows)
        tbl = doc.add_table(rows=len(table_rows), cols=ncols)
        tbl.style = "Table Grid"
        for ri, rd in enumerate(table_rows):
            for ci, ct in enumerate(rd):
                if ci < ncols:
                    tbl.rows[ri].cells[ci].text = ct

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _export_xlsx(content: str) -> bytes:
    """Extract markdown tables from *content* and write them to XLSX sheets."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        raise HTTPException(status_code=500, detail="XLSX export requires openpyxl")

    wb = Workbook()
    wb.remove(wb.active)

    table_pat = re.compile(r"((?:^\|.*\|$\n?)+)", re.MULTILINE)
    tables = table_pat.findall(content)

    if not tables:
        ws = wb.create_sheet("Content")
        for row_idx, line in enumerate(content.split("\n"), 1):
            ws.cell(row=row_idx, column=1, value=line)
    else:
        for tidx, tbl_text in enumerate(tables):
            ws = wb.create_sheet(f"Table {tidx + 1}")
            drow = 0
            for row_text in tbl_text.strip().split("\n"):
                cells = [c.strip() for c in row_text.strip().strip("|").split("|")]
                if all(set(c.strip()) <= set("-: ") for c in cells):
                    continue
                drow += 1
                for ci, ct in enumerate(cells, 1):
                    cell = ws.cell(row=drow, column=ci, value=ct)
                    if drow == 1:
                        cell.font = Font(bold=True)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


def _export_odt(content: str) -> bytes:
    """Render markdown-ish text to an OpenDocument Text file.

    Deliberately conservative: headings, bullets and paragraphs. OCR output is a
    transcription, so fidelity to the words matters more than rich styling.
    """
    from odf.opendocument import OpenDocumentText
    from odf.style import Style, TextProperties, ParagraphProperties
    from odf.text import H, P, List, ListItem

    doc = OpenDocumentText()

    mono = Style(name="Mono", family="paragraph")
    mono.addElement(TextProperties(fontfamily="monospace", fontsize="10pt"))
    mono.addElement(ParagraphProperties(margintop="0.05in", marginbottom="0.05in"))
    doc.styles.addElement(mono)

    in_code = False
    for raw in content.split("\n"):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            doc.text.addElement(P(stylename=mono, text=raw))
            continue
        if not line.strip():
            doc.text.addElement(P(text=""))
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            doc.text.addElement(H(outlinelevel=len(heading.group(1)),
                                  text=_strip_inline_markdown(heading.group(2))))
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if bullet:
            lst = List()
            item = ListItem()
            item.addElement(P(text=_strip_inline_markdown(bullet.group(1))))
            lst.addElement(item)
            doc.text.addElement(lst)
            continue
        doc.text.addElement(P(text=_strip_inline_markdown(line)))

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _strip_inline_markdown(text: str) -> str:
    """Remove inline markdown emphasis markers, keeping the words."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


# Extension + MIME for each supported export format.
_EXPORT_TYPES = {
    "md":   (".md",   "text/markdown"),
    "txt":  (".txt",  "text/plain"),
    "docx": (".docx", "application/vnd.openxmlformats-officedocument."
                      "wordprocessingml.document"),
    "odt":  (".odt",  "application/vnd.oasis.opendocument.text"),
    "xlsx": (".xlsx", "application/vnd.openxmlformats-officedocument."
                      "spreadsheetml.sheet"),
}


class ExportRequest(BaseModel):
    content: str
    format: str            # "md", "txt", "docx", "odt", "xlsx"
    filename: str = "export"


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_ingestion_routes(app):
    """Attach ``/files/*`` upload/list/delete and ``/export`` to *app*."""
    from researcher.auth import get_current_user

    # ── Upload ──

    @app.post("/files/upload")
    async def upload_files(
        request: Request,
        files: list[UploadFile] = File(...),
    ):
        user = await get_current_user(request)
        db = request.app.state.db

        cursor = await db.execute(
            "SELECT COUNT(*) FROM files WHERE user_id = ?", (user["id"],)
        )
        count = (await cursor.fetchone())[0]
        if count + len(files) > MAX_FILES_PER_USER:
            raise HTTPException(
                status_code=400,
                detail=f"File limit exceeded (max {MAX_FILES_PER_USER})",
            )

        results = []
        for f in files:
            name = f.filename or "unnamed"
            ext = Path(name).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                results.append({"filename": name, "error": f"Unsupported type: {ext}"})
                continue

            data = await f.read()
            if len(data) > MAX_FILE_SIZE:
                results.append({
                    "filename": name,
                    "error": f"File too large: {len(data) / 2**20:.0f} MB, "
                             f"limit {MAX_FILE_SIZE // 2**20} MB "
                             f"(raise MAX_FILE_SIZE_MB)",
                })
                continue

            try:
                # OCR can take seconds per page, so keep it off the event
                # loop — and take a turn in the GPU queue, because it runs a
                # vision model on the same card as everything else.
                async with GPU.exclusive(Need("ollama", OCR_MODEL), "OCR",
                                         user=user["id"]):
                    extracted, ocr_used = await asyncio.get_running_loop(
                    ).run_in_executor(None, _extract_text, data, ext)
            except HTTPException as e:
                results.append({"filename": name, "error": e.detail})
                continue

            fid = uuid.uuid4().hex[:10]
            now = datetime.now(timezone.utc).isoformat()

            # Persist original on disk
            user_dir = UPLOAD_DIR / user["id"]
            user_dir.mkdir(parents=True, exist_ok=True)
            (user_dir / f"{fid}_{name}").write_bytes(data)

            await db.execute(
                "INSERT INTO files "
                "(id, user_id, filename, extension, size, extracted_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fid, user["id"], name, ext, len(data), extracted, now),
            )
            await db.commit()

            # A scan or photo is exactly the case where the user has no editable
            # copy, so hand back a ready-to-download document rather than making
            # them ask the agent to retype what we already transcribed.
            auto_export = (
                f"/files/{fid}/export?format={OCR_AUTO_EXPORT}"
                if ocr_used and extracted and OCR_AUTO_EXPORT in ("docx", "odt")
                else None
            )
            results.append({
                "id": fid,
                "filename": name,
                "size": len(data),
                "chars": len(extracted),
                "ocr": ocr_used,
                "export_url": auto_export,
                "export_format": OCR_AUTO_EXPORT if auto_export else None,
            })

        return {"files": results}

    # ── List ──

    @app.get("/files")
    async def list_files(request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT id, filename, extension, size, created_at FROM files "
            "WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        )
        rows = await cursor.fetchall()
        return {
            "files": [
                {
                    "id": r[0], "filename": r[1], "extension": r[2],
                    "size": r[3], "created_at": r[4],
                }
                for r in rows
            ]
        }

    # ── Get one (includes extracted text) ──

    @app.get("/files/{file_id}")
    async def get_file(file_id: str, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT id, filename, extension, size, extracted_text, created_at "
            "FROM files WHERE id = ? AND user_id = ?",
            (file_id, user["id"]),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="File not found")
        return {
            "id": row[0], "filename": row[1], "extension": row[2],
            "size": row[3], "text": row[4], "created_at": row[5],
        }

    # ── Delete ──

    @app.delete("/files/{file_id}")
    async def delete_file(file_id: str, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT id FROM files WHERE id = ? AND user_id = ?",
            (file_id, user["id"]),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="File not found")

        # Remove from disk
        user_dir = UPLOAD_DIR / user["id"]
        for p in user_dir.glob(f"{file_id}_*"):
            p.unlink(missing_ok=True)

        await db.execute("DELETE FROM files WHERE id = ?", (file_id,))
        await db.commit()
        return {"deleted": file_id}

    # ── Export ──

    @app.post("/export")
    async def export_content(req: ExportRequest, request: Request):
        await get_current_user(request)
        fmt = req.format.lower().strip()
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", req.filename)[:50] or "export"

        if fmt == "md":
            return Response(
                content=req.content.encode(),
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}.md"'},
            )

        if fmt == "txt":
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", req.content)
            text = re.sub(r"\*([^*]+)\*", r"\1", text)
            text = re.sub(r"`([^`]+)`", r"\1", text)
            text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
            return Response(
                content=text.encode(),
                media_type="text/plain",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}.txt"'},
            )

        if fmt == "docx":
            images_dir = Path(__file__).parent / "static" / "generated"
            data = _export_docx(req.content, images_dir)
            return Response(
                content=data,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}.docx"'},
            )

        if fmt == "odt":
            data = _export_odt(req.content)
            return Response(
                content=data,
                media_type="application/vnd.oasis.opendocument.text",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}.odt"'},
            )

        if fmt == "xlsx":
            data = _export_xlsx(req.content)
            return Response(
                content=data,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}.xlsx"'},
            )

        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    @app.get("/files/{file_id}/export")
    async def export_file_text(file_id: str, request: Request, format: str = "docx"):
        """Download a file's extracted text (including OCR output) as a document.

        This is the *verbatim* transcription, straight from the files table — it
        never goes near the LLM, so nothing is summarised or reworded. That is
        the difference between this and exporting an agent reply.
        """
        user = await get_current_user(request)
        fmt = (format or "docx").lower().strip()
        if fmt not in _EXPORT_TYPES or fmt == "xlsx":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {fmt}. Use md, txt, docx or odt.",
            )

        db = request.app.state.db
        cursor = await db.execute(
            "SELECT filename, extracted_text FROM files WHERE id = ? AND user_id = ?",
            (file_id, user["id"]),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="File not found")

        filename, text = row[0], (row[1] or "").strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail="No text was extracted from this file, so there is nothing "
                       "to export. For a scanned PDF, check that OCR is enabled "
                       "and the OCR model is pulled.",
            )

        stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(filename).stem)[:50] or "extract"
        ext, media_type = _EXPORT_TYPES[fmt]

        if fmt == "md":
            data = f"# {filename}\n\n{text}\n".encode()
        elif fmt == "txt":
            data = text.encode()
        elif fmt == "docx":
            data = _export_docx(f"# {filename}\n\n{text}")
        else:  # odt
            data = _export_odt(f"# {filename}\n\n{text}")

        return Response(
            content=data,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{stem}{ext}"'},
        )

    # ------------------------------------------------------------------
    # Scanning a document
    #
    # One button, one long job. A scan reads a page at a time through a vision
    # model and holds the GPU for the whole run, which is accepted rather than
    # worked around — so the job says what it is doing, both to the person who
    # started it and to everyone else through the GPU panel.
    # ------------------------------------------------------------------

    @app.get("/files/scan/models")
    async def scan_model_choices(request: Request):
        """Installed vision models, best for reading pages first."""
        await get_current_user(request)
        models = scan_models()
        return {
            "models": models,
            "recommended": models[0]["name"] if models else "",
            "configured": OCR_MODEL,
            "hint": "" if models else
                    "No vision model found in Ollama. Pull one, e.g. "
                    "`ollama pull deepseek-ocr:3b`.",
        }

    @app.post("/files/scan")
    async def scan_document(request: Request, file: UploadFile = File(...),
                            model: str = "", first_page: int = 1,
                            last_page: int = 0, force_ocr: bool = False):
        """Scan a document to text — SSE streaming, one event per page."""
        user = await get_current_user(request)
        uid = user["id"]
        name = file.filename or "unnamed"
        ext = Path(name).suffix.lower()
        if ext != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Scanning reads PDFs. For a photo of a page, attach it "
                       "with 📎 instead.")

        data = await file.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(data) / 2**20:.0f} MB, limit "
                       f"{MAX_FILE_SIZE // 2**20} MB (raise MAX_FILE_SIZE_MB)")

        chosen = model.strip() or best_scan_model()
        installed = {m["name"] for m in scan_models()}
        if installed and chosen not in installed:
            raise HTTPException(status_code=400,
                                detail=f"{chosen} is not an installed vision model")

        def _sse(event: str, payload) -> str:
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

        async def event_stream():
            progress: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            # The whole scan runs inside one reservation: it is the GPU's
            # occupant until it finishes, and says so to everyone.
            async with GPU.reserve(Need("ollama", chosen), "scan",
                                   user=uid) as res:
                async for notice in res.wait():
                    yield _sse("queue", queue_message(
                        notice["position"], notice["estimate"], notice["ticket"]))

                res.say(f"scanning {name}")
                yield _sse("start", {
                    "file": name, "model": chosen, "force_ocr": force_ocr,
                    # A page-image PDF with an OCR layer added by someone else
                    # looks identical to a plain scan until you look.
                    **_text_layer_survey(data),
                })

                def on_page(page, total, kind, chars):
                    loop.call_soon_threadsafe(
                        progress.put_nowait,
                        {"page": page, "total": total, "kind": kind, "chars": chars})

                task = loop.run_in_executor(
                    None, scan_pdf_pages, data, chosen, first_page, last_page,
                    on_page, force_ocr)

                done = 0
                while True:
                    drain = asyncio.ensure_future(progress.get())
                    finished, _ = await asyncio.wait(
                        {drain, task}, return_when=asyncio.FIRST_COMPLETED)
                    if drain in finished:
                        ev = drain.result()
                        done += 1
                        res.say(f"scanning {name}: page {ev['page']} of "
                                f"{ev['total']}")
                        yield _sse("page", ev)
                        continue
                    drain.cancel()
                    break

                while not progress.empty():          # anything still queued
                    ev = progress.get_nowait()
                    yield _sse("page", ev)

                try:
                    text, stats = task.result()
                except Exception as e:
                    logger.exception("[scan] failed")
                    yield _sse("error", str(e))
                    return

            # Stored outside the reservation: the GPU is free the moment the
            # last page is read.
            fid = uuid.uuid4().hex[:10]
            now = datetime.now(timezone.utc).isoformat()
            user_dir = UPLOAD_DIR / uid
            user_dir.mkdir(parents=True, exist_ok=True)
            (user_dir / f"{fid}_{name}").write_bytes(data)
            db = request.app.state.db
            await db.execute(
                "INSERT INTO files "
                "(id, user_id, filename, extension, size, extracted_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fid, uid, name, ext, len(data), text, now))
            await db.commit()
            logger.info("[scan] %s: %d page(s) read (%d text, %d scanned, "
                        "%d unreadable) with %s", name,
                        stats["last"] - stats["first"] + 1, stats["text"],
                        stats["ocr"], stats["empty"], chosen)

            yield _sse("done", {
                "file_id": fid, "filename": name, "chars": len(text),
                "stats": stats,
                # A book is far more text than the model's context window, so
                # say so rather than letting it be attached and silently cut.
                "too_long_for_context": len(text) > 60000,
            })

        return StreamingResponse(event_stream(),
                                 media_type="text/event-stream")
