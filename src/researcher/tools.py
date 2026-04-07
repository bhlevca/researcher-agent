"""Search and geometric drawing tools used by the CrewAI agent.

Extracted from crew.py during v1.1.0 refactoring.
"""

import json as _json
import uuid as _uuid
import logging
from pathlib import Path

from crewai.tools import tool
from crewai_tools import SerperDevTool
from langchain_community.tools import DuckDuckGoSearchRun
from PIL import Image, ImageDraw, ImageFont

_logger = logging.getLogger(__name__)

# --- Smart TrueType font discovery ---
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian / Ubuntu / openSUSE
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",  # Fedora / RHEL
    "/usr/share/fonts/truetype/DejaVuSans.ttf",  # openSUSE alt
    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # Generic Linux
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _find_truetype_font() -> str | None:
    for p in _FONT_SEARCH_PATHS:
        if Path(p).is_file():
            return p
    _logger.warning(
        "No TrueType font found in standard paths; "
        "text rendering will use Pillow default bitmap font"
    )
    return None


_TRUETYPE_FONT_PATH = _find_truetype_font()

# Instantiate once, reuse across calls
_ddg_search = DuckDuckGoSearchRun()
_serper_raw = SerperDevTool()

_GENERATED_DIR = Path(__file__).parent / "static" / "generated"
_GENERATED_DIR.mkdir(parents=True, exist_ok=True)


@tool("InternetSearch")
def serper_search_wrapped(search_query: str) -> str:
    """Search the internet using Google (Serper). Input must be a single search query string."""
    # LLMs sometimes send JSON or arrays instead of a plain string — handle gracefully
    q = search_query.strip()
    if q.startswith("[") or q.startswith("{"):
        try:
            parsed = _json.loads(q)
            if isinstance(parsed, list):
                results = []
                for item in parsed:
                    sq = (
                        item.get("search_query", "")
                        if isinstance(item, dict)
                        else str(item)
                    )
                    if sq:
                        results.append(_serper_raw.run(search_query=sq))
                return "\n---\n".join(results)
            elif isinstance(parsed, dict) and "search_query" in parsed:
                q = parsed["search_query"]
        except (_json.JSONDecodeError, Exception):
            pass
    return _serper_raw.run(search_query=q)


@tool("DuckDuckGoSearch")
def ddg_search_wrapped(query: str) -> str:
    """Search the web using DuckDuckGo. Use as fallback if Serper fails. Input must be a single search query string."""
    q = query.strip()
    if q.startswith("[") or q.startswith("{"):
        try:
            parsed = _json.loads(q)
            if isinstance(parsed, list):
                results = []
                for item in parsed:
                    sq = (
                        item.get("query", item.get("search_query", ""))
                        if isinstance(item, dict)
                        else str(item)
                    )
                    if sq:
                        results.append(_ddg_search.run(sq))
                return "\n---\n".join(results)
            elif isinstance(parsed, dict):
                q = parsed.get("query", parsed.get("search_query", q))
        except (_json.JSONDecodeError, Exception):
            pass
    return _ddg_search.run(q)


@tool("GenerateImage")
def generate_image(instructions: str) -> str:
    """Draw geometric shapes. Input: JSON string with width, height, background, shapes array.
    Shape types: rectangle, circle, triangle, polygon, line, text. Copy the returned image tag into Final Answer.
    """
    try:
        spec = _json.loads(instructions)
    except _json.JSONDecodeError:
        return "Error: instructions must be valid JSON"
    w = min(int(spec.get("width", 512)), 2048)
    h = min(int(spec.get("height", 512)), 2048)
    bg = spec.get("background", "white")
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for shape in spec.get("shapes", []):
        t = shape.get("type", "")
        fill = shape.get("fill", None)
        outline = shape.get("outline", None)
        if t == "rectangle":
            x, y = int(shape.get("x", 0)), int(shape.get("y", 0))
            sw, sh = int(shape.get("width", 100)), int(shape.get("height", 100))
            draw.rectangle([x, y, x + sw, y + sh], fill=fill, outline=outline)
        elif t == "circle":
            cx, cy = int(shape.get("cx", 100)), int(shape.get("cy", 100))
            r = int(shape.get("radius", 50))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline)
        elif t == "line":
            x1, y1 = int(shape.get("x1", 0)), int(shape.get("y1", 0))
            x2, y2 = int(shape.get("x2", 100)), int(shape.get("y2", 100))
            lw = int(shape.get("width", 2))
            draw.line([x1, y1, x2, y2], fill=fill or "black", width=lw)
        elif t in ("triangle", "polygon"):
            pts = shape.get("points", shape.get("vertices", []))
            if pts and len(pts) >= 3:
                flat = [tuple(p) for p in pts]
                draw.polygon(flat, fill=fill, outline=outline)
        elif t == "text":
            x, y = int(shape.get("x", 10)), int(shape.get("y", 10))
            txt = str(shape.get("text", ""))
            size = int(shape.get("size", 20))
            if _TRUETYPE_FONT_PATH:
                try:
                    font = ImageFont.truetype(_TRUETYPE_FONT_PATH, size)
                except (IOError, OSError):
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
            draw.text((x, y), txt, fill=fill or "black", font=font)
    fname = f"{_uuid.uuid4().hex[:12]}.png"
    img.save(_GENERATED_DIR / fname)
    return f"![generated image](/static/generated/{fname})"
