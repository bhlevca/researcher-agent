"""Content-Disposition headers that survive non-ASCII filenames.

HTTP headers are latin-1. A filename containing an em dash, an accent or a
Romanian comma-below character cannot be put in one directly: Starlette encodes
headers with ``v.encode("latin-1")`` and raises, which turns a perfectly good
response into a 500 *after* the work is done.

That is how "French — Lesson.pdf" broke the lesson sheets while
"French-homework-21.pdf" worked: the em dash in the default title, nothing
more. The same latent fault sits behind every download whose name comes from a
document title or an uploaded file.

RFC 6266 is the fix: send an ASCII fallback in ``filename`` and the real name,
percent-encoded, in ``filename*``. Browsers that understand the latter use it;
the rest get a readable transliteration rather than an error.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import quote

__all__ = ["ascii_filename", "content_disposition"]


def ascii_filename(name: str, fallback: str = "download") -> str:
    """A latin-1-safe version of *name*, for the plain ``filename`` parameter.

    Accents are stripped rather than replaced with underscores, so "Leçon" reads
    as "Lecon" and not "Le_on". Characters with no ASCII equivalent — an em
    dash, a CJK character — are dropped.
    """
    # NFKD splits "ç" into "c" + a combining cedilla, which the ASCII encoding
    # then discards, leaving the base letter.
    folded = unicodedata.normalize("NFKD", name or "")
    out = folded.encode("ascii", "ignore").decode("ascii")
    # Quotes, backslashes and control characters would break the header syntax.
    out = re.sub(r'[\\"\x00-\x1f\x7f]', "", out)
    out = re.sub(r"\s+", " ", out).strip(" .")
    return out or fallback


def content_disposition(filename: str, *, inline: bool = False) -> dict[str, str]:
    """The header dict for a download called *filename*.

    Always use this rather than formatting the header by hand: the failure mode
    is a 500 on exactly the documents whose titles are worth having.
    """
    disposition = "inline" if inline else "attachment"
    fallback = ascii_filename(filename)
    return {
        "Content-Disposition": (
            f'{disposition}; filename="{fallback}"; '
            f"filename*=UTF-8''{quote(filename or fallback, safe='')}"
        )
    }
