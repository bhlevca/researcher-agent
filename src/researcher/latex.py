"""Shared LaTeX typesetting for printable sheets.

Both tutors print: the math tutor produces worksheets and answer keys, and the
language tutor produces grammar sheets, tests and exams. They want the same
things from LaTeX — escaping, a preamble, prose that wraps around inline
mathematics, and a compile step that fails softly — so that lives here once
rather than being copied and then drifting.

What is *not* here is anything about problems, exercises, grades or levels. Each
tutor builds its own document body; this module only knows how to turn a .tex
string into bytes.
"""

from __future__ import annotations

import os
import re
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

LATEX_CMD = os.getenv("LATEX_CMD", "pdflatex")

# article accepts only 10, 11 and 12pt and *silently discards* anything else,
# falling back to 10pt — so asking it for 14pt makes a sheet smaller, not
# bigger. extarticle (texlive-extsizes) accepts 8-20pt and is a drop-in.
_EXTARTICLE: bool | None = None


def latex_available() -> bool:
    """Whether printable output is possible on this machine."""
    return bool(shutil.which(LATEX_CMD))


def has_extarticle() -> bool:
    global _EXTARTICLE
    if _EXTARTICLE is None:
        try:
            _EXTARTICLE = subprocess.run(
                ["kpsewhich", "extarticle.cls"], capture_output=True,
                text=True, timeout=10).returncode == 0
        except Exception:
            _EXTARTICLE = False
    return _EXTARTICLE


BASE_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm,top=1.8cm,bottom=1.8cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\parindent}{0pt}
"""


def preamble(pt: int = 11) -> str:
    """The preamble at the requested type size, honouring it for real."""
    if pt <= 12:
        return BASE_PREAMBLE.replace("[11pt,", f"[{max(10, pt)}pt,")
    if has_extarticle():
        return BASE_PREAMBLE.replace("[11pt,a4paper]{article}",
                                     f"[{pt}pt,a4paper]{{extarticle}}")
    return BASE_PREAMBLE.replace("[11pt,", "[12pt,")


def escape_text(s: str) -> str:
    """Escape characters that would break LaTeX in a plain-text field."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


# LaTeX spacing commands, which are not mathematics and must not be left
# stranded inside dollar signs.
_SPACE = r"(?:\\[,;:!]|\\q?quad|\\ |~|\s)"


def mixed(latex: str, display: bool = False) -> str:
    r"""Turn a line written entirely in math mode into wrappable prose.

    Generators write a question as one math expression with the words inside
    \text{...}. That is correct LaTeX, but **math mode does not line-break**, so
    a long worded question runs off the right edge of the page instead of
    wrapping. Here the \text{...} parts become real text and only the
    mathematics stays between dollar signs, which is what lets a paragraph wrap.
    """
    out: list[str] = []
    i, n = 0, len(latex)
    buf: list[str] = []

    def flush_math():
        raw = "".join(buf)
        buf.clear()
        # A fragment that is only LaTeX spacing carries no mathematics; left
        # between dollar signs it compiles to "$\$".
        if re.fullmatch(_SPACE + "*", raw):
            if raw:
                out.append(" ")
            return
        chunk = re.sub(_SPACE + "+$", "", re.sub("^" + _SPACE + "+", "", raw))
        # Trailing punctuation belongs to the sentence, not to the formula.
        tail = ""
        while chunk and chunk[-1] in ".,;:?":
            tail = chunk[-1] + tail
            chunk = re.sub(_SPACE + "+$", "", chunk[:-1])
        if chunk:
            out.append(rf"$\displaystyle {chunk}$" if display else f"${chunk}$")
        if tail:
            # The space that followed the punctuation was consumed above; put
            # one back, or "(6, 5), radius 7" sets as "(6, 5),radius 7".
            out.append(tail + " ")

    def close_brace(start: int) -> int:
        j, depth = start, 1
        while j < n and depth:
            if latex[j] == "{":
                depth += 1
            elif latex[j] == "}":
                depth -= 1
            j += 1
        return j

    while i < n:
        if latex.startswith(r"\text{", i):
            j = close_brace(i + 6)
            # A unit carrying an exponent — \text{cm}^{2} — must stay in math
            # mode, or the exponent is orphaned from what it belongs to.
            if j < n and latex[j] in "^_":
                k = close_brace(j + 2) if latex[j + 1:j + 2] == "{" else j + 2
                buf.append(latex[i:k])
                i = k
                continue
            flush_math()
            out.append(latex[i + 6:j - 1])
            i = j
        else:
            buf.append(latex[i])
            i += 1
    flush_math()
    return " ".join("".join(out).split()) or latex


def compile_pdf(tex: str, extra_files: dict[str, bytes] | None = None,
                tag: str = "sheet") -> bytes | None:
    """Typeset *tex* to PDF bytes, or None if it cannot be done.

    Failure is soft on purpose: a caller falls back to an on-screen preview
    rather than returning an error page, because a missing LaTeX install should
    not take the whole feature down.
    """
    if not latex_available():
        logger.info("[%s] %s not installed; PDF unavailable", tag, LATEX_CMD)
        return None

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, data in (extra_files or {}).items():
            (work / name).write_bytes(data)
        (work / "sheet.tex").write_text(tex, encoding="utf-8")
        proc = None
        try:
            # Twice, so the header and any references settle.
            # -interaction=nonstopmode keeps one bad glyph from hanging the
            # request forever.
            for _ in range(2):
                proc = subprocess.run(
                    [LATEX_CMD, "-interaction=nonstopmode", "-halt-on-error",
                     "sheet.tex"],
                    cwd=work, capture_output=True, timeout=120,
                    # pdflatex echoes source lines in its overfull-box warnings
                    # using its own encoding, so an accented word in a long line
                    # emits a byte that is not valid UTF-8. With text=True and
                    # no error handler that raises here and throws away a PDF
                    # that built perfectly well.
                    text=True, encoding="utf-8", errors="replace")
            pdf = work / "sheet.pdf"
            if not pdf.is_file():
                tail = ((proc.stdout if proc else "") or "")[-700:]
                logger.warning("[%s] LaTeX produced no PDF: %s", tag, tail)
                return None
            return pdf.read_bytes()
        except subprocess.TimeoutExpired:
            logger.warning("[%s] LaTeX timed out", tag)
        except Exception as e:
            logger.warning("[%s] LaTeX failed: %s", tag, e)
    return None
