"""Printable worksheets and answer keys.

Education happens on paper here, so PDF is the real output and the browser view
is the preview. LaTeX does the typesetting because mathematics set in HTML looks
wrong to anyone who reads mathematics: fraction bars misalign, radicals do not
stretch, and integrals sit at the wrong height. pdflatex is already installed.

Two documents come out of one generation: the worksheet a child writes on, and
the answer key the adult marks from. They share the same problem objects, so the
key cannot drift from the sheet.
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

# Space left under each question for working. Primary needs less; a DP
# integration question needs half a page or the child writes in the margin.
_WORK_SPACE = {
    "grade1": "1.5cm", "grade2": "1.6cm", "grade3": "1.8cm",
    "grade4": "2.1cm", "grade5": "2.4cm", "grade6": "2.6cm",
    "grade7": "2.9cm", "grade8": "3.2cm", "grade9": "3.5cm",
    "grade10": "3.8cm", "grade11": "4.5cm", "grade12": "5.5cm",
}

# Younger children need larger type: 11pt is the adult default and is too small
# for a child still learning to decode numerals.
#
# article only accepts 10, 11 and 12pt — it silently DISCARDS anything else and
# falls back to 10pt, so asking it for 14pt makes the sheet smaller, not bigger.
# extarticle (texlive-extsizes) accepts 8-20pt and is otherwise a drop-in.
_FONT_PT = {1: 14, 2: 14, 3: 12, 4: 12, 5: 12}


def _has_extarticle() -> bool:
    global _EXTARTICLE
    if _EXTARTICLE is None:
        try:
            _EXTARTICLE = subprocess.run(
                ["kpsewhich", "extarticle.cls"], capture_output=True,
                text=True, timeout=10).returncode == 0
        except Exception:
            _EXTARTICLE = False
    return _EXTARTICLE


_EXTARTICLE: bool | None = None


def _preamble(level: str) -> str:
    import re as _re

    m = _re.search(r"(\d+)", level or "")
    pt = _FONT_PT.get(int(m.group(1)) if m else 99, 11)
    if pt <= 12:
        return _PREAMBLE.replace("[11pt,", f"[{pt}pt,")
    if _has_extarticle():
        return _PREAMBLE.replace("[11pt,a4paper]{article}",
                                 f"[{pt}pt,a4paper]{{extarticle}}")
    # Without extsizes, 12pt is the largest article will honour.
    return _PREAMBLE.replace("[11pt,", "[12pt,")


_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
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


def _escape_text(s: str) -> str:
    """Escape characters that would break LaTeX in plain-text fields."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


# LaTeX spacing commands, which are not mathematics and must not be left
# stranded inside dollar signs.
_SPACE = r"(?:\\[,;:!]|\\q?quad|\\ |~|\s)"


def _mixed(latex: str, display: bool = False) -> str:
    r"""Turn a question written entirely in math mode into wrappable prose.

    Generators write a question as one math expression with the words inside
    \text{...}. That is correct LaTeX, but **math mode does not line-break**, so
    a long worded question ran off the right edge of the page instead of
    wrapping. Here the \text{...} parts become real text and only the
    mathematics stays between dollar signs, which is what lets a paragraph wrap.
    """
    out: list[str] = []
    i, n = 0, len(latex)
    buf: list[str] = []

    def flush_math():
        raw = "".join(buf)
        buf.clear()
        # A fragment that is only LaTeX spacing (\, \; \quad \ ) carries no
        # mathematics; left between dollar signs it compiles to "$\$".
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
            # one back, or "(6, 5), radius 7" sets as "(6, 5),radius 7". A
            # trailing one is harmless — the join below trims it.
            out.append(tail + " ")

    def close_brace(start: int) -> int:
        """Index just past the brace group opening at *start*."""
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
    # Collapse the double spaces left where a \text{} abutted a formula.
    return " ".join("".join(out).split()) or latex


def _figure_files(problems, workdir: Path) -> dict[int, str]:
    """Render each figure to PDF beside the .tex so graphicx can include it."""
    from researcher.mathtutor.geometry import render_figure

    out: dict[int, str] = {}
    for i, p in enumerate(problems):
        if not p.figure:
            continue
        data = render_figure(p.figure, fmt="pdf")
        if not data:
            continue
        name = f"fig{i}.pdf"
        (workdir / name).write_bytes(data)
        out[i] = name
    return out


# The words printed on the sheet itself, as opposed to in the questions.
_SHEET_WORDS = {
    "Worksheet": {"fr": "Feuille d'exercices", "ro": "Fișă de lucru"},
    "Answer Key": {"fr": "Corrigé", "ro": "Barem de corectare"},
    "Name": {"fr": "Nom", "ro": "Numele"},
    "Date": {"fr": "Date", "ro": "Data"},
    "Answer": {"fr": "Réponse", "ro": "Răspuns"},
}


def _t(word: str, lang: str) -> str:
    return _SHEET_WORDS.get(word, {}).get(lang, word)


def _build_tex(problems, title: str, level_label: str, *,
               answers: bool, figures: dict[int, str],
               level: str, student: str = "", lang: str = "en") -> str:
    from researcher.mathtutor.i18n import codes

    space = _WORK_SPACE.get(level, "3cm")
    lang_label = codes(lang)[0]
    kind = _t("Answer Key" if answers else "Worksheet", lang_label)

    lines = [_preamble(level)]
    lines.append(rf"\fancyhead[L]{{\small {_escape_text(title)}}}")
    lines.append(rf"\fancyhead[R]{{\small {_escape_text(level_label)}}}")
    lines.append(r"\fancyfoot[C]{\small\thepage}")
    lines.append(r"\begin{document}")

    lines.append(rf"\begin{{center}}\Large\textbf{{{_escape_text(title)}}}\\[2pt]")
    lines.append(rf"\normalsize {_escape_text(level_label)} \quad\textbf{{{kind}}}\end{{center}}")
    if not answers:
        # A name line, because these come back in a pile. Pre-filled when the
        # sheet was made for a particular child.
        lines.append(r"\vspace{2mm}")
        who = (rf"\underline{{\hspace{{1mm}}{_escape_text(student)}"
               rf"\hspace{{6cm}}}}" if student
               else r"\underline{\hspace{6cm}}")
        lines.append(rf"{_t('Name', lang_label)}: {who} \hfill "
                     rf"{_t('Date', lang_label)}: "
                     rf"\underline{{\hspace{{3.5cm}}}}")
        lines.append(r"\vspace{4mm}\hrule\vspace{4mm}")
    else:
        lines.append(r"\vspace{3mm}")

    lines.append(r"\begin{enumerate}[leftmargin=*,itemsep=0pt]")
    for i, p in enumerate(problems):
        lines.append(r"\item")
        lines.append(_mixed(p.question.strip()))
        if getattr(p, "question_alt", ""):
            # The support line, set smaller and in italics so the child's eye
            # goes to the first language and only drops to the second if needed.
            lines.append(r"\\[1mm]")
            lines.append(r"{\small\itshape " + _mixed(p.question_alt.strip())
                         + "}")

        if i in figures:
            lines.append(r"\\[2mm]")
            lines.append(rf"\includegraphics[width=0.42\textwidth]{{{figures[i]}}}")

        if answers:
            lines.append(r"\\[1mm]")
            for st in p.steps:
                lines.append(rf"{{\small {_mixed(st, display=True)}}}\\[1mm]")
            lines.append(rf"\textbf{{{_t('Answer', lang_label)}:}} "
                         rf"{_mixed(p.answer, display=True)}")
            if getattr(p, "answer_alt", "") and p.answer_alt != p.answer:
                lines.append(r"\\[1mm]")
                lines.append(r"{\small\itshape "
                             + _mixed(p.answer_alt, display=True) + "}")
            lines.append(r"\vspace{3mm}")
        else:
            lines.append(rf"\\[{space}]")
    lines.append(r"\end{enumerate}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def latex_available() -> bool:
    """Whether printable output is possible on this machine."""
    return bool(shutil.which(LATEX_CMD))


def render_pdf(problems, title: str, level_label: str, level: str = "grade8",
               answers: bool = False, student: str = "",
               lang: str = "en") -> bytes | None:
    """Typeset a worksheet (or its answer key) to PDF.

    Returns None when LaTeX is unavailable or the document fails to compile —
    the caller falls back to the HTML preview rather than failing the request.
    """
    if not shutil.which(LATEX_CMD):
        logger.info("[math] %s not installed; PDF unavailable", LATEX_CMD)
        return None
    if not problems:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        figs = _figure_files(problems, work)
        tex = _build_tex(problems, title, level_label, answers=answers,
                         figures=figs, level=level, student=student, lang=lang)
        (work / "sheet.tex").write_text(tex, encoding="utf-8")
        try:
            # Twice, so \pageref and the header settle. -interaction=nonstopmode
            # keeps a single bad glyph from hanging the request forever.
            for _ in range(2):
                proc = subprocess.run(
                    [LATEX_CMD, "-interaction=nonstopmode", "-halt-on-error",
                     "sheet.tex"],
                    cwd=work, capture_output=True, text=True, timeout=120)
            pdf = work / "sheet.pdf"
            if not pdf.is_file():
                tail = (proc.stdout or "")[-700:]
                logger.warning("[math] LaTeX produced no PDF: %s", tail)
                return None
            return pdf.read_bytes()
        except subprocess.TimeoutExpired:
            logger.warning("[math] LaTeX timed out")
        except Exception as e:
            logger.warning("[math] LaTeX failed: %s", e)
    return None


def render_html(problems, title: str, level_label: str,
                answers: bool = False) -> str:
    """Browser preview. KaTeX renders the same LaTeX the PDF uses."""
    import html as _html
    from researcher.mathtutor.geometry import render_figure
    import base64

    parts = [f"<h2>{_html.escape(title)}</h2>",
             f"<p class='muted'>{_html.escape(level_label)} — "
             f"{'Answer Key' if answers else 'Worksheet'}</p>", "<ol>"]
    for p in problems:
        parts.append("<li>")
        parts.append(f"<div class='q'>\\({p.question}\\)</div>")
        if getattr(p, "question_alt", ""):
            parts.append(f"<div class='q alt muted'><em>\\({p.question_alt}"
                         f"\\)</em></div>")
        if p.figure:
            svg = render_figure(p.figure, fmt="svg")
            if svg:
                b64 = base64.b64encode(svg).decode()
                parts.append(
                    f"<img alt='figure' style='max-width:320px;display:block;"
                    f"margin:6px 0' src='data:image/svg+xml;base64,{b64}'>")
        if answers:
            for st in p.steps:
                parts.append(f"<div class='step muted'>\\({st}\\)</div>")
            parts.append(f"<div class='a'><b>Answer:</b> \\({p.answer}\\)</div>")
        parts.append("</li>")
    parts.append("</ol>")
    return "\n".join(parts)
