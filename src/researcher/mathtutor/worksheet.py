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

# Typesetting primitives are shared with the language tutor; see latex.py.
from researcher.latex import (          # noqa: E402
    LATEX_CMD,
    latex_available,
    compile_pdf,
    escape_text as _escape_text,
    mixed as _mixed,
    preamble as _base_preamble,
)

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
_FONT_PT = {1: 14, 2: 14, 3: 12, 4: 12, 5: 12}


def _preamble(level: str) -> str:
    m = re.search(r"(\d+)", level or "")
    return _base_preamble(_FONT_PT.get(int(m.group(1)) if m else 99, 11))


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
    if not problems:
        return None
    figures = {}
    files = {}
    from researcher.mathtutor.geometry import render_figure

    for i, p in enumerate(problems):
        if not p.figure:
            continue
        data = render_figure(p.figure, fmt="pdf")
        if data:
            files[f"fig{i}.pdf"] = data
            figures[i] = f"fig{i}.pdf"

    tex = _build_tex(problems, title, level_label, answers=answers,
                     figures=figures, level=level, student=student, lang=lang)
    return compile_pdf(tex, files, tag="math")


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
