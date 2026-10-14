"""Markdown to LaTeX, for printing lesson sheets.

The tutor's lessons are written by the model as Markdown — headings, a
vocabulary table, example dialogues, practice exercises and a Solutions
section. They were only ever exportable as .md or .docx, which is a document
for a teacher to read rather than a sheet a child writes on.

This renders that Markdown as a printable sheet, and splits it at the Solutions
heading: everything before is the worksheet, the Solutions section is the answer
key. The split is what makes an answer key possible without asking the model
again — the answers are already in the text it wrote.

Only the subset the lesson template actually produces is supported: headings,
bold, italic, inline code, bullet and numbered lists, tables and rules. Anything
else degrades to plain text rather than breaking the compile, because a lesson
that prints slightly plainly is better than one that does not print.
"""

from __future__ import annotations

import re
import logging

from researcher.latex import escape_text

logger = logging.getLogger(__name__)

# Headings that mark the answers. The model is told to write "Solutions", but
# it is a model, so the common variants are accepted too.
_SOLUTION_WORDS = ("solution", "answer key", "answers", "corrigé", "corrige",
                   "lösungen", "losungen", "răspunsuri", "raspunsuri")

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_RULE = re.compile(r"^\s*([-*_])\s*\1\s*\1[\s\-*_]*$")


def split_solutions(md: str) -> tuple[str, str]:
    """Split a lesson into (worksheet, solutions).

    Returns an empty solutions string when the lesson has no such section, in
    which case the caller prints no answer key rather than inventing one.
    """
    lines = (md or "").splitlines()
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if not m:
            continue
        title = m.group(2).strip().lower().strip("*_ :")
        # "Solutions", "7. Solutions", "Solutions to the exercises" all count.
        cleaned = re.sub(r"^\d+[.)]\s*", "", title)
        if any(cleaned.startswith(w) for w in _SOLUTION_WORDS):
            return "\n".join(lines[:i]).rstrip(), "\n".join(lines[i:]).strip()
    return (md or "").strip(), ""


def _inline(text: str) -> str:
    r"""Escape a line, then restore the inline markup as LaTeX.

    Escaping first matters: the text contains # and % and _ that would
    otherwise be LaTeX commands. Asterisks and backticks survive escaping, so
    the markup can be converted afterwards.
    """
    out = escape_text(text)
    out = re.sub(r"\*\*\*(.+?)\*\*\*", r"\\textbf{\\textit{\1}}", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", out)
    out = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", out)
    out = re.sub(r"`(.+?)`", r"\\texttt{\1}", out)
    return out


def _table(rows: list[str]) -> list[str]:
    """Render a Markdown table as a tabular.

    Ragged rows are padded rather than rejected: a model that drops a cell
    should cost a blank box, not the whole sheet.
    """
    cells = []
    for row in rows:
        parts = [c.strip() for c in row.strip().strip("|").split("|")]
        cells.append(parts)
    width = max(len(r) for r in cells)
    cells = [r + [""] * (width - len(r)) for r in cells]

    # Width by content, not equal shares. A vocabulary table is
    # | French | English | Example sentence | — giving the example the same
    # room as the single word beside it wastes most of the page and wraps the
    # sentence to four lines.
    longest = [max((len(r[i]) for r in cells), default=1) or 1
               for i in range(width)]
    # A gentle curve: a column with four times the text gets rather more than
    # twice the width, not four times, so the short columns stay readable while
    # an example sentence still gets the room it needs.
    weights = [max(1.0, n) ** 0.68 for n in longest]
    total_w = sum(weights)
    # Floor each column so a one-character header still has somewhere to sit.
    shares = [max(0.08, 0.92 * w / total_w) for w in weights]
    scale = 0.92 / sum(shares)
    spec = "|" + "|".join(
        f"p{{{s_ * scale:.3f}\\textwidth}}" for s_ in shares) + "|"
    out = [r"\vspace{1mm}", r"{\small\begin{tabular}{" + spec + "}", r"\hline"]
    for i, row in enumerate(cells):
        body = " & ".join(_inline(c) for c in row)
        out.append(body + r" \\")
        if i == 0:
            out.append(r"\hline")
    out.append(r"\hline")          # close the box; without it the table hangs open
    out.append(r"\end{tabular}}")
    out.append(r"\vspace{2mm}")
    return [ln for ln in out if ln != ""]


def to_latex(md: str, *, blank_lines: bool = False) -> str:
    """Render Markdown as a LaTeX document body.

    *blank_lines* adds writing space after each numbered item, which is what
    turns a list of practice exercises into something a child can write on.
    """
    lines = (md or "").splitlines()
    out: list[str] = []
    i = 0
    list_env: str | None = None

    def close_list():
        nonlocal list_env
        if list_env:
            out.append(rf"\end{{{list_env}}}")
            list_env = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            close_list()
            out.append("")
            i += 1
            continue

        # Table: a header row followed by a |---|---| separator.
        if (stripped.startswith("|") and i + 1 < len(lines)
                and _TABLE_SEP.match(lines[i + 1])):
            close_list()
            rows = [lines[i]]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j])
                j += 1
            out += _table(rows)
            i = j
            continue

        if _RULE.match(stripped):
            close_list()
            out.append(r"\vspace{1mm}\hrule\vspace{2mm}")
            i += 1
            continue

        m = _HEADING.match(stripped)
        if m:
            close_list()
            depth = len(m.group(1))
            size = {1: r"\large\bfseries", 2: r"\large\bfseries",
                    3: r"\normalsize\bfseries"}.get(depth, r"\normalsize\bfseries")
            out.append(rf"\vspace{{3mm}}{{{size} {_inline(m.group(2).strip())}}}"
                       rf"\\[1.5mm]")
            i += 1
            continue

        m = _NUMBERED.match(line)
        if m:
            if list_env != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}[leftmargin=*,itemsep=1mm]")
                list_env = "enumerate"
            out.append(r"\item " + _inline(m.group(1)))
            if blank_lines:
                out.append(r"\\[8mm]")
            i += 1
            continue

        m = _BULLET.match(line)
        if m:
            if list_env != "itemize":
                close_list()
                out.append(r"\begin{itemize}[leftmargin=*,itemsep=0.5mm]")
                list_env = "itemize"
            out.append(r"\item " + _inline(m.group(1)))
            i += 1
            continue

        if list_env:
            # A wrapped continuation line inside a list item.
            out.append(_inline(stripped))
        else:
            out.append(_inline(stripped) + r"\\")
        i += 1

    close_list()
    return "\n".join(out)
