"""Printable sheets for the language tutor: homework, tests, exams, challenges.

Shares its typesetting with the math tutor (``researcher.latex``) so the two
produce recognisably the same paper, and a fix to wrapping or escaping lands in
both at once.

Three things this adds that the math sheets do not have:

- **Sections**, because a test has parts and a worksheet does not.
- **Marks per question and a total**, which is what makes a test a test. The
  math tutor will borrow this later.
- **A passage block**, for a story or reading text followed by comprehension
  questions.

Where an answer came from a language pack it is printed as the answer. Where it
came from a model — comprehension, summaries — it is printed as a *model
answer*, and the sheet says so. Blurring that line is how a key ends up
confidently wrong, which is worse than having no key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from researcher.latex import (
    compile_pdf,
    escape_text,
    latex_available,
    mixed,
    preamble,
)

logger = logging.getLogger(__name__)

# How much room to leave for a written answer, by exercise skill. A conjugation
# needs a word; a summary needs a third of a page.
_SPACE = {
    "conjugation": "0pt", "gender_article": "0pt", "plural": "0pt",
    "adjective_agreement": "0pt", "vocabulary": "0pt",
    "german_case": "0pt", "tense_transform": "6mm",
    "comprehension": "2.2cm", "summary": "6cm", "writing": "7cm",
}

# Sheet furniture in the student's own language where it is theirs, otherwise
# English. The questions are in the target language; the instructions are not,
# because a child who cannot yet read the instruction cannot start.
_WORDS = {
    "Worksheet": "Worksheet", "Homework": "Homework", "Test": "Test",
    "Exam": "Examination", "Challenge": "Challenge",
    "Answer Key": "Answer Key", "Name": "Name", "Date": "Date",
    "Marks": "Marks", "Total": "Total", "Model answer": "Model answer",
}

# Marks a paragraph as the vocabulary glossary rather than story prose.
_GLOSSARY_MARK = "\x01gloss"

_KIND_TITLE = {"homework": "Homework", "test": "Test", "exam": "Exam",
               "challenge": "Challenge", "worksheet": "Worksheet"}


@dataclass
class Section:
    """One part of a sheet."""

    title: str = ""
    instruction: str = ""
    exercises: list = field(default_factory=list)   # list[Exercise]
    marks_each: int = 1
    # Prose shown before the questions — a story or reading text.
    passage: str = ""
    # True when the answers below were written by a model rather than derived
    # from a language pack.
    model_answers: bool = False

    @property
    def marks(self) -> int:
        return self.marks_each * len(self.exercises)


def _item(ex, *, answers: bool, marks: int | None, model: bool) -> list[str]:
    out = [r"\item"]
    if marks is not None:
        # The mark allocation sits in the right margin, as on a real paper.
        out.append(rf"\hfill\makebox[0pt][r]{{\small[{marks}]}}\\[-\baselineskip]")
    out.append(mixed(ex.prompt))
    if ex.options:
        shown = r" \quad ".join(escape_text(o) for o in ex.options)
        out.append(rf"\\[1mm]{{\small ({shown})}}")
    if ex.hint and not answers:
        out.append(rf"\\[0.5mm]{{\small\itshape {escape_text(ex.hint)}}}")
    if answers:
        label = _WORDS["Model answer"] if model else "Answer"
        out.append(rf"\\[1mm]\textbf{{{label}:}} "
                   rf"{{\bfseries {escape_text(ex.answer)}}}")
        out.append(r"\vspace{2mm}")
    else:
        out.append(rf"\\[{_SPACE.get(ex.skill, '8mm')}]")
    return out


def _build_tex(sections: list[Section], *, title: str, subtitle: str,
               kind: str, answers: bool, student: str,
               show_marks: bool) -> str:
    lines = [preamble(11)]
    lines.append(rf"\fancyhead[L]{{\small {escape_text(title)}}}")
    lines.append(rf"\fancyhead[R]{{\small {escape_text(subtitle)}}}")
    lines.append(r"\fancyfoot[C]{\small\thepage}")
    lines.append(r"\begin{document}")

    label = _WORDS["Answer Key"] if answers else _WORDS.get(
        _KIND_TITLE.get(kind, "Worksheet"), "Worksheet")
    lines.append(rf"\begin{{center}}\Large\textbf{{{escape_text(title)}}}\\[2pt]")
    lines.append(rf"\normalsize {escape_text(subtitle)} \quad"
                 rf"\textbf{{{label}}}\end{{center}}")

    total = sum(s.marks for s in sections)
    if not answers:
        lines.append(r"\vspace{2mm}")
        who = (rf"\underline{{\hspace{{1mm}}{escape_text(student)}"
               rf"\hspace{{6cm}}}}" if student
               else r"\underline{\hspace{6cm}}")
        right = (rf"{_WORDS['Total']}: \underline{{\hspace{{1cm}}}} / {total}"
                 if show_marks
                 else rf"{_WORDS['Date']}: \underline{{\hspace{{3.5cm}}}}")
        lines.append(rf"{_WORDS['Name']}: {who} \hfill {right}")
        lines.append(r"\vspace{3mm}\hrule\vspace{4mm}")
    else:
        lines.append(r"\vspace{3mm}")

    for n, sec in enumerate(sections, 1):
        if sec.title or len(sections) > 1:
            head = sec.title or f"Part {n}"
            marks = (rf" \hfill {{\small [{sec.marks} {_WORDS['Marks'].lower()}]}}"
                     if show_marks and sec.marks else "")
            lines.append(rf"\vspace{{1mm}}{{\large\bfseries {escape_text(head)}}}"
                         rf"{marks}\\[1mm]")
        if sec.instruction:
            lines.append(rf"{{\itshape {escape_text(sec.instruction)}}}\\[2mm]")
        if sec.passage:
            # A reading text is set in a quote block so it reads as source
            # material rather than as another question.
            lines.append(r"\begin{quote}")
            for para in sec.passage.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                if para.startswith(_GLOSSARY_MARK):
                    # Vocabulary help, set small and apart from the prose.
                    body = para[len(_GLOSSARY_MARK):].strip()
                    entries = r" \quad ".join(
                        escape_text(e.strip()) for e in body.split("|") if e.strip())
                    lines.append(r"\vspace{1mm}{\small " + entries + "}")
                else:
                    lines.append(escape_text(para) + r"\\[2mm]")
            lines.append(r"\end{quote}\vspace{2mm}")
        if not sec.exercises:
            continue
        lines.append(r"\begin{enumerate}[leftmargin=*,itemsep=0pt]")
        for ex in sec.exercises:
            lines += _item(ex, answers=answers,
                           marks=sec.marks_each if show_marks else None,
                           model=sec.model_answers)
        lines.append(r"\end{enumerate}")
        lines.append(r"\vspace{3mm}")

    if show_marks and not answers:
        lines.append(r"\vfill\hrule\vspace{1mm}")
        lines.append(rf"\hfill\textbf{{{_WORDS['Total']}: "
                     rf"\underline{{\hspace{{1.5cm}}}} / {total}}}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def render_pdf(sections: list[Section], *, title: str, subtitle: str,
               kind: str = "homework", answers: bool = False,
               student: str = "") -> bytes | None:
    """Typeset a sheet, or its answer key, to PDF bytes."""
    if not sections or not any(s.exercises or s.passage for s in sections):
        return None
    show_marks = kind in ("test", "exam")
    tex = _build_tex(sections, title=title, subtitle=subtitle, kind=kind,
                     answers=answers, student=student, show_marks=show_marks)
    return compile_pdf(tex, tag="tutor")


def render_html(sections: list[Section], *, title: str, subtitle: str,
                kind: str = "homework", answers: bool = False) -> str:
    """Browser preview, for when LaTeX is unavailable."""
    import html as _html

    show_marks = kind in ("test", "exam")
    parts = [f"<h2>{_html.escape(title)}</h2>",
             f"<p class='muted'>{_html.escape(subtitle)} — "
             f"{'Answer Key' if answers else _KIND_TITLE.get(kind, 'Sheet')}</p>"]
    for n, sec in enumerate(sections, 1):
        parts.append(f"<h3>{_html.escape(sec.title or f'Part {n}')}"
                     + (f" <small>[{sec.marks}]</small>" if show_marks else "")
                     + "</h3>")
        if sec.instruction:
            parts.append(f"<p><em>{_html.escape(sec.instruction)}</em></p>")
        if sec.passage:
            parts.append(f"<blockquote>{_html.escape(sec.passage)}</blockquote>")
        parts.append("<ol>")
        for ex in sec.exercises:
            parts.append("<li>")
            parts.append(f"<div>\\({ex.prompt}\\)</div>")
            if ex.options:
                parts.append("<div class='muted'>("
                             + " &nbsp; ".join(_html.escape(o) for o in ex.options)
                             + ")</div>")
            if answers:
                label = "Model answer" if sec.model_answers else "Answer"
                parts.append(f"<div><b>{label}:</b> {_html.escape(ex.answer)}</div>")
            parts.append("</li>")
        parts.append("</ol>")
    return "\n".join(parts)


__all__ = ["Section", "render_pdf", "render_html", "latex_available",
           "render_lesson_pdf", "lesson_has_answers"]


def render_lesson_pdf(markdown: str, *, title: str, subtitle: str,
                      answers: bool = False, student: str = "",
                      practice_space: bool = True) -> bytes | None:
    """Print a Markdown lesson as a sheet, or as its answer key.

    The lesson the model writes already ends with a Solutions section, so the
    key needs no second model call: the sheet is everything before Solutions,
    and the key is the Solutions section. That also guarantees the two agree —
    they are the same document, cut in half.
    """
    from researcher.tutor.markdown_tex import split_solutions, to_latex

    body, solutions = split_solutions(markdown or "")
    if answers:
        if not solutions:
            return None          # no Solutions section: say so, do not invent
        content = to_latex(solutions)
    else:
        content = to_latex(body, blank_lines=practice_space)

    lines = [preamble(11)]
    lines.append(rf"\fancyhead[L]{{\small {escape_text(title)}}}")
    lines.append(rf"\fancyhead[R]{{\small {escape_text(subtitle)}}}")
    lines.append(r"\fancyfoot[C]{\small\thepage}")
    lines.append(r"\begin{document}")
    kind = _WORDS["Answer Key"] if answers else "Lesson"
    lines.append(rf"\begin{{center}}\Large\textbf{{{escape_text(title)}}}\\[2pt]")
    lines.append(rf"\normalsize {escape_text(subtitle)} \quad"
                 rf"\textbf{{{kind}}}\end{{center}}")
    if not answers:
        lines.append(r"\vspace{2mm}")
        who = (rf"\underline{{\hspace{{1mm}}{escape_text(student)}"
               rf"\hspace{{6cm}}}}" if student
               else r"\underline{\hspace{6cm}}")
        lines.append(rf"{_WORDS['Name']}: {who} \hfill {_WORDS['Date']}: "
                     rf"\underline{{\hspace{{3.5cm}}}}")
        lines.append(r"\vspace{3mm}\hrule")
    lines.append(content)
    lines.append(r"\end{document}")
    return compile_pdf("\n".join(lines), tag="tutor")


def lesson_has_answers(markdown: str) -> bool:
    """Whether a lesson carries a Solutions section to print as a key."""
    from researcher.tutor.markdown_tex import split_solutions

    return bool(split_solutions(markdown or "")[1])
