"""Story sheets: a levelled story, comprehension questions and a summary task.

This is the half of the language tutor that genuinely needs a model. A story
cannot be derived from a table, so unlike the grammar sheets these answers come
from the model — and two things follow, both of which are the point rather than
a compromise:

1. **The key says "Model answer".** A comprehension answer is one acceptable
   answer, not the answer. Printing it as though it were the answer invites an
   adult to mark a better answer wrong.
2. **The generated text is stored, not the recipe.** For grammar, a seed
   regenerates the sheet exactly; a second model call produces a different
   story, so storing a seed would mean a reprint that no longer matches the
   paper the child was given. The story is kept verbatim.

The model's output is JSON, repaired if it arrives slightly malformed, and
validated before anything reaches a sheet — a missing key must fail loudly here
rather than print as an empty question.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# What a usable story sheet must contain.
_REQUIRED = ("title", "story", "questions")


def parse_story(raw: str) -> dict:
    """Pull the story object out of a model response.

    Models wrap JSON in prose and code fences however they feel. json_repair
    handles the common malformations (trailing commas, unescaped newlines) that
    would otherwise throw away an otherwise good story.
    """
    import re

    candidates = []
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw or "")
    candidates.extend(fenced)
    # The outermost braces, for a response with prose around the JSON.
    first, last = (raw or "").find("{"), (raw or "").rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first:last + 1])
    candidates.append(raw or "")

    for block in candidates:
        block = block.strip()
        if not block:
            continue
        for loader in (json.loads, _repair_load):
            try:
                data = loader(block)
            except Exception:
                continue
            if isinstance(data, dict) and all(k in data for k in _REQUIRED):
                return _clean(data)
    raise ValueError("the model did not return a usable story object")


def _repair_load(block: str):
    from json_repair import repair_json

    return json.loads(repair_json(block))


def _clean(data: dict) -> dict:
    """Normalise the shape so the printer can rely on it."""
    questions = []
    for q in data.get("questions") or []:
        if isinstance(q, dict) and q.get("q"):
            questions.append({"q": str(q["q"]).strip(),
                              "a": str(q.get("a", "")).strip()})
        elif isinstance(q, str) and q.strip():
            questions.append({"q": q.strip(), "a": ""})
    glossary = []
    for g in data.get("glossary") or []:
        if isinstance(g, dict) and g.get("word"):
            glossary.append({"word": str(g["word"]).strip(),
                             "gloss": str(g.get("gloss", "")).strip()})
    return {
        "title": str(data.get("title", "")).strip(),
        "story": str(data.get("story", "")).strip(),
        "questions": questions,
        "summary_prompt": str(data.get("summary_prompt", "")).strip(),
        "summary_answer": str(data.get("summary_answer", "")).strip(),
        "glossary": glossary,
    }


class _Q:
    """An exercise-shaped object, so printing.py needs no special case."""

    __slots__ = ("prompt", "answer", "hint", "skill", "level", "lang", "options")

    def __init__(self, prompt: str, answer: str, skill: str, level: str,
                 lang: str, hint: str = ""):
        self.prompt = prompt
        self.answer = answer
        self.hint = hint
        self.skill = skill
        self.level = level
        self.lang = lang
        self.options = []

    def as_dict(self) -> dict:
        return {"prompt": self.prompt, "answer": self.answer, "hint": self.hint,
                "skill": self.skill, "level": self.level, "lang": self.lang,
                "options": self.options}


def _esc(text: str) -> str:
    r"""Wrap prose for the printer, which expects words inside \text{}."""
    return r"\text{" + (text or "").replace("{", "(").replace("}", ")") + "}"


def to_sections(content: dict, *, level: str, lang: str, marks_each: int = 2,
                summary_marks: int = 5) -> list:
    """Turn a parsed story into printable sections.

    Three sections: the story itself, the comprehension questions, and the
    summary. Each question section is flagged ``model_answers`` so the key
    prints "Model answer" rather than "Answer".
    """
    from researcher.tutor.printing import Section

    out: list[Section] = []

    passage = content.get("story", "")
    if content.get("glossary"):
        # Marked and pipe-separated; the printer sets the entries apart so they
        # do not run together into one unreadable line.
        from researcher.tutor.printing import _GLOSSARY_MARK

        gloss = " | ".join(f"{g['word']} — {g['gloss']}"
                           for g in content["glossary"])
        passage = f"{passage}\n\n{_GLOSSARY_MARK} {gloss}"
    out.append(Section(title=content.get("title") or "Reading",
                       instruction="Read the story, then answer the questions.",
                       passage=passage))

    if content.get("questions"):
        out.append(Section(
            title="Comprehension",
            instruction="Answer in full sentences.",
            exercises=[_Q(_esc(q["q"]), q["a"] or "—", "comprehension",
                          level, lang) for q in content["questions"]],
            marks_each=marks_each,
            model_answers=True))

    if content.get("summary_prompt"):
        out.append(Section(
            title="Summary",
            instruction="",
            exercises=[_Q(_esc(content["summary_prompt"]),
                          content.get("summary_answer") or "—",
                          "summary", level, lang)],
            marks_each=summary_marks,
            model_answers=True))
    return out
