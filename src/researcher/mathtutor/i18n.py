"""Multilingual question text: English, French, Romanian.

Why whole sentences rather than words
-------------------------------------
The obvious approach — translate "Find", "the area", "of this circle" and glue
them together — produces French and Romanian that is visibly machine-made:
wrong gender, wrong article, wrong word order. "Trouver le aire de ce cercle"
is not something you hand a child.

So the unit of translation here is the **whole sentence frame**, with the
mathematics interpolated into it. Each generator builds its question from
exactly one frame, which is what lets a translator put the words in a different
order from the English without anything breaking.

The frames live in phrases.yaml rather than in this file so they can be read and
corrected by someone who speaks the language without touching Python.

No language model is involved. These are written translations, checked in, and
stable: the same question in French today is the same question in French next
year, which matters when a child is working through a course.

Bilingual sheets
----------------
A language is either a single code ("fr") or a pair ("fr+en"). For a pair, T()
returns both renderings joined by ALT, and generate() splits them into the
problem's main and alternate text. Generators stay unaware of this, which is why
the one-frame-per-question rule is worth keeping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

# Separator between the two renderings of a bilingual string. A control
# character, so it can never occur in real question text.
ALT = "\x00"

LANGUAGES = {
    "en": "English",
    "fr": "Français",
    "ro": "Română",
}

# Pairs offered in the UI. English second, because the second line is the
# support line — the child reads the first.
BILINGUAL = ["fr+en", "ro+en", "en+fr", "en+ro", "fr+ro", "ro+fr"]


def available_languages() -> list[dict]:
    """Every selectable language option, single and paired."""
    out = [{"id": c, "label": n, "bilingual": False} for c, n in LANGUAGES.items()]
    for pair in BILINGUAL:
        a, b = pair.split("+")
        out.append({"id": pair, "label": f"{LANGUAGES[a]} + {LANGUAGES[b]}",
                    "bilingual": True})
    return out


@lru_cache(maxsize=1)
def phrases() -> dict:
    import yaml

    path = Path(__file__).parent / "phrases.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def codes(lang: str) -> list[str]:
    """Split a language spec into its one or two codes."""
    parts = [p.strip() for p in (lang or "en").split("+") if p.strip()]
    good = [p for p in parts if p in LANGUAGES]
    return good or ["en"]


def is_bilingual(lang: str) -> bool:
    return len(codes(lang)) > 1


def fmt_num(value, lang: str = "en") -> str:
    """Format a number for the language.

    French and Romanian both write the decimal separator as a comma. Printing
    "4.4 m" on a French worksheet is the kind of detail that tells a teacher the
    sheet was not really written in French.
    """
    primary = codes(lang)[0]
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace(".", ",") if primary in ("fr", "ro") else text


def _render(key: str, code: str, kw: dict) -> str:
    r"""Fill a frame, wrapping the words — and only the words — in \text{}.

    The result is one math expression with prose inside \text{...}, which is
    what the rest of the package already expects: worksheet.py pulls those
    segments back out to make a wrappable paragraph, and the HTML preview hands
    the whole thing to a math renderer.

    Wrapping the filled string as a whole would set the mathematics in text
    mode, so the literal and substituted parts are separated here rather than
    after the fact.
    """
    from string import Formatter

    entry = phrases().get(key)
    if entry is None:
        logger.warning("[i18n] no phrase '%s'", key)
        return rf"\text{{{key}}}"
    template = entry.get(code) or entry.get("en") or key
    if entry.get(code) is None:
        logger.warning("[i18n] phrase '%s' has no %s translation", key, code)

    out: list[str] = []
    for literal, field, _spec, _conv in Formatter().parse(template):
        if literal:
            out.append(rf"\text{{{literal}}}")
        if field is None:
            continue
        if field not in kw:
            logger.warning("[i18n] phrase '%s' (%s) wants '%s'", key, code, field)
            continue
        out.append(str(kw[field]))
    return "".join(out)


def T(key: str, lang: str = "en", **kw) -> str:
    r"""The sentence frame *key*, in *lang*, with **kw interpolated.

    Numbers passed in are formatted for the language, so a generator can pass a
    raw value and still get "4,4" on a French sheet. LaTeX fragments pass
    through untouched.
    """
    cs = codes(lang)
    out = []
    for c in cs:
        local = {k: (fmt_num(v, c) if isinstance(v, (int, float)) else v)
                 for k, v in kw.items()}
        out.append(_render(key, c, local))
    return ALT.join(out)


def split_alt(text: str) -> tuple[str, str]:
    """Separate a bilingual string into its main and alternate renderings."""
    if ALT not in text:
        return text, ""
    first, _, second = text.partition(ALT)
    return first, second


def missing() -> list[tuple[str, str]]:
    """Every (phrase, language) pair with no translation.

    Used by the test suite: a half-translated catalogue produces a sheet with
    English sentences scattered through the French, which is worse than not
    offering French at all.
    """
    gaps = []
    for key, entry in phrases().items():
        for code in LANGUAGES:
            if not (entry or {}).get(code):
                gaps.append((key, code))
    return gaps
