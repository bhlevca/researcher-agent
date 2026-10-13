"""Rule-based expansion of regular verb paradigms.

Why this exists
---------------
A pack entry can give every form explicitly, which is right for irregular verbs.
For regular ones it is both tedious and *less* reliable: a regular French -er
verb needs 24 forms plus a participle, and typing forty of those by hand means
typing a thousand forms, some of which will be wrong.

So a regular verb is one line — ``{inf: donner, gloss: to give, class: er}`` —
and the forms are derived here. The derivation is checked-in code, exercised by
tests that expand the verbs the packs *also* spell out by hand and require the
two to agree. That is a stronger guarantee than hand-typing gives: it is checked
against known-good data rather than against nobody.

Irregular verbs keep their explicit tables and are never touched.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# French
# ---------------------------------------------------------------------------

_FR_PRONOUNS = ["je", "tu", "il", "nous", "vous", "ils"]
_FR_PRESENT = {
    "er": ["e", "es", "e", "ons", "ez", "ent"],
    "ir": ["is", "is", "it", "issons", "issez", "issent"],
    "re": ["s", "s", "", "ons", "ez", "ent"],
}
_FR_IMPARFAIT = ["ais", "ais", "ait", "ions", "iez", "aient"]
_FR_FUTUR = ["ai", "as", "a", "ons", "ez", "ont"]
_FR_PP = {"er": "é", "ir": "i", "re": "u"}


def _fr_stem_changes(inf: str, cls: str) -> tuple[str, str, str]:
    """Return (stem, strong_stem, future_stem) for the orthographic subclasses.

    *strong_stem* is used where the ending is silent — je/tu/il/ils — which is
    exactly where acheter becomes achète and appeler becomes appelle.
    """
    stem = inf[:-2] if cls in ("er", "ir", "re") else inf[:-2]
    if cls == "re":
        stem = inf[:-2]
    strong = stem
    future = inf if cls != "re" else inf[:-1]

    if cls == "e_er":          # acheter, lever, mener — e becomes è
        stem = inf[:-2]
        i = max(stem.rfind("e"), 0)
        strong = stem[:i] + "è" + stem[i + 1:]
        future = strong + "er"
    elif cls == "ele_er":      # appeler, jeter — the consonant doubles
        stem = inf[:-2]
        strong = stem + stem[-1]
        future = strong + "er"
    elif cls == "ere_er":      # préférer, espérer — é becomes è in the present
        stem = inf[:-2]
        i = stem.rfind("é")
        strong = stem[:i] + "è" + stem[i + 1:] if i >= 0 else stem
        future = inf           # the future keeps é in traditional spelling
    return stem, strong, future


def expand_fr(entry: dict) -> dict:
    """Fill in a regular French verb from its infinitive and class."""
    inf = entry["inf"]
    cls = entry["class"]
    base = {"e_er": "er", "ele_er": "er", "ere_er": "er",
            "ger": "er", "cer": "er"}.get(cls, cls)
    if base not in _FR_PRESENT:
        raise ValueError(f"unknown French verb class '{cls}'")

    stem, strong, future = _fr_stem_changes(inf, cls)
    endings = _FR_PRESENT[base]

    present = {}
    for i, p in enumerate(_FR_PRONOUNS):
        # The stem change applies where the ending is silent.
        s = strong if (base == "er" and i in (0, 1, 2, 5)) else stem
        if cls == "ger" and endings[i] == "ons":
            s = stem + "e"          # mangeons keeps the soft g
        if cls == "cer" and endings[i] == "ons":
            s = stem[:-1] + "ç"     # commençons keeps the soft c
        present[p] = s + endings[i]

    # The imperfect stem is the nous-form minus -ons, which is where the
    # -ge-/-ç- of manger and commencer survives into mangeais, commençais.
    imp_stem = present["nous"][:-3]
    imparfait = {}
    for i, p in enumerate(_FR_PRONOUNS):
        s = imp_stem
        if cls == "ger" and _FR_IMPARFAIT[i].startswith("i"):
            s = imp_stem[:-1]       # nous mangions, not mangeions
        if cls == "cer" and _FR_IMPARFAIT[i].startswith("i"):
            s = imp_stem[:-1] + "c"
        imparfait[p] = s + _FR_IMPARFAIT[i]

    futur = {p: future + _FR_FUTUR[i] for i, p in enumerate(_FR_PRONOUNS)}

    out = dict(entry)
    out.pop("class", None)
    out["present"] = present
    out["imparfait"] = imparfait
    out["futur"] = futur
    out.setdefault("passe_compose",
                   {"aux": entry.get("aux", "avoir"),
                    "pp": stem + _FR_PP[base]})
    out.pop("aux", None)
    return out


# ---------------------------------------------------------------------------
# German — weak verbs
# ---------------------------------------------------------------------------

_DE_PRONOUNS = ["ich", "du", "er", "wir", "ihr", "sie"]


def _de_needs_e(stem: str) -> bool:
    """Stems ending in t, d or a consonant cluster take a linking -e-.

    arbeiten -> du arbeitest, er arbeitete, gearbeitet. Without this the forms
    are unpronounceable and wrong.
    """
    return stem.endswith(("t", "d")) or stem[-2:] in ("gn", "fn", "chn", "dn")


# Verbs with an inseparable prefix, and verbs in -ieren, form their participle
# WITHOUT ge-: besucht, not gebesucht; studiert, not gestudiert.
_DE_NO_GE = ("be", "ge", "er", "ver", "zer", "ent", "emp", "miss")


def expand_de(entry: dict) -> dict:
    """Fill in a regular (weak) German verb."""
    inf = entry["inf"]
    # -ern and -eln verbs keep the -er-/-el- in the stem: wandern -> wander.
    if inf.endswith(("ern", "eln")):
        stem = inf[:-1]
    elif inf.endswith("en"):
        stem = inf[:-2]
    else:
        stem = inf[:-1]
    e = "e" if _de_needs_e(stem) else ""

    present = {
        "ich": stem + "e",
        "du": stem + e + "st",
        "er": stem + e + "t",
        "wir": inf,
        "ihr": stem + e + "t",
        "sie": inf,
    }
    praet = {
        "ich": stem + e + "te",
        "du": stem + e + "test",
        "er": stem + e + "te",
        "wir": stem + e + "ten",
        "ihr": stem + e + "tet",
        "sie": stem + e + "ten",
    }
    out = dict(entry)
    out.pop("class", None)
    out["present"] = present
    out["praeteritum"] = praet
    ge = "" if (inf.startswith(_DE_NO_GE) or inf.endswith("ieren")) else "ge"
    out.setdefault("perfekt", {"aux": entry.get("aux", "haben"),
                               "pp": f"{ge}{stem}{e}t"})
    out.pop("aux", None)
    return out


# ---------------------------------------------------------------------------
# Romanian
# ---------------------------------------------------------------------------

_RO_PRONOUNS = ["eu", "tu", "el", "noi", "voi", "ei"]


def expand_ro(entry: dict) -> dict:
    """Fill in a regular Romanian verb of the productive -ez or -esc classes.

    These two are where new verbs go in modern Romanian, so they cover most of
    what a learner meets. Verbs with vowel alternation (a mânca -> mănânc) are
    irregular here and keep explicit tables.
    """
    inf = entry["inf"]
    cls = entry["class"]
    bare = inf[2:] if inf.startswith("a ") else inf
    stem = bare[:-1]                       # drop the final -a or -i

    if cls == "ez":                        # conjugation I: a lucra
        present = {"eu": stem + "ez", "tu": stem + "ezi", "el": stem + "ează",
                   "noi": stem + "ăm", "voi": stem + "ați", "ei": stem + "ează"}
        imperfect = {"eu": stem + "am", "tu": stem + "ai", "el": stem + "a",
                     "noi": stem + "am", "voi": stem + "ați", "ei": stem + "au"}
        participle = stem + "at"
    elif cls == "esc":                     # conjugation IV: a citi
        present = {"eu": stem + "esc", "tu": stem + "ești", "el": stem + "ește",
                   "noi": stem + "im", "voi": stem + "iți", "ei": stem + "esc"}
        imperfect = {"eu": stem + "eam", "tu": stem + "eai", "el": stem + "ea",
                     "noi": stem + "eam", "voi": stem + "eați", "ei": stem + "eau"}
        participle = stem + "it"
    else:
        raise ValueError(f"unknown Romanian verb class '{cls}'")

    out = dict(entry)
    out.pop("class", None)
    out["present"] = present
    out["imperfect"] = imperfect
    out.setdefault("participle", participle)
    out.setdefault("bare_inf", bare)
    return out


EXPANDERS = {"fr": expand_fr, "de": expand_de, "ro": expand_ro}


def expand_pack(pack: dict, code: str) -> dict:
    """Expand every verb in *pack* that gives a class instead of full forms."""
    fn = EXPANDERS.get(code)
    if not fn:
        return pack
    out = []
    for v in pack.get("verbs", []):
        if "class" not in v:
            out.append(v)
            continue
        try:
            out.append(fn(v))
        except Exception as e:
            logger.warning("[tutor] could not expand %s: %s", v.get("inf"), e)
    pack["verbs"] = out
    return pack
