"""Instrument ranges, transposition and playability.

The composer previously wrote notes with no idea which instrument would play
them. Nothing stopped it going below a violin's open G, and transposing
instruments were written at concert pitch — which is *wrong on the page* to any
player, however well it sounds in MIDI. That is the most conspicuously amateur
thing a generated score can do.

music21 carries some of this data but not enough to rely on:

  * ``highestNote`` is ``None`` for essentially every orchestral instrument.
  * Voices (``Soprano``, ``Bass``) have no range at all.

So the table below is explicit. Ranges are the *practical* ones a competent
player is comfortable with, not the extremes a virtuoso can reach — a score that
sits at the absolute limit is unplayable in practice even though every note is
theoretically possible.
"""

import logging

logger = logging.getLogger(__name__)


# (lowest, highest) as WRITTEN pitches, plus the transposition needed to turn
# written pitch into sounding pitch. Written range is what matters here: it is
# what the player reads.
#
# transpose_semitones: sounding = written + n. Bb clarinet sounds a major 2nd
# BELOW what is written, hence -2.
INSTRUMENT_RANGES: dict[str, dict] = {
    # ── Strings ──
    "Violin":       {"low": "G3",  "high": "E7",  "transpose": 0,   "clef": "treble"},
    "Viola":        {"low": "C3",  "high": "E6",  "transpose": 0,   "clef": "alto"},
    "Cello":        {"low": "C2",  "high": "A5",  "transpose": 0,   "clef": "bass"},
    "Violoncello":  {"low": "C2",  "high": "A5",  "transpose": 0,   "clef": "bass"},
    # Written an octave above sounding, by long convention.
    "Double Bass":  {"low": "E2",  "high": "G4",  "transpose": -12, "clef": "bass"},
    "Contrabass":   {"low": "E2",  "high": "G4",  "transpose": -12, "clef": "bass"},
    "Harp":         {"low": "C1",  "high": "G7",  "transpose": 0,   "clef": "grand"},
    "Guitar":       {"low": "E3",  "high": "E6",  "transpose": -12, "clef": "treble"},

    # ── Woodwind ──
    "Flute":        {"low": "C4",  "high": "D7",  "transpose": 0,   "clef": "treble"},
    "Piccolo":      {"low": "D4",  "high": "C7",  "transpose": 12,  "clef": "treble"},
    "Oboe":         {"low": "Bb3", "high": "A6",  "transpose": 0,   "clef": "treble"},
    "English Horn": {"low": "E3",  "high": "A5",  "transpose": -7,  "clef": "treble"},
    "Clarinet":     {"low": "E3",  "high": "C7",  "transpose": -2,  "clef": "treble"},
    "Bass Clarinet":{"low": "E3",  "high": "G6",  "transpose": -14, "clef": "treble"},
    "Bassoon":      {"low": "Bb1", "high": "Eb5", "transpose": 0,   "clef": "bass"},
    "Alto Saxophone":  {"low": "Bb3", "high": "F6", "transpose": -9,  "clef": "treble"},
    "Tenor Saxophone": {"low": "Bb3", "high": "F6", "transpose": -14, "clef": "treble"},
    "Soprano Saxophone": {"low": "Bb3", "high": "F6", "transpose": -2, "clef": "treble"},

    # ── Brass ──
    "Trumpet":      {"low": "F#3", "high": "D6",  "transpose": -2,  "clef": "treble"},
    "Horn":         {"low": "C3",  "high": "C6",  "transpose": -7,  "clef": "treble"},
    "French Horn":  {"low": "C3",  "high": "C6",  "transpose": -7,  "clef": "treble"},
    "Trombone":     {"low": "E2",  "high": "Bb4", "transpose": 0,   "clef": "bass"},
    "Tuba":         {"low": "D1",  "high": "F4",  "transpose": 0,   "clef": "bass"},

    # ── Keyboard / tuned percussion ──
    "Piano":        {"low": "A0",  "high": "C8",  "transpose": 0,   "clef": "grand"},
    "Organ":        {"low": "C2",  "high": "C7",  "transpose": 0,   "clef": "grand"},
    "Celesta":      {"low": "C4",  "high": "C8",  "transpose": 12,  "clef": "treble"},
    "Vibraphone":   {"low": "F3",  "high": "F6",  "transpose": 0,   "clef": "treble"},
    "Marimba":      {"low": "C2",  "high": "C7",  "transpose": 0,   "clef": "grand"},
    "Glockenspiel": {"low": "G5",  "high": "C8",  "transpose": 24,  "clef": "treble"},

    # ── Voices ──
    # music21 supplies nothing here. These are comfortable choral ranges, not
    # soloist extremes: a section must sustain them, not merely reach them.
    "Soprano":      {"low": "C4",  "high": "A5",  "transpose": 0,   "clef": "treble"},
    "Mezzo-Soprano":{"low": "A3",  "high": "F5",  "transpose": 0,   "clef": "treble"},
    "Alto":         {"low": "F3",  "high": "D5",  "transpose": 0,   "clef": "treble"},
    "Contralto":    {"low": "E3",  "high": "C5",  "transpose": 0,   "clef": "treble"},
    # Tenor is written in treble clef but sounds an octave lower.
    "Tenor":        {"low": "C3",  "high": "A4",  "transpose": -12, "clef": "treble8vb"},
    "Baritone":     {"low": "A2",  "high": "F4",  "transpose": 0,   "clef": "bass"},
    "Bass":         {"low": "E2",  "high": "D4",  "transpose": 0,   "clef": "bass"},
}

# Instruments that cannot sustain indefinitely: wind and brass players breathe.
# A line with no gap is unplayable however comfortable each individual note is.
_BREATHING_REQUIRED = {
    "Flute", "Piccolo", "Oboe", "English Horn", "Clarinet", "Bass Clarinet",
    "Bassoon", "Trumpet", "Horn", "French Horn", "Trombone", "Tuba",
    "Alto Saxophone", "Tenor Saxophone", "Soprano Saxophone",
    "Soprano", "Mezzo-Soprano", "Alto", "Contralto", "Tenor", "Baritone", "Bass",
}

# Beats of continuous sound after which a player needs a rest, by family.
_MAX_PHRASE_BEATS = {"brass": 12.0, "wind": 16.0, "voice": 12.0}


def _family(name: str) -> str:
    if name in {"Trumpet", "Horn", "French Horn", "Trombone", "Tuba"}:
        return "brass"
    if name in {"Soprano", "Mezzo-Soprano", "Alto", "Contralto",
                "Tenor", "Baritone", "Bass"}:
        return "voice"
    if name in _BREATHING_REQUIRED:
        return "wind"
    return "other"


def canonical_name(name: str) -> str:
    """Map a UI label onto a table key, tolerating case and spacing."""
    if not name:
        return "Piano"
    cleaned = " ".join(str(name).strip().split())
    if cleaned in INSTRUMENT_RANGES:
        return cleaned
    lowered = cleaned.lower()
    for key in INSTRUMENT_RANGES:
        if key.lower() == lowered:
            return key
    aliases = {
        "double bass": "Double Bass", "contrabass": "Double Bass",
        "french horn": "Horn", "cor anglais": "English Horn",
        "alto sax": "Alto Saxophone", "tenor sax": "Tenor Saxophone",
        "vocals": "Soprano", "voice": "Soprano", "choir": "Soprano",
    }
    return aliases.get(lowered, "Piano")


def range_of(name: str) -> tuple[int, int]:
    """(low, high) MIDI numbers of the written range for *name*."""
    from music21 import pitch

    spec = INSTRUMENT_RANGES.get(canonical_name(name), INSTRUMENT_RANGES["Piano"])
    return pitch.Pitch(spec["low"]).midi, pitch.Pitch(spec["high"]).midi


def check_range(part, name: str) -> dict:
    """Report notes outside the instrument's practical range.

    Reporting rather than silently altering, so a caller can choose between
    fixing the notes and choosing a different instrument.
    """
    lo, hi = range_of(name)
    too_low, too_high = [], []
    for n in part.recurse().notes:
        for p in getattr(n, "pitches", []):
            if p.midi < lo:
                too_low.append(p.nameWithOctave)
            elif p.midi > hi:
                too_high.append(p.nameWithOctave)
    return {
        "instrument": canonical_name(name),
        "low": lo, "high": hi,
        "below": too_low, "above": too_high,
        "ok": not too_low and not too_high,
    }


def fit_to_range(part, name: str) -> int:
    """Move out-of-range notes by whole octaves until they fit.

    Octave displacement is the one transformation that preserves the music:
    the pitch class, and therefore the harmony, is unchanged. Clamping to the
    nearest playable note would silently rewrite the harmony instead.
    """
    lo, hi = range_of(name)
    moved = 0
    for n in part.recurse().notes:
        for p in getattr(n, "pitches", []):
            original = p.midi
            while p.midi < lo:
                p.midi += 12
            while p.midi > hi:
                p.midi -= 12
            # A range narrower than an octave can leave nothing valid; put it
            # back rather than emit a note outside the range in both directions.
            if p.midi < lo or p.midi > hi:
                p.midi = original
            elif p.midi != original:
                moved += 1
    return moved


def check_breathing(part, name: str) -> list[dict]:
    """Find phrases too long to play without a breath."""
    canonical = canonical_name(name)
    if canonical not in _BREATHING_REQUIRED:
        return []
    limit = _MAX_PHRASE_BEATS.get(_family(canonical), 16.0)

    problems, run, start = [], 0.0, 0.0
    for el in part.recurse().notesAndRests:
        if el.isRest:
            if run > limit:
                problems.append({"offset": start, "beats": round(run, 1),
                                 "limit": limit})
            run, start = 0.0, float(el.offset) + float(el.quarterLength)
        else:
            if run == 0.0:
                start = float(el.offset)
            run += float(el.quarterLength)
    if run > limit:
        problems.append({"offset": start, "beats": round(run, 1), "limit": limit})
    return problems


def apply_instrument(part, name: str, fit: bool = True) -> dict:
    """Attach the instrument to a part, fix its range and set transposition.

    Returns a report describing what had to change.
    """
    from music21 import instrument as m21inst, interval, clef

    canonical = canonical_name(name)
    spec = INSTRUMENT_RANGES[canonical]
    report = {"instrument": canonical, "notes_moved": 0,
              "transposed": spec["transpose"] != 0, "breathing": []}

    before = check_range(part, canonical)
    if fit and not before["ok"]:
        report["notes_moved"] = fit_to_range(part, canonical)

    # Remove any instrument already on the part before attaching the real one.
    # assemble() inserts a Piano placeholder, and music21 honours the FIRST
    # instrument it finds — so simply adding another leaves every part reading
    # "Piano" while the report cheerfully claims otherwise.
    for existing in list(part.recurse().getElementsByClass(m21inst.Instrument)):
        try:
            existing.activeSite.remove(existing)
        except Exception:
            pass

    # Attach a real music21 instrument so MusicXML and MIDI both carry it.
    try:
        cls = "".join(w.capitalize() for w in canonical.split())
        ins = getattr(m21inst, cls, None)
        ins = ins() if ins else m21inst.fromString(canonical)
    except Exception:
        ins = m21inst.Piano()
    if spec["transpose"]:
        # music21 writes the sounding pitch and applies this to display the
        # written part, which is what a player actually reads.
        try:
            ins.transposition = interval.Interval(spec["transpose"])
        except Exception:
            pass
    part.insert(0, ins)

    if spec["clef"] == "bass":
        part.insert(0, clef.BassClef())
    elif spec["clef"] == "alto":
        part.insert(0, clef.AltoClef())
    elif spec["clef"] == "treble8vb":
        part.insert(0, clef.Treble8vbClef())

    report["breathing"] = check_breathing(part, canonical)
    report["range_ok_after"] = check_range(part, canonical)["ok"]
    return report


def orchestrate(score, instruments: list[str]) -> dict:
    """Assign instruments to a score's parts in order.

    Fewer instruments than parts reuses the last one; more are ignored.
    """
    parts = list(score.parts) if hasattr(score, "parts") else []
    if not parts:
        return {"parts": 0, "reports": []}
    names = [n for n in (instruments or []) if n] or ["Piano"]

    reports = []
    for i, part in enumerate(parts):
        name = names[i] if i < len(names) else names[-1]
        reports.append(apply_instrument(part, name))
    return {"parts": len(parts), "reports": reports}
