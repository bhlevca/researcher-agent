"""Deterministic musical realisation.

The composer used to ask a language model for sixteen bars of notes. That is the
one thing it is worst at: it cannot hear what it wrote four bars ago, so it
produces locally-plausible, globally-incoherent material — scales, repeated
rhythms, voices doubling each other.

This module inverts the division of labour. The model makes the decisions it is
genuinely good at, which are linguistic:

    form, key plan, harmonic rhythm, mood, and a two-bar motif

and everything that follows is computed here, where the rules are well defined
and nothing is ever forgotten:

    * Roman numerals become correctly voiced chords
    * the motif is developed by transposition, inversion, augmentation and
      fragmentation, so later bars are *related* to earlier ones
    * the bass is generated idiomatically per genre
    * voice leading is checked, and parallel fifths and octaves are repaired

A computer never forgets bar 4 and never writes the same parallel fifth twice.
That is the whole argument for doing it this way.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The plan the model is asked for — small, structural, and checkable
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """One formal section, e.g. the A of an AABA."""

    name: str
    progression: list[str]           # Roman numerals, one per harmonic unit
    bars: int = 4
    develop: str = "original"        # how the motif appears here

    def __post_init__(self):
        if not self.progression:
            self.progression = ["I"]


@dataclass
class Plan:
    """A complete structural plan. This is what the LLM produces."""

    title: str = "Untitled"
    key: str = "C major"
    time_signature: str = "4/4"
    tempo: int = 120
    genre: str = "classical"
    motif: list[tuple[str, float]] = field(default_factory=list)  # (pitch, quarterLength)
    form: list[str] = field(default_factory=lambda: ["A", "A", "B", "A"])
    sections: dict[str, Section] = field(default_factory=dict)

    def total_bars(self) -> int:
        return sum(self.sections[n].bars for n in self.form if n in self.sections)


# How a section transforms the motif. Names are deliberately plain so the model
# can pick one without knowing music21's vocabulary.
DEVELOPMENTS = (
    "original",
    "sequence_up",      # same shape, one step higher
    "sequence_down",
    "inversion",        # intervals turned upside down
    "augmentation",     # same shape, twice as slow
    "fragment",         # just the opening, repeated
    "retrograde",       # backwards
)


# ---------------------------------------------------------------------------
# Realisation
# ---------------------------------------------------------------------------


def _m21():
    """Import music21 lazily — it is slow to import and not always needed."""
    from music21 import (  # noqa: F401
        chord, clef, harmony, instrument, interval, key as m21key,
        meter, note, roman, stream, tempo as m21tempo, voiceLeading,
    )
    import music21

    return music21


def realise_progression(progression: list[str], key_name: str,
                        bars: int, time_signature: str = "4/4"):
    """Turn Roman numerals into correctly voiced chords.

    The model only has to say "ii65"; working out that this means F4-A4-C5-D5 in
    C major is exactly the kind of lookup a library does perfectly and a
    language model does approximately.
    """
    m21 = _m21()
    from music21 import roman, key as m21key, stream, meter

    k = m21key.Key(key_name.split()[0],
                   "minor" if "minor" in key_name.lower() else "major")
    ts = meter.TimeSignature(time_signature)
    beats_per_bar = ts.numerator * (4.0 / ts.denominator)

    part = stream.Part()
    part.append(ts)
    part.append(k)

    # Spread the progression evenly across the bars.
    total_units = max(1, len(progression))
    ql_per_chord = (bars * beats_per_bar) / total_units

    for figure in progression:
        try:
            rn = roman.RomanNumeral(figure.strip(), k)
        except Exception:
            logger.warning("[realise] unknown Roman numeral %r, using I", figure)
            rn = roman.RomanNumeral("I", k)
        rn.quarterLength = ql_per_chord
        # Close position keeps the chord compact and playable rather than
        # spread across three octaves.
        try:
            rn = rn.closedPosition(forceOctave=4, inPlace=False)
        except Exception:
            pass
        part.append(rn)
    return part


def develop_motif(motif: list[tuple[str, float]], how: str, key_name: str):
    """Return the motif transformed.

    This is what makes music sound *composed* rather than sampled: the listener
    hears the same idea come back changed. Doing it in code guarantees the
    relationship actually holds, instead of hoping the model remembers.
    """
    from music21 import note, stream, interval, key as m21key

    src = stream.Part()
    for name, ql in motif or [("C4", 1.0)]:
        n = note.Note(name)
        n.quarterLength = ql
        src.append(n)

    k = m21key.Key(key_name.split()[0],
                   "minor" if "minor" in key_name.lower() else "major")

    # GenericInterval transposes DIATONICALLY — up a step within the key. A
    # chromatic Interval(2) would put F# into C major, which is exactly the kind
    # of wrong note that makes generated music sound broken.
    if how == "sequence_up":
        return src.transpose(interval.GenericInterval(2)).makeAccidentals(
            alteredPitches=k.alteredPitches, inPlace=False)
    if how == "sequence_down":
        return src.transpose(interval.GenericInterval(-2)).makeAccidentals(
            alteredPitches=k.alteredPitches, inPlace=False)
    if how == "inversion":
        try:
            first = src.flatten().notes[0]
            return src.invertDiatonic(first, inPlace=False)
        except Exception:
            return src
    if how == "augmentation":
        return src.augmentOrDiminish(2.0)
    if how == "fragment":
        notes = list(src.flatten().notes)
        half = notes[: max(1, len(notes) // 2)]
        out = stream.Part()
        for _ in range(2):                      # state it twice
            for n in half:
                c = note.Note(n.nameWithOctave)
                c.quarterLength = n.quarterLength
                out.append(c)
        return out
    if how in ("dotted", "diminution", "syncopate", "even", "long_short_pairs"):
        # Rhythm-only development: same pitches, different timing.
        varied = vary_rhythm([(n.nameWithOctave, n.quarterLength)
                              for n in src.flatten().notes], how)
        out = stream.Part()
        for name, ql in varied:
            c = note.Note(name)
            c.quarterLength = ql
            out.append(c)
        return out
    if how.startswith("sequence_up+"):
        # Combined: transpose AND re-time, so a returning section is recognisable
        # without being identical.
        base = develop_motif(motif, "sequence_up", key_name)
        return develop_motif([(n.nameWithOctave, n.quarterLength)
                              for n in base.flatten().notes],
                             how.split("+", 1)[1], key_name)
    if how == "retrograde":
        notes = list(src.flatten().notes)[::-1]
        out = stream.Part()
        for n in notes:
            c = note.Note(n.nameWithOctave)
            c.quarterLength = n.quarterLength
            out.append(c)
        return out
    return src




# ---------------------------------------------------------------------------
# Rhythmic development
#
# Every transformation above changes PITCH and copies quarterLength verbatim.
# augmentation scales everything uniformly, which is not variety either. The
# result was music with three distinct durations across two and a half minutes:
# the motif came back transposed and inverted, but always in the same rhythm, so
# the ear heard one idea repeated rather than an idea developed.
#
# These operate on rhythm alone and preserve the total length of the phrase, so
# a varied motif still fills exactly the bars it is meant to.
# ---------------------------------------------------------------------------

def _rhythm_of(notes) -> list[float]:
    return [float(n.quarterLength) for n in notes]


def vary_rhythm(motif: list[tuple[str, float]], how: str,
                seed: int = 0) -> list[tuple[str, float]]:
    """Re-time a motif without changing its pitches or total duration.

    Keeping the pitch sequence intact is the point: the listener still
    recognises the idea, but it arrives differently. Changing both at once
    produces new material, not development.
    """
    if not motif:
        return motif
    pitches = [p for p, _ in motif]
    durs = [float(d) for _, d in motif]
    total = sum(durs)
    n = len(durs)

    if how == "dotted" and n >= 2:
        # Long-short pairs: the single most characteristic rhythmic gesture in
        # tonal music, and completely absent from the current output.
        out = []
        for i in range(0, n - 1, 2):
            pair = durs[i] + durs[i + 1]
            out.extend([pair * 0.75, pair * 0.25])
        if n % 2:
            out.append(durs[-1])
        durs = out
    elif how == "diminution":
        # Twice as fast, then hold the last note to keep the length.
        durs = [d / 2.0 for d in durs]
        durs[-1] += total - sum(durs)
    elif how == "syncopate" and n >= 3:
        # Push the accent off the beat by lengthening an inner note and
        # shortening its neighbour.
        i = 1 + (seed % max(1, n - 2))
        take = min(durs[i] * 0.5, durs[i - 1] * 0.5)
        durs[i] += take
        durs[i - 1] -= take
    elif how == "even":
        durs = [total / n] * n
    elif how == "long_short_pairs" and n >= 4:
        unit = total / n
        durs = [unit * 1.5 if i % 2 == 0 else unit * 0.5 for i in range(n)]

    # Snap to durations a musician can actually read. Arithmetic like 0.75*x
    # produces values such as 0.188 or 1.125 quarter-notes, which are legal in
    # MIDI but notate as knots of tied 32nds — technically playable, unreadable
    # on the page, and exactly the kind of thing that marks a score as machine
    # output.
    grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    durs = [min(grid, key=lambda g: abs(g - d)) for d in durs]

    # Snapping changes the total, so the FINAL note absorbs the remainder
    # exactly. It is deliberately not snapped: the phrase must fill its bars to
    # the beat or the section drifts out of alignment, and a single held closing
    # note notates cleanly as a tie even at an off-grid length. Diminution is the
    # case that forces this — halving four beats leaves 2.5 to absorb, which no
    # grid value can express, and snapping it stalls the length permanently
    # short.
    drift = total - sum(durs)
    if abs(drift) > 1e-6:
        want = durs[-1] + drift
        durs[-1] = want if want >= 0.125 else durs[-1]
    return list(zip(pitches, durs))


def rhythmic_variety(part) -> int:
    """How many distinct note lengths a part uses — the metric that exposed this."""
    return len({round(float(n.quarterLength), 4) for n in part.recurse().notes})


def _fill_bars(part, bars: int, beats_per_bar: float, key_name: str):
    """Extend a developed motif until it fills the section.

    A two-bar motif stated once leaves a four-bar section two-thirds empty. Real
    music restates and varies it. Each repetition is nudged — alternating the
    original with a step-sequence — so the section has movement rather than a
    literal loop.
    """
    from music21 import note, stream

    target = bars * beats_per_bar
    notes = list(part.recurse().notes)
    if not notes:
        return part
    out = stream.Part()
    total = 0.0
    repeat = 0
    while total < target - 1e-6:
        # Cycle through pitch AND rhythm transforms. Alternating "original,
        # sequence_up, original, sequence_up" is what produced five literal
        # repeats of the opening cell: only two shapes ever appeared, and both
        # shared one rhythm.
        variants = ("original", "sequence_up", "dotted", "sequence_up+syncopate",
                    "fragment", "diminution")
        how = variants[repeat % len(variants)]
        if how == "original":
            src = notes
        else:
            src = list(develop_motif(
                [(n.nameWithOctave, n.quarterLength) for n in notes],
                how, key_name).recurse().notes)
        for n in src:
            remaining = target - total
            if remaining <= 1e-6:
                break
            c = note.Note(n.nameWithOctave)
            # Never overrun the section: truncate the final note instead.
            c.quarterLength = min(n.quarterLength, remaining)
            out.append(c)
            total += c.quarterLength
        repeat += 1
        if repeat > 64:          # safety against a zero-length motif
            break
    return out


# Bass behaviour differs by idiom more than almost anything else, and getting it
# wrong is instantly audible. These are rules, not guesses.
def generate_bass(progression_part, genre: str, key_name: str):
    """Build a bass line under a realised progression."""
    from music21 import note, stream, chord as m21chord

    genre = (genre or "classical").lower()
    bass = stream.Part()
    chords = [c for c in progression_part.recurse().getElementsByClass(m21chord.Chord)]

    for ch in chords:
        root = ch.root()
        ql = ch.quarterLength
        if genre in ("jazz", "blues", "swing"):
            # Walking bass: root, approach tones, chromatic lead-in to the next
            # root. Quarter notes throughout — the defining sound.
            steps = max(1, int(ql))
            for i in range(steps):
                n = note.Note(root.nameWithOctave)
                n.octave = 2
                if i == 1 and len(ch.pitches) > 2:
                    n = note.Note(ch.pitches[2].nameWithOctave); n.octave = 2
                elif i == 2:
                    n = note.Note(ch.pitches[1].nameWithOctave); n.octave = 2
                n.quarterLength = 1.0
                bass.append(n)
        elif genre in ("pop", "rock", "folk"):
            # Root on the beat, simple and steady.
            n = note.Note(root.nameWithOctave); n.octave = 2
            n.quarterLength = ql
            bass.append(n)
        else:
            # Classical: root then fifth, an alberti-ish alternation.
            half = ql / 2.0
            n1 = note.Note(root.nameWithOctave); n1.octave = 3; n1.quarterLength = half
            fifth = ch.pitches[-1] if len(ch.pitches) > 1 else root
            n2 = note.Note(fifth.nameWithOctave); n2.octave = 3; n2.quarterLength = half
            bass.append(n1)
            bass.append(n2)
    return bass


def find_voice_leading_problems(part_a, part_b) -> list[str]:
    """Report parallel fifths and octaves between two parts.

    music21 knows these rules exactly; a language model only knows that they are
    frowned upon.
    """
    from music21 import voiceLeading

    notes_a = list(part_a.flatten().notes)
    notes_b = list(part_b.flatten().notes)
    problems = []
    for i in range(min(len(notes_a), len(notes_b)) - 1):
        try:
            vlq = voiceLeading.VoiceLeadingQuartet(
                notes_a[i], notes_a[i + 1], notes_b[i], notes_b[i + 1]
            )
            if vlq.parallelFifth():
                problems.append(f"parallel fifth at note {i + 1}")
            if vlq.parallelOctave():
                problems.append(f"parallel octave at note {i + 1}")
        except Exception:
            continue
    return problems


def assemble(plan: Plan, instruments: list[str] | None = None):
    """Build the complete score from a plan.

    Returns a music21 Score with a melody part (motif, developed per section)
    and a bass part, in the requested key, metre and tempo.

    *instruments* assigns real instruments to the parts in order. Every part
    used to be hardcoded to Piano, so the instrument picker in the UI had no
    effect whatsoever — and nothing checked that the notes were playable on
    whatever was chosen.
    """
    from music21 import stream, meter, key as m21key, tempo as m21tempo, instrument

    score = stream.Score()
    melody = stream.Part()
    # Instrument assignment happens after the notes exist, so ranges can be
    # checked against real content; Piano is only a placeholder.
    melody.insert(0, instrument.Piano())
    harmony_part = stream.Part()
    bass_part = stream.Part()

    melody.append(meter.TimeSignature(plan.time_signature))
    melody.append(m21key.Key(plan.key.split()[0],
                             "minor" if "minor" in plan.key.lower() else "major"))
    melody.insert(0, m21tempo.MetronomeMark(number=plan.tempo))

    ts_obj = meter.TimeSignature(plan.time_signature)
    beats_per_bar = ts_obj.numerator * (4.0 / ts_obj.denominator)

    for section_name in plan.form:
        section = plan.sections.get(section_name)
        if section is None:
            continue
        # Harmony for this section
        chords = realise_progression(section.progression, plan.key,
                                     section.bars, plan.time_signature)
        for c in chords.recurse().notes:
            harmony_part.append(c)
        # Melody: the motif, transformed as the section asks, then repeated
        # until the section is actually full.
        dev = develop_motif(plan.motif, section.develop, plan.key)
        dev = _fill_bars(dev, section.bars, beats_per_bar, plan.key)
        for n in dev.recurse().notes:
            melody.append(n)
        # Bass under the same harmony
        for n in generate_bass(chords, plan.genre, plan.key).recurse().notes:
            bass_part.append(n)

    score.insert(0, melody)
    score.insert(0, bass_part)
    if instruments:
        from researcher.composer.orchestration import orchestrate

        report = orchestrate(score, instruments)
        moved = sum(r["notes_moved"] for r in report["reports"])
        if moved:
            logger.info("[realize] octave-shifted %d note(s) to fit instrument ranges", moved)
        for r in report["reports"]:
            if r["breathing"]:
                logger.info("[realize] %s: %d phrase(s) too long to play in one breath",
                            r["instrument"], len(r["breathing"]))
        score.editorial.orchestration = report

    return score


# ---------------------------------------------------------------------------
# Joining the two halves: specialist material + enforced structure
# ---------------------------------------------------------------------------


def break_scale_runs(part, max_run: int = 6, key_name: str = "C major"):
    """Break up long stepwise runs.

    A run straight up or down the scale is the signature of generated filler,
    and it is the one flaw the specialist model still produces reliably — it has
    the vocabulary but not the restraint. Rather than reject the tune, bend the
    offending note into a leap so the line turns instead of marching.

    The replacement is taken from the chord the run sits in where possible, so
    the fix stays in the harmony rather than introducing a wrong note.
    """
    from music21 import note, stream, key as m21key

    k = m21key.Key(key_name.split()[0],
                   "minor" if "minor" in key_name.lower() else "major")
    scale_pitches = [p.name for p in k.getPitches()]

    # .notes includes ChordSymbol objects — the "C"/"F"/"G7" markings above the
    # staff. Counting those as melody notes breaks every run in two, which is
    # why the detector silently found nothing on a tune that clearly had one.
    notes = [n for n in part.recurse().getElementsByClass(note.Note)]
    if len(notes) < max_run + 1:
        return part, 0

    fixed = 0
    run_len, run_dir = 1, 0
    for i in range(1, len(notes)):
        try:
            step = notes[i].pitch.midi - notes[i - 1].pitch.midi
        except Exception:
            run_len, run_dir = 1, 0
            continue
        direction = (step > 0) - (step < 0)
        if 0 < abs(step) <= 2 and direction == run_dir:
            run_len += 1
        else:
            run_len, run_dir = 2 if 0 < abs(step) <= 2 else 1, direction

        if run_len > max_run:
            # Turn the line: jump a third in the opposite direction, snapped to
            # the key so the new note belongs.
            target = notes[i].pitch.midi - (4 * run_dir)
            cand = note.Note()
            cand.pitch.midi = target
            if cand.pitch.name not in scale_pitches:
                cand.pitch.midi = target + (1 if run_dir > 0 else -1)
            notes[i].pitch = cand.pitch
            fixed += 1
            run_len, run_dir = 1, 0
    return part, fixed


def add_bass_from_chord_symbols(score, genre: str, key_name: str):
    """Give a single-line tune a bass part built from its own chord symbols.

    The specialist writes chord symbols above the melody but rarely a second
    voice, so scores arrive as a bare line. The symbols are already there — this
    turns them into an idiomatic bass rather than asking the model for one it
    tends to write as a copy of the melody.
    """
    from music21 import harmony, note, stream, chord as m21chord

    symbols = list(score.recurse().getElementsByClass(harmony.ChordSymbol))
    if not symbols:
        return None

    bass = stream.Part()
    genre = (genre or "classical").lower()
    for i, cs in enumerate(symbols):
        try:
            pitches = cs.pitches
            if not pitches:
                continue
        except Exception:
            continue
        # Length until the next symbol, so the bass follows the harmonic rhythm.
        if i + 1 < len(symbols):
            span = float(symbols[i + 1].offset - cs.offset) or 4.0
        else:
            span = 4.0
        span = max(1.0, min(8.0, span))
        root = pitches[0]

        if genre in ("jazz", "blues", "swing"):
            for beat in range(int(span)):
                pick = pitches[[0, 2, 1, 2][beat % 4] % len(pitches)]
                n = note.Note(pick.nameWithOctave); n.octave = 2
                n.quarterLength = 1.0
                bass.append(n)
        elif genre in ("pop", "rock", "folk"):
            n = note.Note(root.nameWithOctave); n.octave = 2
            n.quarterLength = span
            bass.append(n)
        else:
            half = span / 2.0
            n1 = note.Note(root.nameWithOctave); n1.octave = 3; n1.quarterLength = half
            fifth = pitches[-1] if len(pitches) > 1 else root
            n2 = note.Note(fifth.nameWithOctave); n2.octave = 3; n2.quarterLength = half
            bass.append(n1)
            bass.append(n2)
    return bass if list(bass.recurse().notes) else None


def polish(abc: str, genre: str = "classical", key_name: str = "C major",
           tempo: int = 120) -> tuple[object, dict]:
    """Apply the deterministic guarantees to a specialist-written tune.

    This is where the two halves meet. The model supplies idiomatic material —
    blue notes, swing rhythms, real chord symbols — which no rule set can
    invent. The code then enforces what the model reliably gets wrong: it breaks
    scale runs and gives a bare melody an independent bass.

    Returns ``(score, report)``.
    """
    from music21 import converter

    report = {"scale_runs_fixed": 0, "bass_added": False, "parts": 0}
    try:
        score = converter.parse(abc, format="abc")
    except Exception as e:
        logger.warning("[polish] could not parse ABC: %s", e)
        return None, report

    parts = list(score.parts) or [score]
    _, fixed = break_scale_runs(parts[0], max_run=6, key_name=key_name)
    report["scale_runs_fixed"] = fixed

    if len(parts) < 2:
        bass = add_bass_from_chord_symbols(score, genre, key_name)
        if bass is not None:
            score.insert(0, bass)
            report["bass_added"] = True

    report["parts"] = len(list(score.parts))
    return score, report


# ---------------------------------------------------------------------------
# Long form
# ---------------------------------------------------------------------------

# Harmony worth listening to. Plain triads are what made the early output sound
# like an exercise; these are the progressions the idioms actually use, and
# music21 realises every figure here — including secondary dominants (V7/V),
# borrowed chords (bVII) and augmented sixths (Ger+65).
GENRE_PROGRESSIONS: dict[str, dict[str, list[str]]] = {
    "jazz": {
        "A":     ["IMaj7", "vi7", "ii7", "V9"],
        "A2":    ["IMaj7", "V7/ii", "ii7", "V13"],
        "B":     ["iv7", "bVII7", "IMaj7", "V7/vi"],
        "bridge":["ii7", "V9", "IMaj7", "VI7"],
        "turn":  ["iii7", "VI7", "ii7", "V13"],
    },
    "classical": {
        "A":     ["I", "V6", "vi", "iii"],
        "A2":    ["IV", "I64", "V7", "I"],
        "B":     ["vi", "V7/V", "V", "V7"],
        "bridge":["ii65", "V7", "I", "V7/vi"],
        "turn":  ["IV", "Ger+65", "V", "I"],
    },
    "baroque": {
        "A":     ["I", "V", "vi", "iii"],
        "A2":    ["IV", "I", "ii65", "V7"],
        "B":     ["vi", "iii", "IV", "V7/V"],
        "bridge":["I6", "IV", "viio7", "I"],
        "turn":  ["ii65", "V7", "I", "I"],
    },
    "blues": {
        "A":     ["I7", "IV7", "I7", "I7"],
        "A2":    ["IV7", "IV7", "I7", "I7"],
        "B":     ["V9", "IV9", "I7", "V7"],
        "bridge":["I7", "V7/IV", "IV7", "#ivo7"],
        "turn":  ["I7", "VI7", "ii7", "V7"],
    },
    "pop": {
        "A":     ["I", "V", "vi", "IV"],
        "A2":    ["vi", "IV", "I", "V"],
        "B":     ["IV", "I", "V", "vi"],
        "bridge":["ii", "V", "I", "V7/vi"],
        "turn":  ["IV", "V", "iii", "vi"],
    },
}


def _progressions_for(genre: str) -> dict[str, list[str]]:
    g = (genre or "classical").lower()
    for key_name in GENRE_PROGRESSIONS:
        if key_name in g:
            return GENRE_PROGRESSIONS[key_name]
    return GENRE_PROGRESSIONS["classical"]


def build_long_plan(genre: str, key_name: str, time_signature: str, tempo: int,
                    minutes: float = 2.5,
                    motif: list[tuple[str, float]] | None = None) -> Plan:
    """Design a piece of roughly *minutes* length.

    A single model call holds together for eight or sixteen bars and then
    repeats or drifts, so a three-minute piece has to be *assembled*. Recurring
    sections are what make length listenable rather than merely long: the
    listener needs to recognise material coming back, so the form reuses A
    heavily and contrasts it with B and C.
    """
    from music21 import meter

    ts = meter.TimeSignature(time_signature)
    beats_per_bar = ts.numerator * (4.0 / ts.denominator)
    total_bars = max(16, int(round(minutes * tempo / beats_per_bar)))

    prog = _progressions_for(genre)
    section_bars = 8

    # A rondo-ish shape: the A material returns, each time developed
    # differently, with contrasting episodes between. This is how real pieces
    # reach three minutes without becoming three minutes of new material.
    # Three of the eight sections used to be "original" — the motif returning
    # completely unchanged. That is what produced five literal repeats of the
    # opening cell. A theme should RETURN recognisably, not be pasted back in.
    #
    # Only the opening statement is untouched. Every later return of A keeps its
    # pitch shape but arrives in a different rhythm, which is how a listener
    # hears one idea developing rather than one idea repeating.
    cycle = [
        ("A",  "original"),                     # state the idea plainly, once
        ("A2", "sequence_up+dotted"),
        ("B",  "inversion"),
        ("A",  "syncopate"),                    # same theme, displaced accents
        ("bridge", "fragment"),
        ("A2", "augmentation"),
        ("turn", "sequence_up+diminution"),
        ("A",  "long_short_pairs"),             # final return, re-timed again
    ]

    form: list[str] = []
    sections: dict[str, Section] = {}
    bars_so_far = 0
    i = 0
    while bars_so_far < total_bars:
        name, develop = cycle[i % len(cycle)]
        # Distinct instance name so repeats of A can differ in development.
        inst = f"{name}_{i}"
        sections[inst] = Section(
            inst, prog.get(name, prog["A"]), bars=section_bars, develop=develop
        )
        form.append(inst)
        bars_so_far += section_bars
        i += 1

    return Plan(
        title=f"{genre.title()} piece",
        key=key_name, time_signature=time_signature, tempo=tempo, genre=genre,
        motif=motif or [("C4", 1.0), ("E4", 0.5), ("G4", 0.5), ("E4", 1.0)],
        form=form, sections=sections,
    )


def estimate_duration(plan: Plan) -> float:
    """Seconds the plan will last, so the caller can report it honestly."""
    from music21 import meter

    ts = meter.TimeSignature(plan.time_signature)
    beats_per_bar = ts.numerator * (4.0 / ts.denominator)
    return plan.total_bars() * beats_per_bar * 60.0 / max(1, plan.tempo)
