"""Composer-specific CrewAI tools.

These tools return real reference data so the LLM agent can use them
as lookup tables rather than re-prompting itself.
"""

import json
import logging

from crewai.tools import tool

logger = logging.getLogger(__name__)

# ── Static data ──────────────────────────────────────────────────────

_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_FLAT_NOTES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

_SCALE_PATTERNS = {
    'major':            [2, 2, 1, 2, 2, 2, 1],
    'natural minor':    [2, 1, 2, 2, 1, 2, 2],
    'harmonic minor':   [2, 1, 2, 2, 1, 3, 1],
    'melodic minor':    [2, 1, 2, 2, 2, 2, 1],
    'dorian':           [2, 1, 2, 2, 2, 1, 2],
    'phrygian':         [1, 2, 2, 2, 1, 2, 2],
    'lydian':           [2, 2, 2, 1, 2, 2, 1],
    'mixolydian':       [2, 2, 1, 2, 2, 1, 2],
    'aeolian':          [2, 1, 2, 2, 1, 2, 2],
    'locrian':          [1, 2, 2, 1, 2, 2, 2],
    'pentatonic major': [2, 2, 3, 2, 3],
    'pentatonic minor': [3, 2, 2, 3, 2],
    'blues':            [3, 2, 1, 1, 3, 2],
    'whole tone':       [2, 2, 2, 2, 2, 2],
    'chromatic':        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

_CHORD_QUALITIES = {
    'major':  [0, 4, 7],
    'minor':  [0, 3, 7],
    'dim':    [0, 3, 6],
    'aug':    [0, 4, 8],
    'dom7':   [0, 4, 7, 10],
    'maj7':   [0, 4, 7, 11],
    'min7':   [0, 3, 7, 10],
    'dim7':   [0, 3, 6, 9],
    'hdim7':  [0, 3, 6, 10],
    'sus2':   [0, 2, 7],
    'sus4':   [0, 5, 7],
}

_DIATONIC_QUALITY = {
    'major': ['major', 'minor', 'minor', 'major', 'major', 'minor', 'dim'],
    'natural minor': ['minor', 'dim', 'major', 'minor', 'minor', 'major', 'major'],
}

_ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']

_KEY_FIFTHS = {
    'C major': 0, 'G major': 1, 'D major': 2, 'A major': 3, 'E major': 4,
    'B major': 5, 'F# major': 6, 'Gb major': -6, 'F major': -1,
    'Bb major': -2, 'Eb major': -3, 'Ab major': -4, 'Db major': -5,
    'Cb major': -7,
    'A minor': 0, 'E minor': 1, 'B minor': 2, 'F# minor': 3,
    'C# minor': 4, 'G# minor': 5, 'D minor': -1, 'G minor': -2,
    'C minor': -3, 'F minor': -4, 'Bb minor': -5, 'Eb minor': -6,
}

# Instrument data: (lowest, highest, clef, transposition, MIDI program)
_INSTRUMENTS = {
    'piano':            ('A0', 'C8', 'treble+bass', None, 1),
    'violin':           ('G3', 'E7', 'treble', None, 41),
    'viola':            ('C3', 'A6', 'alto', None, 42),
    'cello':            ('C2', 'A5', 'bass', None, 43),
    'double bass':      ('E1', 'G4', 'bass', 'sounds octave lower', 44),
    'contrabass':       ('E1', 'G4', 'bass', 'sounds octave lower', 44),
    'flute':            ('C4', 'D7', 'treble', None, 74),
    'oboe':             ('Bb3', 'A6', 'treble', None, 69),
    'clarinet':         ('D3', 'Bb6', 'treble', 'Bb: sounds M2 lower', 72),
    'clarinet in bb':   ('D3', 'Bb6', 'treble', 'sounds M2 lower', 72),
    'bassoon':          ('Bb1', 'Eb5', 'bass', None, 71),
    'french horn':      ('B1', 'F5', 'treble', 'F: sounds P5 lower', 61),
    'horn in f':        ('B1', 'F5', 'treble', 'sounds P5 lower', 61),
    'trumpet':          ('F#3', 'D6', 'treble', 'Bb: sounds M2 lower', 57),
    'trumpet in bb':    ('F#3', 'D6', 'treble', 'sounds M2 lower', 57),
    'trombone':         ('E2', 'Bb5', 'bass', None, 58),
    'tuba':             ('D1', 'F4', 'bass', None, 59),
    'alto saxophone':   ('Db3', 'A5', 'treble', 'Eb: sounds M6 lower', 66),
    'tenor saxophone':  ('Ab2', 'E5', 'treble', 'Bb: sounds M9 lower', 67),
    'soprano saxophone':('Ab3', 'E6', 'treble', 'Bb: sounds M2 lower', 65),
    'baritone saxophone':('Db2', 'A4', 'treble', 'Eb: sounds M13 lower', 68),
    'guitar':           ('E2', 'B5', 'treble', 'sounds octave lower', 25),
    'acoustic guitar':  ('E2', 'B5', 'treble', 'sounds octave lower', 26),
    'electric guitar':  ('E2', 'B5', 'treble', 'sounds octave lower', 28),
    'bass guitar':      ('E1', 'G4', 'bass', 'sounds octave lower', 34),
    'harp':             ('Cb1', 'G#7', 'treble+bass', None, 47),
    'xylophone':        ('F4', 'C8', 'treble', 'sounds octave higher', 14),
    'marimba':          ('C2', 'C7', 'treble+bass', None, 13),
    'timpani':          ('D2', 'C4', 'bass', None, 48),
    'voice':            ('C3', 'C6', 'treble', None, 53),
    'soprano':          ('C4', 'C6', 'treble', None, 53),
    'alto':             ('F3', 'F5', 'treble', None, 53),
    'tenor':            ('C3', 'C5', 'treble', 'sounds octave lower', 53),
    'bass voice':       ('E2', 'E4', 'bass', None, 53),
    'baritone':         ('A2', 'A4', 'bass', None, 53),
}

_PROGRESSIONS = {
    'pop':         ['I', 'V', 'vi', 'IV'],
    'jazz':        ['IIm7', 'V7', 'Imaj7', 'VIm7'],
    'blues':       ['I7', 'I7', 'I7', 'I7', 'IV7', 'IV7', 'I7', 'I7', 'V7', 'IV7', 'I7', 'V7'],
    'classical':   ['I', 'IV', 'V', 'I'],
    'romantic':    ['I', 'vi', 'IV', 'V'],
    'folk':        ['I', 'IV', 'I', 'V'],
    'rock':        ['I', 'bVII', 'IV', 'I'],
    'lullaby':     ['I', 'IV', 'V', 'I', 'I', 'vi', 'IV', 'V'],
    'waltz':       ['I', 'IV', 'V7', 'I'],
    'ballad':      ['I', 'vi', 'IV', 'V', 'I', 'iii', 'IV', 'V'],
    'melancholic': ['i', 'VI', 'III', 'VII'],
    'uplifting':   ['I', 'IV', 'vi', 'V'],
}


def _note_index(note: str) -> int:
    """Return semitone index (0-11) for a note name."""
    n = note.strip().upper()
    for i, name in enumerate(_NOTES):
        if n == name:
            return i
    for i, name in enumerate(_FLAT_NOTES):
        if n == name.upper():
            return i
    return 0


def _build_scale(root: str, pattern: list[int]) -> list[str]:
    """Build scale notes from root and interval pattern."""
    idx = _note_index(root)
    use_flats = 'b' in root or root in ('F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb')
    names = _FLAT_NOTES if use_flats else _NOTES
    notes = [names[idx % 12]]
    for step in pattern:
        idx += step
        notes.append(names[idx % 12])
    return notes


# ── Tools ────────────────────────────────────────────────────────────

@tool("ScaleReference")
def scale_reference(scale_name: str) -> str:
    """Get the notes and intervals for a scale (e.g. 'C major', 'A harmonic minor', 'D dorian').
    Returns the actual notes, interval pattern, and diatonic chords."""
    parts = scale_name.strip().split(None, 1)
    root = parts[0] if parts else 'C'
    mode = parts[1].lower() if len(parts) > 1 else 'major'
    mode = mode.replace('-', ' ')

    pattern = _SCALE_PATTERNS.get(mode, _SCALE_PATTERNS['major'])
    notes = _build_scale(root, pattern)
    interval_str = ' '.join(['W' if s == 2 else 'H' if s == 1 else str(s) for s in pattern])

    lines = [
        f"Scale: {root} {mode}",
        f"Notes: {' '.join(notes[:-1])}",
        f"Intervals: {interval_str}",
    ]

    base = 'major' if mode in ('major', 'lydian', 'mixolydian') else 'natural minor'
    if base in _DIATONIC_QUALITY and len(notes) >= 8:
        chords = []
        for i, q in enumerate(_DIATONIC_QUALITY[base]):
            roman = _ROMAN[i]
            if q == 'minor':
                roman = roman.lower()
            elif q == 'dim':
                roman = roman.lower() + '°'
            chords.append(f"{roman}({notes[i]}{'' if q == 'major' else 'm' if q == 'minor' else 'dim'})")
        lines.append(f"Diatonic chords: {', '.join(chords)}")

    return '\n'.join(lines)


@tool("InstrumentRange")
def instrument_range(instrument: str) -> str:
    """Get the range, clef, transposition, and MIDI program for an instrument.
    Input: instrument name (e.g. 'piano', 'violin', 'trumpet in Bb').
    Returns concrete data: range, clef, transposition, MIDI program number."""
    key = instrument.strip().lower()
    data = _INSTRUMENTS.get(key)
    if not data:
        # Try partial match
        for k, v in _INSTRUMENTS.items():
            if key in k or k in key:
                data = v
                key = k
                break
    if not data:
        return f"Unknown instrument: {instrument}. Available: {', '.join(sorted(_INSTRUMENTS.keys()))}"

    low, high, clef, transp, midi = data
    lines = [
        f"Instrument: {key.title()}",
        f"Range: {low} to {high}",
        f"Clef: {clef}",
        f"Transposition: {transp or 'Concert pitch (non-transposing)'}",
        f"MIDI program: {midi}",
    ]
    return '\n'.join(lines)


@tool("ChordProgressionBuilder")
def chord_progression_builder(key: str = "C major", style: str = "classical",
                              bars: str = "8", feel: str = "") -> str:
    """Build a chord progression for a given key and style.
    Args:
        key: The key, e.g. 'C major', 'A minor', 'G major'.
        style: Style of progression, e.g. 'classical', 'jazz', 'pop', 'blues', 'romantic', 'lullaby'.
        bars: Number of bars (default 8).
        feel: Optional feel modifier, e.g. 'gentle', 'dramatic'.
    Returns chord symbols and Roman numeral analysis."""
    key_str = key.strip() if key else 'C major'
    style = style.strip().lower() if style else 'classical'
    feel = feel.strip().lower() if feel else ''
    try:
        bars = int(bars)
    except (ValueError, TypeError):
        bars = 8

    # Pick progression template
    prog = _PROGRESSIONS.get(style) or _PROGRESSIONS.get(feel) or _PROGRESSIONS['classical']

    # Extend/trim to requested bar count
    full = []
    while len(full) < bars:
        full.extend(prog)
    full = full[:bars]

    # Resolve root notes
    parts = key_str.strip().split(None, 1)
    root = parts[0] if parts else 'C'
    mode = parts[1].lower() if len(parts) > 1 else 'major'
    scale = _build_scale(root, _SCALE_PATTERNS.get(mode, _SCALE_PATTERNS['major']))

    # Map Roman numerals to scale degree indices
    _roman_to_degree = {
        'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6,
        'i': 0, 'ii': 1, 'iii': 2, 'iv': 3, 'v': 4, 'vi': 5, 'vii': 6,
    }

    def _resolve_chord(roman_str, scale_notes):
        """Resolve a Roman numeral like 'vi', 'IIm7', 'V7', 'bVII' to a chord name."""
        r = roman_str.strip()
        # Handle flats (bVII -> VII, degree shifted)
        flat = r.startswith('b')
        if flat:
            r = r[1:]
        # Split base Roman from suffix (e.g., 'IIm7' -> 'II', 'm7'; 'V7' -> 'V', '7')
        base = ''
        suffix = ''
        for ch in r:
            if ch.upper() in 'IV':
                base += ch
            else:
                suffix = r[len(base):]
                break
        if not base:
            return roman_str  # Can't parse
        deg = _roman_to_degree.get(base)
        if deg is None:
            return roman_str
        if deg < len(scale_notes) - 1:
            note = scale_notes[deg]
        else:
            note = scale_notes[deg % (len(scale_notes) - 1)]
        if flat:
            # Flatten by one semitone
            idx = _note_index(note)
            use_flats = 'b' in root or root in ('F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb')
            names = _FLAT_NOTES if use_flats else _NOTES
            note = names[(idx - 1) % 12]
        # Add quality suffix
        if base == base.lower() and not suffix:  # lowercase = minor
            suffix = 'm'
        return note + suffix

    lines = [
        f"Key: {key_str} | Style: {style} | Bars: {bars}",
        f"Scale: {' '.join(scale[:-1])}",
        "",
        "Bar | Roman | Chord",
        "----|-------|------",
    ]
    for i, roman in enumerate(full):
        chord = _resolve_chord(roman, scale)
        lines.append(f"  {i+1} | {roman:6s} | {chord}")

    lines.append("")
    lines.append(f"Progression: {' - '.join(full)}")

    return '\n'.join(lines)


@tool("JSONScoreTemplate")
def json_score_template(title: str = "Untitled", composer: str = "AI Composer",
                        instruments: str = "Piano", key: str = "C major",
                        time_signature: str = "4/4", tempo: str = "120") -> str:
    """Generate a JSON score template for given instruments.
    Args:
        title: Title of the piece.
        composer: Composer name.
        instruments: Comma-separated instrument names, e.g. 'Piano' or 'Violin, Cello'.
        key: Key signature, e.g. 'C major', 'G major', 'A minor'.
        time_signature: Time signature, e.g. '4/4', '3/4', '6/8'.
        tempo: Tempo in BPM (default 120).
    Returns a JSON score skeleton with parts ready to have notes added."""
    instruments_str = instruments.strip() if instruments else 'Piano'
    if instruments_str.startswith('['):
        try:
            instrument_list = json.loads(instruments_str)
        except (json.JSONDecodeError, TypeError):
            instrument_list = [instruments_str.strip('[] "\'')]
    else:
        instrument_list = [x.strip() for x in instruments_str.split(',') if x.strip()]
    if not instrument_list:
        instrument_list = ['Piano']

    parts = []
    for inst_name in instrument_list:
        inst_key = inst_name.strip().lower()
        inst_data = _INSTRUMENTS.get(inst_key)
        is_grand_staff = inst_data and inst_data[2] == 'treble+bass'

        if is_grand_staff:
            # Grand staff: ONE part with both staff 1 and staff 2 notes in each measure
            parts.append({
                "name": inst_name,
                "measures": [
                    {
                        "notes": [
                            {"pitch": "C5", "duration": "quarter", "staff": 1},
                            {"pitch": "E5", "duration": "quarter", "staff": 1},
                            {"pitch": "G4", "duration": "half", "staff": 1},
                            {"pitch": "C3", "duration": "half", "staff": 2},
                            {"pitch": "G3", "duration": "half", "staff": 2},
                        ]
                    },
                ],
            })
        else:
            parts.append({
                "name": inst_name,
                "measures": [
                    {"notes": [{"pitch": "C4", "duration": "quarter"}]},
                ],
            })

    template = {
        "title": title,
        "composer": composer,
        "parts": parts,
    }

    result = json.dumps(template, indent=2)
    result += f"\n\nKey: {key} | Time: {time_signature} | Tempo: {tempo} BPM"
    result += "\n\nReplace the placeholder notes with your composed music."
    result += "\nFor Piano/Harp: put BOTH staff 1 (treble) and staff 2 (bass) notes in the SAME measure."
    result += "\nDuration values: whole, half, dotted half, quarter, dotted quarter, eighth, 16th"
    result += '\nPitch format: "C4", "F#5", "Bb3", "rest"'
    result += "\n\nIMPORTANT: Do NOT call this tool again. Use it as a starting template, "
    result += "fill in your composed music, and output the full JSON as your Final Answer."
    return result


# All tools exported for use by the ComposerCrew
COMPOSER_TOOLS = [
    scale_reference,
    instrument_range,
    chord_progression_builder,
    json_score_template,
]
