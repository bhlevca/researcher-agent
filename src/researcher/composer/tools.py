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
    # ── Dropdown genres (must match composer.html <select> values) ──
    'classical':      ['I', 'IV', 'V', 'I', 'vi', 'IV', 'V', 'I'],
    'jazz':           ['IIm7', 'V7', 'Imaj7', 'VIm7', 'IIm7', 'V7', 'IIIm7', 'VIm7'],
    'pop':            ['I', 'V', 'vi', 'IV'],
    'rock':           ['I', 'bVII', 'IV', 'I'],
    'film':           ['I', 'iii', 'vi', 'IV', 'I', 'V', 'vi', 'IV'],
    'folk':           ['I', 'IV', 'I', 'V', 'I', 'IV', 'V', 'I'],
    'blues':          ['I7', 'I7', 'I7', 'I7', 'IV7', 'IV7', 'I7', 'I7', 'V7', 'IV7', 'I7', 'V7'],
    'latin':          ['I', 'IV', 'V', 'V', 'I', 'IV', 'ii', 'V'],
    'electronic':     ['i', 'VI', 'III', 'VII', 'i', 'VI', 'III', 'VII'],
    'baroque':        ['I', 'V', 'vi', 'iii', 'IV', 'I', 'IV', 'V'],
    'romantic':       ['I', 'vi', 'ii', 'V', 'I', 'IV', 'V', 'vi'],
    'impressionist':  ['Imaj7', 'IIm7', 'bVII', 'IV', 'Imaj7', 'vi', 'ii', 'V'],
    'minimalist':     ['I', 'V', 'I', 'V', 'IV', 'I', 'IV', 'I'],
    'ragtime':        ['I', 'I', 'IV', 'IV', 'I', 'V7', 'I', 'V7'],
    'bossa_nova':     ['Imaj7', 'IIm7', 'IIm7', 'V7', 'Imaj7', 'VIm7', 'IIm7', 'V7'],
    # ── Extra styles (feel / free-form requests) ──
    'lullaby':        ['I', 'vi', 'IV', 'V', 'I', 'iii', 'IV', 'V',
                       'vi', 'IV', 'ii', 'V', 'I', 'IV', 'V', 'I'],
    'waltz':          ['I', 'IV', 'V7', 'I', 'I', 'ii', 'V', 'I'],
    'ballad':         ['I', 'vi', 'IV', 'V', 'I', 'iii', 'IV', 'V'],
    'melancholic':    ['i', 'VI', 'III', 'VII', 'i', 'iv', 'V', 'i'],
    'uplifting':      ['I', 'IV', 'vi', 'V'],
    'hymn':           ['I', 'IV', 'I', 'V', 'I', 'IV', 'V', 'I'],
    'march':          ['I', 'I', 'IV', 'V', 'I', 'I', 'V', 'I'],
    'minuet':         ['I', 'V', 'I', 'IV', 'V', 'I', 'ii', 'V'],
}

# Style-specific composition guidelines (appended to CompositionPrep output)
_STYLE_GUIDELINES = {
    'classical':     "Balanced 4+4 phrases. Alberti bass or broken chords. Clear cadences: HC at phrase midpoint, PAC at end. Sparse ornamentation (trills, turns).",
    'jazz':          "Swing eighth feel. Use 7th/9th chord extensions. Walking bass in steady quarters. Chromatic approach tones. ii-V-I voice leading.",
    'pop':           "Strong hook melody, repetitive motifs. Simple bass (root on beat 1). Clear verse/chorus structure. Syncopated rhythms.",
    'rock':          "Power chord roots in bass. Driving eighth-note rhythm. Pentatonic melody. Strong backbeat (beats 2 & 4).",
    'film':          "Wide dynamic range. Lush sustained chords. Sweeping melodic arcs. Use octave doubling for climax. Modal interchange for color.",
    'folk':          "Diatonic melody, stepwise with small leaps. Simple bass (root-fifth). Repetitive strophic form. Singable range (one octave).",
    'blues':         "Shuffle/swing feel. Blue notes (b3, b5, b7). Call-and-response phrasing. Bend-like grace notes. 12-bar structure.",
    'latin':         "Syncopated montuno rhythm. Bass anticipates beat 1 (plays on 'and' of 4). Clave-based phrasing. Bright upper-register melody.",
    'electronic':    "Repetitive ostinato patterns. Arpeggiated chords. Build and drop dynamics. Layered rhythmic cells. Minimal melodic variation.",
    'baroque':       "Contrapuntal motion between voices. Sequences (repeated patterns at different pitch levels). Running sixteenths. Terraced dynamics.",
    'romantic':      "Expressive rubato-style phrasing. Rich chromaticism. Wide melodic leaps for drama. Thick chordal textures. Gradual dynamic swells.",
    'impressionist': "Parallel chord motion. Whole-tone and pentatonic fragments. Avoid strong cadences. Sustained pedal tones. Coloristic register changes.",
    'minimalist':    "Gradual additive process. Phase-shifting patterns. Steady pulse. Subtle variations over many repetitions. Limited pitch set.",
    'ragtime':       "Syncopated treble over steady oom-pah bass (root on 1&3, chord on 2&4). 16-bar sections. March-like left hand. Jaunty dotted rhythms.",
    'bossa_nova':    "Gentle syncopation. Chord tones in melody. Soft bass (root-fifth-octave). Relaxed 2-feel. Smooth voice leading between 7th chords.",
    # Extra styles
    'lullaby':       "Gentle 3/4 or 6/8 feel. Rocking bass (root-fifth-root). Narrow melodic range (one octave). Soft dynamics (p-mp). Stepwise melody.",
    'waltz':         "Strong beat 1, lighter beats 2-3. Bass note on 1, chord on 2-3. Graceful melodic turns. Moderate tempo.",
    'ballad':        "Slow expressive melody. Sustained chords. Arpeggiated accompaniment. Build to emotional climax then resolve quietly.",
    'melancholic':   "Minor key. Descending melodic lines. Suspended chords resolving slowly. Sparse texture. Sighing motifs (falling 2nds).",
    'uplifting':     "Major key. Rising melodic contour. Bright register. Increasing rhythmic energy. Strong authentic cadences.",
    'hymn':          "Four-part chorale style. Homophonic texture. One chord per beat. Smooth voice leading. Bass in root position mostly.",
    'march':         "Strong duple meter. Dotted rhythms. Brass-like melody in middle register. Bass on beats 1 and 3. Crisp articulation.",
    'minuet':        "Elegant 3/4. Moderate tempo. Graceful ornaments. Binary form (A-B). Light texture. Courtly melodic style.",
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

def _get_scale_info(scale_name: str) -> list[str]:
    """Return scale info lines for a given scale name like 'C major'."""
    parts = scale_name.strip().split(None, 1)
    root = parts[0] if parts else 'C'
    mode = parts[1].lower() if len(parts) > 1 else 'major'
    mode = mode.replace('-', ' ')

    pattern = _SCALE_PATTERNS.get(mode, _SCALE_PATTERNS['major'])
    notes = _build_scale(root, pattern)

    lines = [
        f"Scale: {root} {mode}",
        f"Notes: {' '.join(notes[:-1])}",
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

    return lines


def _get_instrument_info(instrument: str) -> list[str]:
    """Return instrument info lines."""
    key = instrument.strip().lower()
    data = _INSTRUMENTS.get(key)
    if not data:
        for k, v in _INSTRUMENTS.items():
            if key in k or k in key:
                data = v
                key = k
                break
    if not data:
        return [f"Unknown instrument: {instrument}"]
    low, high, clef, transp, midi = data
    return [f"{key.title()}: range {low}-{high}, clef {clef}, MIDI {midi}"]


# Map Roman numerals to scale degree indices
_ROMAN_TO_DEGREE = {
    'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6,
    'i': 0, 'ii': 1, 'iii': 2, 'iv': 3, 'v': 4, 'vi': 5, 'vii': 6,
}


def _resolve_chord(roman_str: str, scale_notes: list[str], root: str) -> str:
    """Resolve a Roman numeral like 'vi', 'IIm7', 'V7', 'bVII' to a chord name."""
    r = roman_str.strip()
    flat = r.startswith('b')
    if flat:
        r = r[1:]
    base = ''
    suffix = ''
    for ch in r:
        if ch.upper() in 'IV':
            base += ch
        else:
            suffix = r[len(base):]
            break
    if not base:
        return roman_str
    deg = _ROMAN_TO_DEGREE.get(base)
    if deg is None:
        return roman_str
    note = scale_notes[deg % (len(scale_notes) - 1)]
    if flat:
        idx = _note_index(note)
        use_flats = 'b' in root or root in ('F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb')
        names = _FLAT_NOTES if use_flats else _NOTES
        note = names[(idx - 1) % 12]
    if base == base.lower() and not suffix:
        suffix = 'm'
    return note + suffix


def _build_progression(key_str: str, style: str, bars: int, feel: str) -> list[str]:
    """Build chord progression lines for given key/style/bars."""
    prog = _PROGRESSIONS.get(style) or _PROGRESSIONS.get(feel) or _PROGRESSIONS['classical']
    full = []
    while len(full) < bars:
        full.extend(prog)
    full = full[:bars]

    parts = key_str.strip().split(None, 1)
    root = parts[0] if parts else 'C'
    mode = parts[1].lower() if len(parts) > 1 else 'major'
    scale = _build_scale(root, _SCALE_PATTERNS.get(mode, _SCALE_PATTERNS['major']))

    chords = [_resolve_chord(r, scale, root) for r in full]
    bar_items = [f"{i+1}:{chords[i]}" for i in range(bars)]
    lines = [
        f"Key: {key_str} | Style: {style} | Bars: {bars}",
        f"Chords: {' | '.join(bar_items)}",
    ]
    return lines


def _build_skeleton(title: str, instrument_list: list[str],
                    key: str, time_signature: str, tempo: str) -> str:
    """Build a JSON score skeleton string for the given instruments."""
    parts = []
    for inst_name in instrument_list:
        inst_key = inst_name.strip().lower()
        inst_data = _INSTRUMENTS.get(inst_key)
        is_grand_staff = inst_data and inst_data[2] == 'treble+bass'

        if is_grand_staff:
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

    template = {"title": title, "composer": "AI Composer", "parts": parts}
    result = json.dumps(template, indent=2)
    result += f"\n\nKey: {key} | Time: {time_signature} | Tempo: {tempo} BPM"
    return result


@tool("CompositionPrep")
def composition_prep(key: str = "C major", style: str = "classical",
                     bars: str = "16", instruments: str = "Piano",
                     feel: str = "", title: str = "Untitled",
                     time_signature: str = "4/4", tempo: str = "120") -> str:
    """Get scale, chords, instrument ranges, theory, AND JSON skeleton for composing.
    This is the ONLY tool you need. Call it ONCE, then write Final Answer.
    Args:
        key: Key signature, e.g. 'C major', 'A minor'.
        style: Style, e.g. 'classical', 'jazz', 'pop', 'lullaby', 'romantic'.
        bars: Number of bars (default 16).
        instruments: Comma-separated instrument names, e.g. 'Piano'.
        feel: Optional feel modifier, e.g. 'gentle', 'dramatic'.
        title: Title of the piece.
        time_signature: Time signature, e.g. '4/4', '3/4', '6/8'.
        tempo: Tempo in BPM (default 120).
    Returns scale, chords, guidelines, and a JSON skeleton to fill."""
    key_str = key.strip() if key else 'C major'
    style = style.strip().lower() if style else 'classical'
    feel = feel.strip().lower() if feel else ''
    try:
        num_bars = int(bars)
    except (ValueError, TypeError):
        num_bars = 16

    inst_list = [x.strip() for x in instruments.split(',') if x.strip()] if instruments else ['Piano']

    # 1. Scale
    sections = _get_scale_info(key_str)
    sections.append("")

    # 2. Instruments
    for inst in inst_list:
        sections.extend(_get_instrument_info(inst))
    sections.append("")

    # 3. Chord progression
    sections.extend(_build_progression(key_str, style, num_bars, feel))
    sections.append("")

    # 4. Music theory composition guidelines
    sections.append("=== COMPOSITION GUIDELINES ===")
    guide = _STYLE_GUIDELINES.get(style) or _STYLE_GUIDELINES.get(feel) or _STYLE_GUIDELINES['classical']
    sections.append(f"Style ({style}): {guide}")
    sections.append("Melody: stepwise motion (2nds/3rds) mostly, leaps (4ths+) for emphasis.")
    sections.append("Phrases: 4-bar phrases. Antecedent (bars 1-4) + consequent (bars 5-8).")
    sections.append("Voice leading: move each voice to nearest chord tone. Avoid parallel 5ths/octaves.")
    sections.append("Cadences: half cadence (->V) at phrase midpoint, authentic (V->I) at phrase end.")
    sections.append("")

    # 5. JSON skeleton
    sections.append("=== JSON SKELETON ===")
    skeleton = _build_skeleton(title, inst_list, key_str, time_signature, tempo)
    sections.append(skeleton)
    sections.append("")
    sections.append("=== YOUR NEXT STEP ===")
    sections.append("DO NOT call any tool. Write 'Final Answer:' followed by the complete JSON score.")
    sections.append("Replace the example measure with ALL your composed measures.")
    sections.append("Piano/Harp: staff 1 (treble) + staff 2 (bass) notes in SAME measure.")
    sections.append("Durations: whole, half, dotted half, quarter, dotted quarter, eighth, 16th.")
    sections.append('Pitch: "C4", "F#5", "Bb3", "rest". Fill every measure to match the time signature.')

    return '\n'.join(sections)


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
    result += "\n\n=== YOUR NEXT STEP ==="
    result += "\nDO NOT call any more tools. Write 'Final Answer:' followed by the complete JSON score."
    result += "\nReplace the example measure above with ALL your composed measures."
    result += "\nPiano/Harp: staff 1 (treble) + staff 2 (bass) notes in SAME measure."
    result += "\nDurations: whole, half, dotted half, quarter, dotted quarter, eighth, 16th."
    result += '\nPitch: "C4", "F#5", "Bb3", "rest". Fill every measure to match the time signature.'
    return result


# All tools exported for use by the ComposerCrew
COMPOSER_TOOLS = [
    composition_prep,
]
