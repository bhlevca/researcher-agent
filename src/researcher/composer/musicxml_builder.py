"""MusicXML Builder — Converts a simple JSON score to valid MusicXML 4.0.

The LLM outputs a JSON structure describing notes, and this module
builds syntactically perfect MusicXML that opens in MuseScore 4.

JSON schema the LLM must produce:
{
  "title": "My Piece",
  "composer": "AI Composer",
  "parts": [
    {
      "name": "Piano",
      "measures": [
        {
          "notes": [
            {"pitch": "C4", "duration": "quarter"},
            {"pitch": "E4", "duration": "quarter"},
            {"pitch": "G4", "duration": "half"},
            {"pitch": "rest", "duration": "half"},
            ...
          ]
        }
      ]
    }
  ]
}

For grand-staff instruments (piano, harp), each measure has two note
lists: "treble" and "bass" (or "staff1"/"staff2"):
{
  "notes": [
    {"pitch": "C5", "duration": "quarter", "staff": 1},
    {"pitch": "E5", "duration": "quarter", "staff": 1},
    {"pitch": "C3", "duration": "half", "staff": 2},
    {"pitch": "G3", "duration": "half", "staff": 2}
  ]
}
"""

import re
import logging
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Known grand-staff instruments
_GRAND_STAFF = {'piano', 'harp', 'marimba', 'organ', 'harpsichord',
                'celesta', 'accordion'}

# Duration name → divisions (with divisions=4: whole=16, half=8, quarter=4, etc.)
_DIVISIONS = 4  # subdivisions per quarter note

_DUR_DIVS = {
    'whole': 16,
    'half': 8,
    'dotted half': 12,
    'quarter': 4,
    'dotted quarter': 6,
    'eighth': 2,
    'dotted eighth': 3,
    '16th': 1,
    'sixteenth': 1,
}

_DUR_TYPE = {
    'whole': 'whole',
    'half': 'half',
    'dotted half': 'half',
    'quarter': 'quarter',
    'dotted quarter': 'quarter',
    'eighth': 'eighth',
    'dotted eighth': 'eighth',
    '16th': '16th',
    'sixteenth': '16th',
}

_DOTTED = {'dotted half', 'dotted quarter', 'dotted eighth'}

# Common LLM abbreviations → canonical names
_DUR_ALIASES = {
    'w': 'whole', 'h': 'half', 'q': 'quarter', 'e': 'eighth',
    'dh': 'dotted half', 'dq': 'dotted quarter', 'de': 'dotted eighth',
    's': '16th',
}


def _normalize_duration(dur: str) -> str:
    """Normalize duration string: handle abbreviations and case."""
    d = dur.strip().lower()
    return _DUR_ALIASES.get(d, d)

# Key signature fifths lookup
_KEY_FIFTHS = {
    'C major': 0, 'G major': 1, 'D major': 2, 'A major': 3, 'E major': 4,
    'B major': 5, 'F# major': 6, 'Gb major': -6, 'F major': -1,
    'Bb major': -2, 'Eb major': -3, 'Ab major': -4, 'Db major': -5,
    'Cb major': -7,
    'A minor': 0, 'E minor': 1, 'B minor': 2, 'F# minor': 3,
    'C# minor': 4, 'G# minor': 5, 'D minor': -1, 'G minor': -2,
    'C minor': -3, 'F minor': -4, 'Bb minor': -5, 'Eb minor': -6,
}

# MIDI program numbers for common instruments
_MIDI_PROGRAMS = {
    'piano': 1, 'acoustic grand piano': 1, 'bright acoustic piano': 2,
    'electric piano': 5, 'harpsichord': 7, 'celesta': 9,
    'organ': 20, 'accordion': 22, 'harmonica': 23,
    'acoustic guitar': 26, 'guitar': 25, 'electric guitar': 28,
    'bass guitar': 34, 'electric bass': 34,
    'violin': 41, 'viola': 42, 'cello': 43, 'contrabass': 44,
    'double bass': 44,
    'harp': 47, 'marimba': 13, 'xylophone': 14, 'timpani': 48,
    'trumpet': 57, 'trombone': 58, 'tuba': 59, 'french horn': 61,
    'horn': 61,
    'soprano saxophone': 65, 'alto saxophone': 66,
    'tenor saxophone': 67, 'baritone saxophone': 68,
    'oboe': 69, 'bassoon': 71, 'clarinet': 72, 'flute': 74,
    'recorder': 75, 'piccolo': 73,
    'voice': 53, 'soprano': 53, 'alto': 53, 'tenor': 53,
    'baritone': 53, 'bass voice': 53, 'choir': 53,
}

# Clef defaults
_CLEF_MAP = {
    'piano': ('G', '2', 'F', '4'),  # treble + bass
    'harp': ('G', '2', 'F', '4'),
    'marimba': ('G', '2', 'F', '4'),
    'organ': ('G', '2', 'F', '4'),
    'harpsichord': ('G', '2', 'F', '4'),
    'celesta': ('G', '2', 'F', '4'),
    'accordion': ('G', '2', 'F', '4'),
    'viola': ('C', '3'),
    'cello': ('F', '4'),
    'contrabass': ('F', '4'),
    'double bass': ('F', '4'),
    'bassoon': ('F', '4'),
    'trombone': ('F', '4'),
    'tuba': ('F', '4'),
    'bass guitar': ('F', '4'),
    'electric bass': ('F', '4'),
    'timpani': ('F', '4'),
    'bass voice': ('F', '4'),
    'baritone': ('F', '4'),
}


# ── Pitch parsing ────────────────────────────────────────────────────

_PITCH_RE = re.compile(r'^([A-Ga-g])([#b]?)(\d)?$')


def _parse_pitch(pitch_str: str, default_octave: int = 4) -> tuple[str, int, int] | None:
    """Parse 'C#4' → ('C', 1, 4). Returns None for rests.

    If octave is omitted (e.g. 'C', 'F#'), defaults to *default_octave*.
    """
    if not pitch_str or pitch_str.lower() == 'rest':
        return None
    m = _PITCH_RE.match(pitch_str.strip())
    if not m:
        logger.warning("Cannot parse pitch '%s', treating as rest", pitch_str)
        return None
    step = m.group(1).upper()
    alter = {'#': 1, 'b': -1, '': 0}[m.group(2)]
    octave = int(m.group(3)) if m.group(3) else default_octave
    return (step, alter, octave)


# ── Builder ──────────────────────────────────────────────────────────

def _normalize_instrument_key(name: str) -> str:
    """Extract the base instrument name for lookup.

    Handles names like 'Piano - Treble', 'Piano (Bass)', 'Piano Left Hand'.
    """
    key = name.strip().lower()
    # Strip common suffixes
    for suffix in (
        ' - treble', ' - bass', ' (treble)', ' (bass)',
        ' treble', ' bass', ' left hand', ' right hand',
        ' rh', ' lh', ' upper', ' lower',
    ):
        if key.endswith(suffix):
            key = key[: -len(suffix)].strip()
    return key


def _merge_split_parts(parts: list[dict]) -> list[dict]:
    """Merge parts that are split treble/bass for the same grand-staff instrument.

    LLMs sometimes output:
      {"name": "Piano - Treble", "staff": 1, ...}
      {"name": "Piano - Bass",   "staff": 2, ...}
    instead of a single Piano part with staff assignments on each note.
    This function detects and merges them.
    """
    merged: list[dict] = []
    skip_indices: set[int] = set()

    for i, part in enumerate(parts):
        if i in skip_indices:
            continue
        base_key = _normalize_instrument_key(part.get('name', ''))
        if base_key not in _GRAND_STAFF:
            merged.append(part)
            continue

        # Look for a companion part (bass or treble) right after
        staff_val = part.get('staff', 0)
        companion = None
        companion_idx = None
        for j in range(i + 1, len(parts)):
            if j in skip_indices:
                continue
            other_key = _normalize_instrument_key(parts[j].get('name', ''))
            if other_key == base_key:
                companion = parts[j]
                companion_idx = j
                break

        if companion is None:
            merged.append(part)
            continue

        skip_indices.add(companion_idx)

        # Determine which is treble (staff 1) and which is bass (staff 2)
        p_staff = part.get('staff', 1)
        c_staff = companion.get('staff', 2)
        if p_staff == 2 and c_staff != 2:
            treble_part, bass_part = companion, part
        else:
            treble_part, bass_part = part, companion

        # Merge measures: combine note lists with staff assignments
        treble_measures = treble_part.get('measures', [])
        bass_measures = bass_part.get('measures', [])
        max_measures = max(len(treble_measures), len(bass_measures))

        combined_measures = []
        for m_idx in range(max_measures):
            notes = []
            if m_idx < len(treble_measures):
                for n in treble_measures[m_idx].get('notes', []):
                    nc = dict(n)
                    nc['staff'] = 1
                    notes.append(nc)
            if m_idx < len(bass_measures):
                for n in bass_measures[m_idx].get('notes', []):
                    nc = dict(n)
                    nc['staff'] = 2
                    notes.append(nc)
            combined_measures.append({'notes': notes})

        merged.append({
            'name': base_key.title(),  # "Piano"
            'measures': combined_measures,
        })

    return merged


def build_musicxml(score: dict, key: str = "C major",
                   time_signature: str = "4/4",
                   tempo: int = 120) -> str:
    """Build a complete MusicXML 4.0 string from a JSON score dict.

    Args:
        score: JSON dict with title, composer, parts[{name, measures[{notes}]}]
        key: Key signature string, e.g. "C major"
        time_signature: e.g. "4/4", "3/4"
        tempo: BPM

    Returns:
        Valid MusicXML 4.0 string ready for MuseScore.
    """
    title = score.get('title', 'Untitled')
    composer = score.get('composer', 'AI Composer')
    parts = _merge_split_parts(score.get('parts', []))

    if not parts:
        raise ValueError("Score must have at least one part")

    fifths = _KEY_FIFTHS.get(key, 0)
    ts_parts = time_signature.split('/')
    beats = int(ts_parts[0]) if len(ts_parts) == 2 else 4
    beat_type = int(ts_parts[1]) if len(ts_parts) == 2 else 4

    # Calculate expected duration per measure
    # With divisions=4, a quarter note = 4 divisions
    # In 4/4: measure = 4 quarter notes = 16 divisions
    # In 3/4: measure = 3 quarter notes = 12 divisions
    # In 6/8: 6 eighth notes = 12 divisions (compound)
    if beat_type == 8:
        measure_duration = beats * (_DIVISIONS // 2)  # eighth = 2 divisions
    else:
        measure_duration = beats * _DIVISIONS  # quarter = 4 divisions

    # Build XML tree
    root = ET.Element('score-partwise', version='4.0')

    # Work title
    work = ET.SubElement(root, 'work')
    wt = ET.SubElement(work, 'work-title')
    wt.text = title

    # Identification
    ident = ET.SubElement(root, 'identification')
    creator = ET.SubElement(ident, 'creator', type='composer')
    creator.text = composer
    encoding = ET.SubElement(ident, 'encoding')
    sw = ET.SubElement(encoding, 'software')
    sw.text = 'MusicXML Generator'

    # Part list
    part_list = ET.SubElement(root, 'part-list')

    for i, part_data in enumerate(parts):
        part_name = part_data.get('name', f'Part {i+1}')
        pid = f'P{i+1}'
        iid = f'{pid}-I1'
        inst_key = _normalize_instrument_key(part_name)
        midi_prog = _MIDI_PROGRAMS.get(inst_key, 1)

        sp = ET.SubElement(part_list, 'score-part', id=pid)
        pn = ET.SubElement(sp, 'part-name')
        pn.text = part_name
        si = ET.SubElement(sp, 'score-instrument', id=iid)
        iname = ET.SubElement(si, 'instrument-name')
        iname.text = part_name
        mi = ET.SubElement(sp, 'midi-instrument', id=iid)
        mc = ET.SubElement(mi, 'midi-channel')
        mc.text = str(i + 1)
        mp = ET.SubElement(mi, 'midi-program')
        mp.text = str(midi_prog)

    # Parts with measures
    for i, part_data in enumerate(parts):
        part_name = part_data.get('name', f'Part {i+1}')
        pid = f'P{i+1}'
        inst_key = _normalize_instrument_key(part_name)
        is_grand = inst_key in _GRAND_STAFF
        measures_data = part_data.get('measures', [])

        part_elem = ET.SubElement(root, 'part', id=pid)

        for m_idx, m_data in enumerate(measures_data):
            measure = ET.SubElement(part_elem, 'measure', number=str(m_idx + 1))

            # Attributes only in measure 1
            if m_idx == 0:
                attrs = ET.SubElement(measure, 'attributes')
                div = ET.SubElement(attrs, 'divisions')
                div.text = str(_DIVISIONS)
                key_elem = ET.SubElement(attrs, 'key')
                fifths_elem = ET.SubElement(key_elem, 'fifths')
                fifths_elem.text = str(fifths)
                time_elem = ET.SubElement(attrs, 'time')
                beats_elem = ET.SubElement(time_elem, 'beats')
                beats_elem.text = str(beats)
                bt_elem = ET.SubElement(time_elem, 'beat-type')
                bt_elem.text = str(beat_type)

                if is_grand:
                    staves = ET.SubElement(attrs, 'staves')
                    staves.text = '2'
                    clef_data = _CLEF_MAP.get(inst_key, ('G', '2', 'F', '4'))
                    clef1 = ET.SubElement(attrs, 'clef', number='1')
                    s1 = ET.SubElement(clef1, 'sign')
                    s1.text = clef_data[0]
                    l1 = ET.SubElement(clef1, 'line')
                    l1.text = clef_data[1]
                    clef2 = ET.SubElement(attrs, 'clef', number='2')
                    s2 = ET.SubElement(clef2, 'sign')
                    s2.text = clef_data[2]
                    l2 = ET.SubElement(clef2, 'line')
                    l2.text = clef_data[3]
                else:
                    clef_data = _CLEF_MAP.get(inst_key, ('G', '2'))
                    clef_elem = ET.SubElement(attrs, 'clef')
                    cs = ET.SubElement(clef_elem, 'sign')
                    cs.text = clef_data[0]
                    cl = ET.SubElement(clef_elem, 'line')
                    cl.text = clef_data[1]

                # Tempo direction (first part only)
                if i == 0:
                    direction = ET.SubElement(measure, 'direction', placement='above')
                    dt = ET.SubElement(direction, 'direction-type')
                    metro = ET.SubElement(dt, 'metronome')
                    bu = ET.SubElement(metro, 'beat-unit')
                    bu.text = 'quarter'
                    pm = ET.SubElement(metro, 'per-minute')
                    pm.text = str(tempo)
                    sound = ET.SubElement(direction, 'sound', tempo=str(tempo))

            # Process notes
            notes = m_data.get('notes', [])

            if is_grand:
                _build_grand_staff_measure(measure, notes, measure_duration)
            else:
                _build_single_staff_measure(measure, notes, measure_duration)

        # Final barline on last measure
        if measures_data:
            last_measure = list(part_elem)[-1]
            barline = ET.SubElement(last_measure, 'barline', location='right')
            bs = ET.SubElement(barline, 'bar-style')
            bs.text = 'light-heavy'

    # Serialize
    ET.indent(root, space='  ')
    xml_str = ET.tostring(root, encoding='unicode', xml_declaration=False)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC '
        '"-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        + xml_str
    )


def _build_note_element(parent: ET.Element, note_data: dict,
                        voice: int, staff: int | None) -> int:
    """Add a <note> element to parent. Returns the duration in divisions."""
    dur_name = _normalize_duration(note_data.get('duration', 'quarter'))
    pitch_str = note_data.get('pitch', 'rest')
    is_chord = note_data.get('chord', False)

    dur_divs = _DUR_DIVS.get(dur_name, _DIVISIONS)  # default to quarter
    dur_type = _DUR_TYPE.get(dur_name, 'quarter')
    is_dotted = dur_name in _DOTTED

    note_elem = ET.SubElement(parent, 'note')

    # Chord flag (means this note plays at the same time as previous)
    if is_chord:
        ET.SubElement(note_elem, 'chord')

    # Pitch or rest
    default_oct = 3 if (staff is not None and staff == 2) else 4
    parsed = _parse_pitch(pitch_str, default_octave=default_oct)
    if parsed is None:
        ET.SubElement(note_elem, 'rest')
    else:
        step_name, alter, octave = parsed
        pitch = ET.SubElement(note_elem, 'pitch')
        step_e = ET.SubElement(pitch, 'step')
        step_e.text = step_name
        if alter != 0:
            alter_e = ET.SubElement(pitch, 'alter')
            alter_e.text = str(alter)
        oct_e = ET.SubElement(pitch, 'octave')
        oct_e.text = str(octave)

    # Duration
    dur_e = ET.SubElement(note_elem, 'duration')
    dur_e.text = str(dur_divs)

    # Voice
    voice_e = ET.SubElement(note_elem, 'voice')
    voice_e.text = str(voice)

    # Type
    type_e = ET.SubElement(note_elem, 'type')
    type_e.text = dur_type

    # Dot
    if is_dotted:
        ET.SubElement(note_elem, 'dot')

    # Staff (for grand staff)
    if staff is not None:
        staff_e = ET.SubElement(note_elem, 'staff')
        staff_e.text = str(staff)

    return dur_divs if not is_chord else 0  # chords don't add duration


def _pad_measure(parent: ET.Element, remaining: int,
                 voice: int, staff: int | None) -> None:
    """Fill remaining duration with rests to complete the measure."""
    # Use largest possible rest values
    rest_vals = [
        (16, 'whole'), (12, 'half'),  # dotted half = 12, but simpler with half
        (8, 'half'), (4, 'quarter'), (2, 'eighth'), (1, '16th'),
    ]
    while remaining > 0:
        for divs, rtype in rest_vals:
            if divs <= remaining:
                _build_note_element(parent, {
                    'pitch': 'rest', 'duration': rtype,
                }, voice=voice, staff=staff)
                remaining -= divs
                break
        else:
            break  # safety: shouldn't happen


def _build_single_staff_measure(measure: ET.Element, notes: list,
                                measure_duration: int) -> None:
    """Build notes for a single-staff measure."""
    total = 0
    for nd in notes:
        added = _build_note_element(measure, nd, voice=1, staff=None)
        total += added

    # Pad if needed
    if total < measure_duration:
        _pad_measure(measure, measure_duration - total, voice=1, staff=None)


def _build_grand_staff_measure(measure: ET.Element, notes: list,
                               measure_duration: int) -> None:
    """Build notes for a grand-staff measure with backup between staves."""
    # Separate notes by staff
    staff1_notes = []
    staff2_notes = []
    for nd in notes:
        s = nd.get('staff', 1)
        # Accept various formats: 1, 2, "1", "2", "treble", "bass"
        if isinstance(s, str):
            if s.lower() in ('2', 'bass', 'left'):
                staff2_notes.append(nd)
            else:
                staff1_notes.append(nd)
        elif s == 2:
            staff2_notes.append(nd)
        else:
            staff1_notes.append(nd)

    # If no staff separation provided, put all in staff 1
    if not staff2_notes and staff1_notes:
        pass  # all treble, no bass

    # Staff 1 notes
    s1_total = 0
    for nd in staff1_notes:
        added = _build_note_element(measure, nd, voice=1, staff=1)
        s1_total += added

    # Pad staff 1 if short
    if s1_total < measure_duration and s1_total > 0:
        _pad_measure(measure, measure_duration - s1_total, voice=1, staff=1)
    elif s1_total == 0:
        # Empty staff 1: add full-measure rest
        _pad_measure(measure, measure_duration, voice=1, staff=1)

    s1_dur = max(s1_total, measure_duration)

    # Backup
    if staff2_notes or True:  # Always write staff 2 for grand staff
        backup = ET.SubElement(measure, 'backup')
        dur = ET.SubElement(backup, 'duration')
        dur.text = str(s1_dur)

        # Staff 2 notes
        s2_total = 0
        for nd in staff2_notes:
            added = _build_note_element(measure, nd, voice=2, staff=2)
            s2_total += added

        # Pad staff 2 if short
        if s2_total < measure_duration:
            _pad_measure(measure, measure_duration - s2_total, voice=2, staff=2)
