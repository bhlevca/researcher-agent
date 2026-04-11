"""MusicXML postprocessor — fixes common LLM generation errors.

Applies a two-phase pipeline:
  Phase 1 — Regex pre-fixes: repair broken XML so it can parse.
  Phase 2 — ElementTree structural repairs: fix semantics.

Error patterns handled (observed from qwen3.5:9b, qwen2.5, etc.):
  - Note attributes (timestamp, staff, octave, duration) instead of children
  - Garbage closing tags (</ major>, </note major>, etc.)
  - Malformed attribute values (timestamp="hold>2")
  - Broken/duplicated tag names (<divisions> divisions>1)
  - <notes> instead of <note>
  - Wrong <harmony> syntax (attributes instead of child elements)
  - Wrong <metronome> / <sound> nesting
  - <articulation> singular instead of <articulations> plural
  - <unpitched> mixed with <pitch>
  - Naked text (bare note names like "F") inside <note>
  - Missing <voice>, <type>, <staff>, <duration> child elements
  - Missing <backup> between staff changes in same measure
  - Repeated <attributes> blocks in measures 2+
  - HTML comments that skip measures
  - Incomplete/truncated final measure
"""

import re
import logging
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# Duration→type mapping for divisions=1
_DUR_TYPE_MAP = {'4': 'whole', '2': 'half', '1': 'quarter'}
# Reverse for inferring duration from type
_TYPE_DUR_MAP = {'whole': '4', 'half': '2', 'quarter': '1', 'eighth': '1', '16th': '1'}


# ---------------------------------------------------------------------------
# Phase 1: Regex-based pre-fixes (before XML parsing)
# ---------------------------------------------------------------------------

def _regex_prefixes(xml: str) -> str:
    """Fix tag-level errors that would prevent XML parsing."""

    # ── 0. Strip incomplete final measure (LLM ran out of tokens) ──
    # If the XML ends with an incomplete <measure> (no </measure>), drop it.
    last_measure_open = xml.rfind('<measure ')
    last_measure_close = xml.rfind('</measure>')
    if last_measure_open > last_measure_close:
        xml = xml[:last_measure_open]

    # ── 1. Ensure we close </part> and </score-partwise> if truncated ──
    if '<part ' in xml and '</part>' not in xml:
        xml += '\n  </part>'
    if '<score-partwise' in xml and '</score-partwise>' not in xml:
        xml += '\n</score-partwise>'

    # ── 2. <notes> → <note>, </notes> → </note> ──
    xml = re.sub(r'<notes\b', '<note', xml)
    xml = re.sub(r'</notes>', '</note>', xml)

    # ── 3. Remove HTML comments ──
    xml = re.sub(r'<!--[\s\S]*?-->', '', xml)

    # ── 4. Fix garbage closing tags: </ major>, </note major>, etc. ──
    xml = re.sub(r'</\s+\w+>', '</note>', xml)          # "</ major>" → "</note>"
    xml = re.sub(r'</note\s+\w+>', '</note>', xml)      # "</note major>" → "</note>"

    # ── 5. Fix broken/duplicated tag names ──
    # "<divisions> divisions>1" → "<divisions>1"
    xml = re.sub(r'<(\w+)>\s*\1>', r'<\1>', xml)
    # "<divisions>1</divisions>" should remain; fix "<divisions> 1" too
    xml = re.sub(r'<(divisions|beats|beat-type)>\s+(\d+)', r'<\1>\2', xml)

    # ── 6. Fix note elements with attributes instead of children ──
    # Pattern: <note timestamp="..." staff="..." octave="..." duration="...">
    #   <pitch><step>C</step><octave>4</octave></pitch>
    # </note>
    # → <note> with <duration>, <staff>, <voice>, <type> as children
    def _fix_note_attrs(m):
        attrs_str = m.group('attrs')
        inner = m.group('inner')
        # Extract attribute values
        staff = _attr_val(attrs_str, 'staff')
        dur = _attr_val(attrs_str, 'duration')
        # The inner content might already have <pitch> etc.
        return f'<note>{inner}</note>'  # attrs will be added as children in Phase 2

    # This handles <note attr="val" attr2="val2">...</note>
    xml = re.sub(
        r'<note\s+(?P<attrs>[^>]+?)>(?P<inner>[\s\S]*?)</note>',
        _fix_note_attrs, xml
    )

    # ── 7. Fix malformed attribute values that break XML parsing ──
    # e.g., timestamp="hold>2" — the > breaks the tag
    # Fix by removing the offending attributes entirely
    xml = re.sub(r'\s+timestamp="[^"]*"', '', xml)

    # ── 8. Fix naked text inside <note> that should be <pitch> ──
    # Remove stray bare note names (A-G, possibly with # or b) sitting
    # between closing and opening tags (outside any element).
    xml = re.sub(
        r'(</\w+>)\s*[A-G][#b]?\s*(?=<)',
        r'\1',
        xml
    )
    # Inside a <note>, replace bare text that isn't inside any element
    xml = re.sub(
        r'(<note>[\s\S]*?)\n\s*([A-G][#b]?)\s*\n([\s\S]*?</note>)',
        r'\1\3',  # Just remove the bare text, pitch should already be there
        xml
    )

    # ── 9. Fix <harmony> attributes → child elements ──
    def _fix_harmony(m):
        root = m.group('root') or 'C'
        kind = m.group('kind') or 'major'
        return (
            f'<harmony>\n'
            f'          <root><root-step>{root}</root-step></root>\n'
            f'          <kind>{kind}</kind>\n'
            f'        </harmony>'
        )
    xml = re.sub(
        r'<harmony\s+[^>]*?root="(?P<root>[A-Ga-g][#b]?)"'
        r'\s+type="(?P<kind>[^"]*)"[^>]*>.*?</harmony>',
        _fix_harmony, xml
    )
    xml = re.sub(
        r'<harmony\s+[^>]*?type="(?P<kind>[^"]*)"'
        r'\s+root="(?P<root>[A-Ga-g][#b]?)"[^>]*>.*?</harmony>',
        _fix_harmony, xml
    )
    def _fix_harmony_simple(m):
        text = m.group('text').strip()
        root_m = re.match(r'([A-G][#b]?)', text)
        root = root_m.group(1) if root_m else 'C'
        rest = text[len(root):]
        kind = 'major'
        if rest.startswith('m') and not rest.startswith('maj'):
            kind = 'minor'
        elif 'maj7' in rest:
            kind = 'major-seventh'
        elif '7' in rest:
            kind = 'dominant'
        elif 'dim' in rest:
            kind = 'diminished'
        elif 'aug' in rest:
            kind = 'augmented'
        return (
            f'<harmony>\n'
            f'          <root><root-step>{root}</root-step></root>\n'
            f'          <kind>{kind}</kind>\n'
            f'        </harmony>'
        )
    xml = re.sub(
        r'<harmony\s+function="[^"]*">(?P<text>[^<]+)</harmony>',
        _fix_harmony_simple, xml
    )

    # ── 10. Fix <metronome>: <beat>N</beat> → <beat-unit>quarter</beat-unit> ──
    xml = re.sub(r'<beat>(?:1|quarter)</beat>', '<beat-unit>quarter</beat-unit>', xml)
    xml = re.sub(r'<beat>(?:2|half)</beat>', '<beat-unit>half</beat-unit>', xml)

    # ── 11. Fix <sound> containing children → self-closing ──
    def _fix_sound_metronome(m):
        content = m.group(1)
        pm = re.search(r'<per-minute>(\d+)</per-minute>', content)
        tempo = pm.group(1) if pm else '120'
        return f'<sound tempo="{tempo}"/>'
    xml = re.sub(r'<sound>\s*([\s\S]*?)\s*</sound>', _fix_sound_metronome, xml)

    # ── 12. Remove <unpitched> when <pitch> is present ──
    xml = re.sub(r'<unpitched>[\s\S]*?</unpitched>\s*(?=<pitch>)', '', xml)
    xml = re.sub(r'<unpitched>\s*(<pitch>[\s\S]*?</pitch>)\s*</unpitched>', r'\1', xml)

    # ── 13. <articulation> singular → <notations><articulations> ──
    xml = re.sub(r'<articulation>', '<notations><articulations>', xml)
    xml = re.sub(r'</articulation>', '</articulations></notations>', xml)

    # ── 14. Remove invalid elements ──
    xml = re.sub(r'<hairpin>[\s\S]*?</hairpin>', '', xml)
    xml = re.sub(r'<hairpin[^/]*/>', '', xml)
    xml = re.sub(r'<instrument\s+id="[^"]*">\d+</instrument>', '', xml)
    xml = re.sub(r'<style>[^<]*</style>', '', xml)

    # ── 15. Remove empty wrappers ──
    xml = re.sub(r'<notations>\s*</notations>', '', xml)

    # ── 16. Clean up whitespace ──
    xml = re.sub(r'\n\s*\n\s*\n', '\n\n', xml)

    return xml


def _attr_val(attrs_str: str, name: str) -> str | None:
    """Extract an attribute value from a raw attribute string."""
    m = re.search(rf'{name}="([^"]*)"', attrs_str)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Phase 2: ElementTree structural repair
# ---------------------------------------------------------------------------

def _reorder_note_children(root: ET.Element) -> None:
    """Reorder children of <note> elements to MusicXML spec order."""
    ORDER = [
        'grace', 'chord', 'pitch', 'unpitched', 'rest',
        'duration', 'tie', 'instrument', 'footnote', 'level',
        'voice', 'type', 'dot', 'accidental', 'time-modification',
        'stem', 'notehead', 'notehead-text', 'staff', 'beam',
        'notations', 'lyric',
    ]
    order_map = {tag: i for i, tag in enumerate(ORDER)}

    for note in root.iter('note'):
        children = list(note)
        if not children:
            continue
        children.sort(key=lambda e: order_map.get(e.tag, 999))
        for child in list(note):
            note.remove(child)
        for child in children:
            note.append(child)


def _fix_direction_placement(root: ET.Element) -> None:
    """Ensure <direction> elements have <sound> as direct child."""
    for direction in root.iter('direction'):
        for dt in direction.findall('direction-type'):
            for sound in dt.findall('sound'):
                dt.remove(sound)
                direction.append(sound)


def _ensure_divisions(root: ET.Element) -> None:
    """Ensure the first measure has <divisions> in <attributes>."""
    for part in root.iter('part'):
        measures = list(part.iter('measure'))
        if not measures:
            continue
        first = measures[0]
        attrs = first.find('attributes')
        if attrs is None:
            attrs = ET.SubElement(first, 'attributes')
            children = list(first)
            first.remove(attrs)
            first.insert(0, attrs)
        if attrs.find('divisions') is None:
            div = ET.SubElement(attrs, 'divisions')
            div.text = '1'


def _deduplicate_attributes(root: ET.Element) -> None:
    """Remove redundant <attributes> blocks from measures 2+.

    MusicXML only needs <attributes> when something changes. LLMs
    often repeat the full attributes block in every measure.
    """
    for part in root.iter('part'):
        measures = list(part.iter('measure'))
        if len(measures) < 2:
            continue
        first_attrs = measures[0].find('attributes')
        if first_attrs is None:
            continue

        # Serialize first measure's attributes for comparison
        def _attrs_sig(attrs_elem):
            """Create a comparable signature of an attributes element."""
            parts = []
            for child in attrs_elem:
                parts.append(ET.tostring(child, encoding='unicode'))
            return ''.join(sorted(parts))

        first_sig = _attrs_sig(first_attrs)

        for measure in measures[1:]:
            attrs = measure.find('attributes')
            if attrs is not None and _attrs_sig(attrs) == first_sig:
                measure.remove(attrs)


def _ensure_note_children(root: ET.Element) -> None:
    """Ensure every <note> has required children: duration, voice, type.

    Also adds <staff> if the part has <staves>2</staves>.
    """
    # Determine which parts have multiple staves
    parts_staves = {}
    for part in root.iter('part'):
        pid = part.get('id', '')
        staves_elem = None
        for attrs in part.iter('attributes'):
            s = attrs.find('staves')
            if s is not None and s.text:
                staves_elem = s
                break
        parts_staves[pid] = int(staves_elem.text) if staves_elem is not None else 1

    for part in root.iter('part'):
        pid = part.get('id', '')
        num_staves = parts_staves.get(pid, 1)

        # Determine divisions from first measure
        divisions = 1
        first_attrs = None
        for measure in part.iter('measure'):
            first_attrs = measure.find('attributes')
            if first_attrs is not None:
                div_elem = first_attrs.find('divisions')
                if div_elem is not None and div_elem.text:
                    divisions = int(div_elem.text)
                break

        for note in part.iter('note'):
            # Ensure <duration>
            if note.find('duration') is None:
                type_elem = note.find('type')
                dur = ET.SubElement(note, 'duration')
                if type_elem is not None and type_elem.text:
                    dur.text = _TYPE_DUR_MAP.get(type_elem.text, str(divisions))
                else:
                    dur.text = str(divisions)  # default to quarter

            # Ensure <voice>
            if note.find('voice') is None:
                voice = ET.SubElement(note, 'voice')
                staff_elem = note.find('staff')
                if staff_elem is not None and staff_elem.text == '2':
                    voice.text = '2'
                else:
                    voice.text = '1'

            # Ensure <type>
            if note.find('type') is None:
                dur_elem = note.find('duration')
                type_elem = ET.SubElement(note, 'type')
                if dur_elem is not None and dur_elem.text:
                    type_elem.text = _DUR_TYPE_MAP.get(dur_elem.text, 'quarter')
                else:
                    type_elem.text = 'quarter'

            # Ensure <staff> for multi-staff parts
            if num_staves > 1 and note.find('staff') is None:
                staff = ET.SubElement(note, 'staff')
                voice_elem = note.find('voice')
                if voice_elem is not None and voice_elem.text == '2':
                    staff.text = '2'
                else:
                    staff.text = '1'


def _insert_backups(root: ET.Element) -> None:
    """Insert <backup> elements between staff changes in measures.

    When a measure has notes on staff 1 then staff 2, the switch needs
    a <backup> element with the total duration of the first staff's notes.
    """
    for part in root.iter('part'):
        # Check if this is a multi-staff part
        num_staves = 1
        for attrs in part.iter('attributes'):
            s = attrs.find('staves')
            if s is not None and s.text:
                num_staves = int(s.text)
                break
        if num_staves < 2:
            continue

        for measure in part.iter('measure'):
            notes = list(measure.iter('note'))
            if not notes:
                continue

            # Check if we already have a <backup>
            if measure.find('backup') is not None:
                continue

            # Group notes by staff
            staff1_notes = []
            staff2_notes = []
            for note in notes:
                staff_elem = note.find('staff')
                staff_num = staff_elem.text if staff_elem is not None else '1'
                if staff_num == '2':
                    staff2_notes.append(note)
                else:
                    staff1_notes.append(note)

            if not staff1_notes or not staff2_notes:
                continue

            # Calculate total duration of staff 1 notes
            total_dur = 0
            for note in staff1_notes:
                dur_elem = note.find('duration')
                if dur_elem is not None and dur_elem.text:
                    try:
                        total_dur += int(dur_elem.text)
                    except ValueError:
                        total_dur += 1
                else:
                    total_dur += 1

            if total_dur <= 0:
                continue

            # Find the first staff-2 note and insert backup before it
            first_s2 = staff2_notes[0]
            children = list(measure)
            idx = list(measure).index(first_s2)

            backup = ET.Element('backup')
            dur = ET.SubElement(backup, 'duration')
            dur.text = str(total_dur)
            measure.insert(idx, backup)


def _remove_empty_elements(root: ET.Element) -> None:
    """Remove empty wrapper elements that have no children and no text."""
    for parent in root.iter():
        to_remove = []
        for child in parent:
            if child.tag in ('notations', 'articulations', 'ornaments',
                             'technical', 'dynamics') and len(child) == 0:
                if not (child.text and child.text.strip()):
                    to_remove.append(child)
        for child in to_remove:
            parent.remove(child)


def _fix_accent_elements(root: ET.Element) -> None:
    """Convert <accent></accent> to self-closing <accent/>."""
    for tag in ('accent', 'staccato', 'tenuto', 'fermata'):
        for elem in root.iter(tag):
            if elem.text and not elem.text.strip():
                elem.text = None


def _remove_notes_without_pitch_or_rest(root: ET.Element) -> None:
    """Remove <note> elements that have no <pitch> and no <rest>.

    These are garbage notes produced by LLM hallucination.
    """
    for measure in root.iter('measure'):
        to_remove = []
        for note in measure.findall('note'):
            has_pitch = note.find('pitch') is not None
            has_rest = note.find('rest') is not None
            if not has_pitch and not has_rest:
                to_remove.append(note)
        for note in to_remove:
            measure.remove(note)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fix_musicxml(xml: str) -> str:
    """Apply all MusicXML fixes and return corrected XML string.

    Two-phase pipeline:
      Phase 1 — regex pre-fixes (make it parseable)
      Phase 2 — ElementTree structural repairs (make it valid)
    """
    if not xml or '<score-partwise' not in xml:
        return xml

    # Phase 1: regex fixes
    xml = _regex_prefixes(xml)

    # Phase 2: try ElementTree parse for structural fixes
    try:
        root = ET.fromstring(xml)

        _ensure_divisions(root)
        _remove_notes_without_pitch_or_rest(root)
        _ensure_note_children(root)
        _deduplicate_attributes(root)
        _insert_backups(root)
        _reorder_note_children(root)
        _fix_direction_placement(root)
        _remove_empty_elements(root)
        _fix_accent_elements(root)

        # Re-serialize
        ET.indent(root, space='  ')
        xml_out = ET.tostring(root, encoding='unicode', xml_declaration=False)

        # Add XML declaration and DOCTYPE back
        xml_out = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE score-partwise PUBLIC '
            '"-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">\n'
            + xml_out
        )
        logger.info("[MusicXML] Postprocessor: structural fixes applied successfully")
        return xml_out

    except ET.ParseError as e:
        logger.warning("[MusicXML] Postprocessor: XML still malformed after regex fixes: %s", e)
        return xml
