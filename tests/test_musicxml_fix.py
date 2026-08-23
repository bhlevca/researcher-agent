"""Tests for the MusicXML postprocessor."""

from xml.etree import ElementTree as ET
from researcher.composer.musicxml_fix import fix_musicxml, _regex_prefixes


BROKEN_SAMPLE = '''\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work><work-title>Lullaby in C Major</work-title></work>
  <identification>
    <creator type="composer">AI Composer</creator>
  </identification>
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
      <score-instrument id="P1-I1">
        <instrument-name>Piano</instrument-name>
      </score-instrument>
      <midi-instrument id="P1-I1">
        <midi-channel>1</midi-channel>
        <midi-program>1</midi-program>
      </midi-instrument>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>3</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="above">
        <direction-type>
          <metronome>
            <beat>1</beat>
            <per-minute>80</per-minute>
          </metronome>
        </direction-type>
        <sound><metronome><beat>1</beat><per-minute>80</per-minute></metronome></sound>
      </direction>
      <harmony function="Tc" root="C" type="major">C</harmony>
      <notes>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>2</duration>
        <voice>1</voice>
        <type>half</type>
        <stem>up</stem>
      </notes>
      <notes>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>1</duration>
        <voice>1</voice>
        <type>quarter</type>
        <stem>up</stem>
      </notes>
    </measure>
    <measure number="2">
      <harmony function="S" root="F" type="major">F</harmony>
      <notes>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>2</duration>
        <voice>1</voice>
        <type>half</type>
        <stem>up</stem>
      </notes>
      <notes>
        <pitch><step>C</step><octave>5</octave></pitch>
        <duration>1</duration>
        <voice>1</voice>
        <type>quarter</type>
        <stem>up</stem>
        <articulation><accent></accent></articulation>
      </notes>
    </measure>
    <!-- Continue with similar structure for remaining measures -->
  </part>
</score-partwise>
'''


# The exact LLM output from qwen3.5:9b that produced 77 MuseScore errors.
# Contains: note attributes, garbage tags, broken tags, repeated attributes,
# missing backup/voice/type/staff, truncated output.
QWEN_BROKEN = '''\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work><work-title>Lullaby in C</work-title></work>
  <identification>
    <creator type="composer">AI Composer</creator>
    <encoding>
      <software>MusicXML Generator</software>
      <encoding-date>2024-01-01</encoding-date>
    </encoding>
  </identification>
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
      <score-instrument id="P1-I1">
        <instrument-name>Piano</instrument-name>
      </score-instrument>
      <midi-instrument id="P1-I1">
        <midi-channel>1</midi-channel>
        <midi-program>1</midi-program>
      </midi-instrument>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <direction placement="above">
        <direction-type>
          <metronome>
            <beat-unit>quarter</beat-unit>
            <per-minute>120</per-minute>
          </metronome>
        </direction-type>
        <sound tempo="120"/>
      </direction>
      <!-- Measure 1: C major chord -->
      <note timestamp="0" staff="1" octave="4" duration="1">
        <pitch><step>C</step><octave>4</octave></pitch>
      </note>
      <note timestamp="1" staff="1" octave="4" duration="1">
        <pitch><step>E</step><octave>4</octave></pitch>
      </note>
      <note timestamp="2" staff="1" octave="4" duration="1">
        <pitch><step>G</step><octave>3</octave></pitch>
      </note>
      <note timestamp="3" staff="1" octave="4" duration="1">
        <pitch><step>C</step><octave>4</octave></pitch>
      </note>
      <note timestamp="0" staff="2" octave="3" duration="1">
        <pitch><step>C</step><octave>3</octave></pitch>
      </note>
      <note timestamp="1" staff="2" octave="3" duration="1">
        <pitch><step>E</step><octave>3</octave></pitch>
      </note>
      <note timestamp="2" staff="2" octave="3" duration="1">
        <pitch><step>G</step><octave>3</octave></pitch>
      </note>
      <note timestamp="3" staff="2" octave="3" duration="1">
        <pitch><step>C</step><octave>3</octave></pitch>
      </note>
    </measure>
    <measure number="2">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <!-- Measure 2: F major chord (IV) -->
      <note timestamp="0" staff="1" octave="4" duration="1">
        <pitch><step>F</step><octave>4</octave></pitch>
      </note>
      <note timestamp="1" staff="1" octave="4" duration="1">
        <pitch><step>A</step><octave>4</octave></pitch>
      </note>
      <note timestamp="2" staff="1" octave="4" duration="1">
        <pitch><step>C</step><octave>4</octave></pitch>
      </note>
      <note timestamp="3" staff="1" octave="4" duration="1">
        <pitch><step>F</step><octave>4</octave></pitch>
      </note>
      <note timestamp="0" staff="2" octave="3" duration="1">
        <pitch><step>F</step><octave>3</octave></pitch>
      </note>
      <note timestamp="1" staff="2" octave="3" duration="1">
        <pitch><step>A</step><octave>3</octave></pitch>
      </note>
      <note timestamp="2" staff="2" octave="3" duration="1">
        <pitch><step>C</step><octave>3</octave></pitch>
      </note>
      <note timestamp="3" staff="2" octave="3" duration="1">
        <pitch><step>F</step><octave>3</octave></pitch>
      </note>
    </measure>
  </part>
</score-partwise>
'''


def test_notes_renamed_to_note():
    """<notes> should become <note>."""
    result = _regex_prefixes(BROKEN_SAMPLE)
    assert '<note>' in result
    assert '<notes>' not in result
    assert '</note>' in result
    assert '</notes>' not in result


def test_harmony_fixed():
    """<harmony root="C" type="major"> should become child elements."""
    result = _regex_prefixes(BROKEN_SAMPLE)
    assert '<root-step>' in result
    assert '<kind>' in result
    # Old attribute form should be gone
    assert 'root="C"' not in result
    assert 'root="F"' not in result


def test_beat_unit_fixed():
    """<beat>1</beat> should become <beat-unit>quarter</beat-unit>."""
    result = _regex_prefixes(BROKEN_SAMPLE)
    assert '<beat-unit>quarter</beat-unit>' in result


def test_sound_fixed():
    """<sound> containing <metronome> should become <sound tempo="80"/>."""
    result = _regex_prefixes(BROKEN_SAMPLE)
    assert 'tempo="80"' in result
    # There should be no <sound> with children
    assert '<sound><metronome>' not in result


def test_html_comments_removed():
    """HTML comments should be stripped."""
    result = _regex_prefixes(BROKEN_SAMPLE)
    assert '<!--' not in result
    assert '-->' not in result


def test_articulation_fixed():
    """<articulation> (singular) → <notations><articulations>."""
    result = _regex_prefixes(BROKEN_SAMPLE)
    assert '<notations><articulations>' in result
    assert '</articulations></notations>' in result


def test_full_postprocessor_parses():
    """The full fix_musicxml output should be valid XML."""
    result = fix_musicxml(BROKEN_SAMPLE)
    assert result is not None
    # Strip the declaration + DOCTYPE for ET parsing
    xml_body = result
    if '<!DOCTYPE' in xml_body:
        idx = xml_body.index('<!DOCTYPE')
        end = xml_body.index('>', idx) + 1
        xml_body = xml_body[:idx] + xml_body[end:]
    root = ET.fromstring(xml_body)
    assert root.tag == 'score-partwise'


def test_full_postprocessor_note_order():
    """After full processing, <note> children should be in spec order."""
    result = fix_musicxml(BROKEN_SAMPLE)
    xml_body = result
    if '<!DOCTYPE' in xml_body:
        idx = xml_body.index('<!DOCTYPE')
        end = xml_body.index('>', idx) + 1
        xml_body = xml_body[:idx] + xml_body[end:]
    root = ET.fromstring(xml_body)

    for note in root.iter('note'):
        children = [c.tag for c in note]
        # pitch/rest must come before duration
        if 'pitch' in children and 'duration' in children:
            assert children.index('pitch') < children.index('duration')
        if 'duration' in children and 'voice' in children:
            assert children.index('duration') < children.index('voice')


def test_passthrough_non_musicxml():
    """Non-MusicXML strings should pass through unchanged."""
    plain = "Hello, this is just text."
    assert fix_musicxml(plain) == plain
    assert fix_musicxml("") == ""
    assert fix_musicxml(None) is None


def test_divisions_ensured():
    """First measure should have <divisions> after processing."""
    result = fix_musicxml(BROKEN_SAMPLE)
    xml_body = result
    if '<!DOCTYPE' in xml_body:
        idx = xml_body.index('<!DOCTYPE')
        end = xml_body.index('>', idx) + 1
        xml_body = xml_body[:idx] + xml_body[end:]
    root = ET.fromstring(xml_body)
    parts = list(root.iter('part'))
    assert len(parts) >= 1
    first_measure = list(parts[0].iter('measure'))[0]
    attrs = first_measure.find('attributes')
    assert attrs is not None
    assert attrs.find('divisions') is not None


# ═══════════════════════════════════════════════════════════════════
# Tests for qwen3.5 note-attribute pattern
# ═══════════════════════════════════════════════════════════════════

def _parse_fixed(xml_str):
    """Helper: run fix_musicxml, strip DOCTYPE, parse to ET."""
    fixed = fix_musicxml(xml_str)
    body = fixed
    if '<!DOCTYPE' in body:
        idx = body.index('<!DOCTYPE')
        end = body.index('>', idx) + 1
        body = body[:idx] + body[end:]
    return ET.fromstring(body)


def test_qwen_note_attrs_stripped():
    """Note attributes (timestamp, staff, octave, duration) must be removed."""
    result = fix_musicxml(QWEN_BROKEN)
    assert 'timestamp=' not in result


def test_qwen_parses_as_xml():
    """Fixed qwen output must parse as valid XML."""
    root = _parse_fixed(QWEN_BROKEN)
    assert root.tag == 'score-partwise'


def test_qwen_notes_have_duration():
    """Every <note> must have a <duration> child element."""
    root = _parse_fixed(QWEN_BROKEN)
    for note in root.iter('note'):
        assert note.find('duration') is not None, "Note missing <duration>"


def test_qwen_notes_have_voice():
    """Every <note> must have a <voice> child element."""
    root = _parse_fixed(QWEN_BROKEN)
    for note in root.iter('note'):
        assert note.find('voice') is not None, "Note missing <voice>"


def test_qwen_notes_have_type():
    """Every <note> must have a <type> child element."""
    root = _parse_fixed(QWEN_BROKEN)
    for note in root.iter('note'):
        assert note.find('type') is not None, "Note missing <type>"


def test_qwen_notes_have_staff():
    """Grand-staff notes must have a <staff> child element."""
    root = _parse_fixed(QWEN_BROKEN)
    for note in root.iter('note'):
        assert note.find('staff') is not None, "Note missing <staff>"


def test_qwen_backup_inserted():
    """Grand-staff measures must have <backup> between staff 1 and staff 2."""
    root = _parse_fixed(QWEN_BROKEN)
    for measure in root.iter('measure'):
        notes = list(measure.iter('note'))
        if not notes:
            continue
        staff_nums = set()
        for n in notes:
            s = n.find('staff')
            if s is not None and s.text:
                staff_nums.add(s.text)
        if '1' in staff_nums and '2' in staff_nums:
            assert measure.find('backup') is not None, \
                f"Measure {measure.get('number')} missing <backup>"


def test_qwen_redundant_attributes_removed():
    """Attributes block in measure 2 (identical to measure 1) should be removed."""
    root = _parse_fixed(QWEN_BROKEN)
    measures = list(root.iter('measure'))
    assert len(measures) >= 2
    # Measure 2 should NOT have <attributes> (it's identical to measure 1)
    assert measures[1].find('attributes') is None, \
        "Measure 2 still has redundant <attributes>"


def test_qwen_note_child_order():
    """Note children must be in spec order: pitch, duration, voice, type, staff."""
    root = _parse_fixed(QWEN_BROKEN)
    for note in root.iter('note'):
        children = [c.tag for c in note]
        if 'pitch' in children and 'duration' in children:
            assert children.index('pitch') < children.index('duration')
        if 'duration' in children and 'voice' in children:
            assert children.index('duration') < children.index('voice')
        if 'voice' in children and 'type' in children:
            assert children.index('voice') < children.index('type')
        if 'type' in children and 'staff' in children:
            assert children.index('type') < children.index('staff')


def test_truncated_xml_recovered():
    """XML truncated mid-measure should still parse (incomplete measure dropped)."""
    truncated = QWEN_BROKEN.replace('</score-partwise>', '')
    truncated = truncated.rsplit('</measure>', 1)[0]  # Remove last </measure>
    result = fix_musicxml(truncated)
    assert '<score-partwise' in result
    assert '</score-partwise>' in result


def test_garbage_closing_tag():
    """Garbage closing tags like '</ major>' should be fixed."""
    xml = QWEN_BROKEN.replace(
        '<pitch><step>C</step><octave>4</octave></pitch>\n      </note>',
        '<pitch><step>C</step><octave>4</octave></pitch>\n      </ major>',
        1
    )
    result = fix_musicxml(xml)
    assert '</ major>' not in result
    root = _parse_fixed(xml)
    assert root.tag == 'score-partwise'


# ---------------------------------------------------------------------------
# ABC composition path (replaces the JSON note-list format)
# ---------------------------------------------------------------------------

import pytest
from researcher.composer.crew import ComposerCrew


class TestAbcExtraction:
    def test_pulls_abc_from_a_fenced_block(self):
        raw = "Here you go:\n```abc\nX:1\nT:T\nM:4/4\nL:1/8\nK:C\n|C4 E4|\n```"
        abc = ComposerCrew.extract_abc(raw)
        assert abc and abc.startswith("X:1")

    def test_accepts_bare_abc_without_a_fence(self):
        assert ComposerCrew.extract_abc("X:1\nT:T\nM:4/4\nL:1/8\nK:C\n|C4 E4|")

    @pytest.mark.parametrize("text", [
        "Sorry, I cannot compose that.", "", "```json\n{\"parts\": []}\n```",
        "X:1\nT:no key line\n|C4|",
    ])
    def test_rejects_anything_without_the_required_headers(self, text):
        assert ComposerCrew.extract_abc(text) is None


class TestAbcValidation:
    """These catch the faults music21 parses without complaint."""

    THREE_FOUR_SHORT = ("X:1\nT:T\nM:3/4\nL:1/8\nK:G\n"
                        "| D>E F>G | A B c d | e f g a | b c d B |"
                        " D>E F>G | A B c d | e f g a | b a g f |")

    def test_detects_bars_that_are_the_wrong_length(self):
        r = ComposerCrew.validate_abc(self.THREE_FOUR_SHORT, "3/4")
        assert r["short_bars"] > 0 and not r["ok"]
        assert any("beats" in i for i in r["issues"])

    def test_detects_a_scale_masquerading_as_a_melody(self):
        r = ComposerCrew.validate_abc(self.THREE_FOUR_SHORT, "3/4")
        assert r["longest_run"] >= 8
        assert any("scale" in i for i in r["issues"])

    def test_accepts_a_well_formed_tune(self):
        good = ("X:1\nT:T\nM:4/4\nL:1/8\nK:C\n"
                "| C2 E2 G2 E2 | F2 A2 c2 A2 | G2 E2 C2 E2 | D2 F2 A2 F2 |"
                " C2 E2 G2 E2 | A2 F2 D2 F2 | G2 B2 d2 B2 | C8 |")
        r = ComposerCrew.validate_abc(good, "4/4")
        assert r["short_bars"] == 0
        assert r["longest_run"] < 8

    def test_unparseable_input_reports_rather_than_raises(self):
        r = ComposerCrew.validate_abc("this is not music", "4/4")
        assert r["ok"] is False and r["issues"]


class TestBarRepair:
    """The model could not count bar durations in any attempt, so code does it."""

    def test_rebarring_makes_every_interior_bar_correct(self):
        from music21 import converter
        abc = ("X:1\nT:T\nM:3/4\nL:1/8\nK:G\n"
               "|\"G\" G>E D2 C4 | \"C\" B>A G4 | \"D\" E>C D2 E2 |"
               " \"G\" F>D E2 F4 | \"Bm\" D>A G4 |")
        assert ComposerCrew.validate_abc(abc, "3/4")["short_bars"] > 0
        xml, bars = ComposerCrew.repair_bar_lengths(abc, "3/4")
        assert xml and bars > 0
        ms = list(converter.parse(xml).recurse().getElementsByClass("Measure"))
        wrong = [i for i, m in enumerate(ms)
                 if i < len(ms) - 1 and abs(float(m.duration.quarterLength) - 3.0) > 0.01]
        assert not wrong, f"bars still wrong after re-barring: {wrong}"

    def test_repair_returns_empty_rather_than_raising_on_junk(self):
        xml, bars = ComposerCrew.repair_bar_lengths("not abc at all", "4/4")
        assert xml == "" and bars == 0


class TestAbcKeyMapping:
    @pytest.mark.parametrize("ui,abc", [
        ("C major", "C"), ("Bb major", "Bb"), ("A minor", "Am"),
        ("F# minor", "F#m"), ("D dorian", "DDor"),
    ])
    def test_ui_key_names_become_abc_key_fields(self, ui, abc):
        assert ComposerCrew._abc_key(ui) == abc
