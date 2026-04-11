"""Tests for the MusicXML builder (JSON → MusicXML)."""

import json
from xml.etree import ElementTree as ET

from researcher.composer.musicxml_builder import build_musicxml


# ── Test data ────────────────────────────────────────────────────────

SIMPLE_SCORE = {
    "title": "Test Piece",
    "composer": "Test Composer",
    "parts": [
        {
            "name": "Violin",
            "measures": [
                {
                    "notes": [
                        {"pitch": "A4", "duration": "quarter"},
                        {"pitch": "B4", "duration": "quarter"},
                        {"pitch": "C5", "duration": "half"},
                    ]
                },
                {
                    "notes": [
                        {"pitch": "D5", "duration": "whole"},
                    ]
                },
            ],
        }
    ],
}

PIANO_SCORE = {
    "title": "Piano Test",
    "composer": "Test",
    "parts": [
        {
            "name": "Piano",
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
                {
                    "notes": [
                        {"pitch": "F4", "duration": "quarter", "staff": 1},
                        {"pitch": "A4", "duration": "quarter", "staff": 1},
                        {"pitch": "rest", "duration": "half", "staff": 1},
                        {"pitch": "F3", "duration": "half", "staff": 2},
                        {"pitch": "C3", "duration": "half", "staff": 2},
                    ]
                },
            ],
        }
    ],
}

CHORD_SCORE = {
    "title": "Chord Test",
    "composer": "Test",
    "parts": [
        {
            "name": "Piano",
            "measures": [
                {
                    "notes": [
                        {"pitch": "C4", "duration": "whole", "staff": 1},
                        {"pitch": "E4", "duration": "whole", "staff": 1, "chord": True},
                        {"pitch": "G4", "duration": "whole", "staff": 1, "chord": True},
                        {"pitch": "C3", "duration": "whole", "staff": 2},
                    ]
                },
            ],
        }
    ],
}

REST_SCORE = {
    "title": "Rest Test",
    "composer": "Test",
    "parts": [
        {
            "name": "Flute",
            "measures": [
                {
                    "notes": [
                        {"pitch": "rest", "duration": "whole"},
                    ]
                },
            ],
        }
    ],
}

MULTI_INSTRUMENT = {
    "title": "Duo",
    "composer": "Test",
    "parts": [
        {
            "name": "Violin",
            "measures": [
                {"notes": [{"pitch": "A4", "duration": "whole"}]},
            ],
        },
        {
            "name": "Cello",
            "measures": [
                {"notes": [{"pitch": "A2", "duration": "whole"}]},
            ],
        },
    ],
}


def _parse(xml_str: str) -> ET.Element:
    """Strip DOCTYPE and parse XML."""
    body = xml_str
    if '<!DOCTYPE' in body:
        idx = body.index('<!DOCTYPE')
        end = body.index('>', idx) + 1
        body = body[:idx] + body[end:]
    return ET.fromstring(body)


# ── Tests ────────────────────────────────────────────────────────────


def test_simple_score_parses():
    """Simple violin score should produce valid XML."""
    xml = build_musicxml(SIMPLE_SCORE)
    root = _parse(xml)
    assert root.tag == 'score-partwise'


def test_xml_declaration():
    """Output should have XML declaration and DOCTYPE."""
    xml = build_musicxml(SIMPLE_SCORE)
    assert xml.startswith('<?xml version="1.0"')
    assert '<!DOCTYPE score-partwise' in xml


def test_title_and_composer():
    """Title and composer should appear in the output."""
    xml = build_musicxml(SIMPLE_SCORE)
    root = _parse(xml)
    assert root.find('.//work-title').text == 'Test Piece'
    assert root.find('.//creator').text == 'Test Composer'


def test_part_list():
    """Part list should have the right instrument."""
    xml = build_musicxml(SIMPLE_SCORE)
    root = _parse(xml)
    sp = root.find('.//score-part')
    assert sp is not None
    assert sp.find('part-name').text == 'Violin'


def test_measure_count():
    """Should have correct number of measures."""
    xml = build_musicxml(SIMPLE_SCORE)
    root = _parse(xml)
    measures = list(root.iter('measure'))
    assert len(measures) == 2


def test_notes_have_required_children():
    """Every note should have duration, voice, type."""
    xml = build_musicxml(SIMPLE_SCORE)
    root = _parse(xml)
    for note in root.iter('note'):
        assert note.find('duration') is not None
        assert note.find('voice') is not None
        assert note.find('type') is not None


def test_note_pitch():
    """First note should be A4."""
    xml = build_musicxml(SIMPLE_SCORE)
    root = _parse(xml)
    notes = list(root.iter('note'))
    first = notes[0]
    pitch = first.find('pitch')
    assert pitch is not None
    assert pitch.find('step').text == 'A'
    assert pitch.find('octave').text == '4'


def test_key_signature():
    """G major should have fifths=1."""
    xml = build_musicxml(SIMPLE_SCORE, key="G major")
    root = _parse(xml)
    fifths = root.find('.//fifths')
    assert fifths.text == '1'


def test_time_signature():
    """3/4 time should appear in attributes."""
    xml = build_musicxml(SIMPLE_SCORE, time_signature="3/4")
    root = _parse(xml)
    beats = root.find('.//beats')
    bt = root.find('.//beat-type')
    assert beats.text == '3'
    assert bt.text == '4'


def test_tempo_marking():
    """Tempo should appear as direction + sound."""
    xml = build_musicxml(SIMPLE_SCORE, tempo=90)
    root = _parse(xml)
    pm = root.find('.//per-minute')
    assert pm.text == '90'
    sound = root.find('.//sound')
    assert sound.get('tempo') == '90'


def test_divisions():
    """Divisions should be 4."""
    xml = build_musicxml(SIMPLE_SCORE)
    root = _parse(xml)
    div = root.find('.//divisions')
    assert div.text == '4'


def test_attributes_only_in_first_measure():
    """Attributes should only appear in measure 1."""
    xml = build_musicxml(SIMPLE_SCORE)
    root = _parse(xml)
    measures = list(root.iter('measure'))
    assert measures[0].find('attributes') is not None
    assert measures[1].find('attributes') is None


def test_final_barline():
    """Last measure should have a final barline."""
    xml = build_musicxml(SIMPLE_SCORE)
    root = _parse(xml)
    measures = list(root.iter('measure'))
    barline = measures[-1].find('barline')
    assert barline is not None
    assert barline.find('bar-style').text == 'light-heavy'


# ── Piano (Grand Staff) tests ───────────────────────────────────────

def test_piano_grand_staff():
    """Piano should have staves=2 and two clefs."""
    xml = build_musicxml(PIANO_SCORE)
    root = _parse(xml)
    staves = root.find('.//staves')
    assert staves is not None
    assert staves.text == '2'
    clefs = root.findall('.//clef')
    assert len(clefs) == 2


def test_piano_backup():
    """Piano measures should have <backup> between staves."""
    xml = build_musicxml(PIANO_SCORE)
    root = _parse(xml)
    for measure in root.iter('measure'):
        if measure.find('.//note') is not None:
            backup = measure.find('backup')
            assert backup is not None, f"Measure {measure.get('number')} missing backup"


def test_piano_staff_assignment():
    """Piano notes should have <staff> elements."""
    xml = build_musicxml(PIANO_SCORE)
    root = _parse(xml)
    for note in root.iter('note'):
        staff = note.find('staff')
        assert staff is not None
        assert staff.text in ('1', '2')


def test_piano_voice_assignment():
    """Staff 1 = voice 1, staff 2 = voice 2."""
    xml = build_musicxml(PIANO_SCORE)
    root = _parse(xml)
    for note in root.iter('note'):
        staff = note.find('staff')
        voice = note.find('voice')
        if staff.text == '1':
            assert voice.text == '1'
        elif staff.text == '2':
            assert voice.text == '2'


def test_piano_note_order():
    """Notes should be: staff-1 notes, backup, staff-2 notes."""
    xml = build_musicxml(PIANO_SCORE)
    root = _parse(xml)
    for measure in root.iter('measure'):
        children = [c.tag for c in measure]
        if 'backup' not in children:
            continue
        backup_idx = children.index('backup')
        # All notes before backup should be staff 1
        for c in measure:
            if c.tag == 'backup':
                break
            if c.tag == 'note':
                s = c.find('staff')
                if s is not None:
                    assert s.text == '1'


# ── Chord tests ──────────────────────────────────────────────────────

def test_chord_flag():
    """Notes with chord=true should have <chord/> element."""
    xml = build_musicxml(CHORD_SCORE)
    root = _parse(xml)
    notes = [n for n in root.iter('note') if n.find('chord') is not None]
    assert len(notes) == 2  # E4 and G4 are chords


# ── Rest tests ───────────────────────────────────────────────────────

def test_rest_note():
    """Rest should have <rest/> element."""
    xml = build_musicxml(REST_SCORE)
    root = _parse(xml)
    notes = list(root.iter('note'))
    assert len(notes) >= 1
    rest = notes[0].find('rest')
    assert rest is not None


# ── Multi-instrument tests ───────────────────────────────────────────

def test_multi_instrument_parts():
    """Multiple instruments should create multiple parts."""
    xml = build_musicxml(MULTI_INSTRUMENT)
    root = _parse(xml)
    parts = list(root.iter('part'))
    assert len(parts) == 2


def test_multi_instrument_clefs():
    """Violin=treble, Cello=bass."""
    xml = build_musicxml(MULTI_INSTRUMENT)
    root = _parse(xml)
    parts = list(root.iter('part'))
    # Violin part
    v_clef = parts[0].find('.//clef/sign')
    assert v_clef.text == 'G'
    # Cello part
    c_clef = parts[1].find('.//clef/sign')
    assert c_clef.text == 'F'


def test_multi_instrument_midi():
    """Violin=41, Cello=43."""
    xml = build_musicxml(MULTI_INSTRUMENT)
    root = _parse(xml)
    programs = [mp.text for mp in root.findall('.//midi-program')]
    assert '41' in programs  # Violin
    assert '43' in programs  # Cello


# ── Sharp/flat tests ────────────────────────────────────────────────

def test_sharp_note():
    """F#5 should produce alter=1."""
    score = {
        "title": "T", "composer": "C",
        "parts": [{"name": "Violin", "measures": [
            {"notes": [{"pitch": "F#5", "duration": "whole"}]}
        ]}],
    }
    xml = build_musicxml(score)
    root = _parse(xml)
    alter = root.find('.//alter')
    assert alter is not None
    assert alter.text == '1'


def test_flat_note():
    """Bb3 should produce alter=-1."""
    score = {
        "title": "T", "composer": "C",
        "parts": [{"name": "Violin", "measures": [
            {"notes": [{"pitch": "Bb3", "duration": "whole"}]}
        ]}],
    }
    xml = build_musicxml(score)
    root = _parse(xml)
    alter = root.find('.//alter')
    assert alter is not None
    assert alter.text == '-1'


# ── Measure padding tests ───────────────────────────────────────────

def test_short_measure_padded():
    """A measure with only 2 quarter notes in 4/4 should be padded with rests."""
    score = {
        "title": "T", "composer": "C",
        "parts": [{"name": "Flute", "measures": [
            {"notes": [
                {"pitch": "C5", "duration": "quarter"},
                {"pitch": "D5", "duration": "quarter"},
            ]}
        ]}],
    }
    xml = build_musicxml(score)
    root = _parse(xml)
    notes = list(root.iter('note'))
    # Should have 2 real notes + rest(s) to fill measure
    total_dur = sum(int(n.find('duration').text) for n in notes
                    if n.find('chord') is None)
    assert total_dur == 16  # 4 quarter notes * 4 divisions


# ── Dotted note tests ───────────────────────────────────────────────

def test_dotted_half():
    """Dotted half should have duration=12 and <dot/> element."""
    score = {
        "title": "T", "composer": "C",
        "parts": [{"name": "Violin", "measures": [
            {"notes": [
                {"pitch": "C5", "duration": "dotted half"},
                {"pitch": "D5", "duration": "quarter"},
            ]}
        ]}],
    }
    xml = build_musicxml(score)
    root = _parse(xml)
    notes = list(root.iter('note'))
    first = notes[0]
    assert first.find('duration').text == '12'
    assert first.find('dot') is not None
    assert first.find('type').text == 'half'


# ── Crew JSON extraction test ───────────────────────────────────────

def test_crew_extract_json():
    """ComposerCrew._extract_json_score should parse JSON and build XML."""
    from researcher.composer.crew import ComposerCrew

    response = '''Here's a lullaby:

```json
{
  "title": "Lullaby",
  "composer": "AI",
  "parts": [
    {
      "name": "Piano",
      "measures": [
        {
          "notes": [
            {"pitch": "C4", "duration": "whole", "staff": 1},
            {"pitch": "C3", "duration": "whole", "staff": 2}
          ]
        }
      ]
    }
  ]
}
```

Enjoy!'''

    xml = ComposerCrew._extract_json_score(response)
    assert xml is not None
    assert '<?xml' in xml
    assert '<score-partwise' in xml


def test_crew_extract_json_with_trailing_comma():
    """Should handle trailing commas in JSON (common LLM error)."""
    from researcher.composer.crew import ComposerCrew

    response = '''```json
{
  "title": "Test",
  "composer": "AI",
  "parts": [
    {
      "name": "Violin",
      "measures": [
        {
          "notes": [
            {"pitch": "C4", "duration": "whole"},
          ]
        },
      ]
    },
  ]
}
```'''

    xml = ComposerCrew._extract_json_score(response)
    assert xml is not None
    assert '<score-partwise' in xml


def test_crew_extract_musicxml_prefers_json():
    """extract_musicxml should prefer JSON over XML when both present."""
    from researcher.composer.crew import ComposerCrew
    crew = ComposerCrew.__new__(ComposerCrew)  # skip __init__

    response = '''```json
{"title": "T", "composer": "C", "parts": [{"name": "Violin", "measures": [{"notes": [{"pitch": "A4", "duration": "whole"}]}]}]}
```

```xml
<score-partwise><part><measure><note><pitch><step>B</step><octave>4</octave></pitch></note></measure></part></score-partwise>
```'''

    xml = crew.extract_musicxml(response)
    assert xml is not None
    # Should contain A4 from JSON, not B4 from XML
    root = _parse(xml)
    step = root.find('.//step')
    assert step.text == 'A'


def test_crew_repair_hallucinated_text():
    """_repair_json should strip hallucinated text lines and still parse."""
    from researcher.composer.crew import ComposerCrew

    # Simulates LLM inserting stray `" feel,` and a lone `"` in JSON
    bad_json = '''{
  "title": "Test",
  "composer": "AI",
  "parts": [
    {
      "name": "Piano",
      "measures": [
        {
          "notes": [
            {"pitch": "C5", "duration": "quarter", "staff": 1},
            {"pitch": "G5", "duration": "quarter",
" feel,
"staff": 1},
            {"pitch": "C3", "duration": "quarter", "staff": 2}
          ]
        },
        {
          "notes": [
            {"pitch": "D5",
"
"duration": "quarter", "staff": 1}
          ]
        }
      ]
    }
  ]
}'''
    repaired = ComposerCrew._repair_json(bad_json)
    import json
    score = json.loads(repaired)
    assert score["title"] == "Test"
    assert len(score["parts"][0]["measures"]) == 2


def test_crew_repair_truncated_json():
    """_repair_json should close truncated JSON with missing brackets."""
    from researcher.composer.crew import ComposerCrew

    # JSON truncated mid-measure (missing closing brackets)
    truncated = '''{
  "title": "Truncated",
  "composer": "AI",
  "parts": [
    {
      "name": "Piano",
      "measures": [
        {
          "notes": [
            {"pitch": "C4", "duration": "whole", "staff": 1}
          ]
        },
        {
          "notes": [
            {"pitch": "D4", "duration": "quarter", "staff": 1},
            {"pitch": "E4", "duration": "quarter",'''
    repaired = ComposerCrew._repair_json(truncated)
    import json
    score = json.loads(repaired)
    assert score["title"] == "Truncated"
    # First complete measure should be intact
    assert len(score["parts"][0]["measures"]) >= 1


def test_crew_repair_intraline_junk():
    """_repair_json should strip junk words spliced after { on the same line."""
    from researcher.composer.crew import ComposerCrew

    bad_json = '''{
  "title": "T",
  "composer": "C",
  "parts": [
    {
      "name": "Piano",
      "measures": [
        {
          "notes": [
            {"pitch": "C4", "duration": "whole", "staff": 1}
          ]
        },
        {
          "notes": [
            {"pitch": "D4", "duration": "quarter", "staff": 1}
          ]
        },
        {
          "notes": [
            {"pitch": "E4", "duration": "quarter", "staff": 1}
          ]
        },
        { piek
          "notes": [
            {"pitch": "F4", "duration": "quarter", "staff": 1}
          ]
        }
      ]
    }
  ]
}'''
    repaired = ComposerCrew._repair_json(bad_json)
    import json
    score = json.loads(repaired)
    assert score["title"] == "T"
    assert len(score["parts"][0]["measures"]) == 4


def test_crew_extract_bare_json():
    """_extract_json_score should handle bare (unfenced) JSON output."""
    from researcher.composer.crew import ComposerCrew

    response = '''Final Answer:
{
  "title": "Bare",
  "composer": "AI",
  "parts": [
    {
      "name": "Piano",
      "measures": [
        {
          "notes": [
            {"pitch": "A4", "duration": "whole", "staff": 1},
            {"pitch": "A3", "duration": "whole", "staff": 2}
          ]
        }
      ]
    }
  ]
}'''
    xml = ComposerCrew._extract_json_score(response)
    assert xml is not None
    assert '<score-partwise' in xml


# ── Split-part merging ───────────────────────────────────────────────

SPLIT_PIANO_SCORE = {
    "title": "Split Piano",
    "composer": "Test",
    "parts": [
        {
            "name": "Piano - Treble",
            "staff": 1,
            "measures": [
                {"notes": [{"pitch": "C5", "duration": "quarter"},
                           {"pitch": "E5", "duration": "quarter"},
                           {"pitch": "G5", "duration": "half"}]},
            ],
        },
        {
            "name": "Piano - Bass",
            "staff": 2,
            "measures": [
                {"notes": [{"pitch": "C3", "duration": "whole"}]},
            ],
        },
    ],
}


def test_split_parts_merged_into_one():
    """Split treble/bass parts should merge into a single grand-staff part."""
    xml = build_musicxml(SPLIT_PIANO_SCORE)
    root = _parse(xml)
    parts = root.findall('part')
    assert len(parts) == 1, f"Expected 1 merged part, got {len(parts)}"


def test_split_parts_has_grand_staff():
    """Merged part must have staves=2."""
    xml = build_musicxml(SPLIT_PIANO_SCORE)
    root = _parse(xml)
    staves = root.find('.//staves')
    assert staves is not None
    assert staves.text == '2'


def test_split_parts_contains_both_staffs():
    """Merged part should contain notes from both treble and bass."""
    xml = build_musicxml(SPLIT_PIANO_SCORE)
    root = _parse(xml)
    notes = root.findall('.//note')
    # 3 treble notes + 1 bass note (+ possible padding rests)
    assert len(notes) >= 4
    staffs = {n.find('staff').text for n in notes if n.find('staff') is not None}
    assert '1' in staffs
    assert '2' in staffs


def test_split_parts_lowercase_pitch():
    """Lowercase pitch like 'c5' should be parsed correctly."""
    score = {
        "title": "Lowercase Test",
        "composer": "Test",
        "parts": [{"name": "Violin", "measures": [
            {"notes": [{"pitch": "c5", "duration": "whole"}]}
        ]}],
    }
    xml = build_musicxml(score)
    root = _parse(xml)
    step = root.find('.//step')
    assert step is not None
    assert step.text == 'C'
    octave = root.find('.//octave')
    assert octave.text == '5'
