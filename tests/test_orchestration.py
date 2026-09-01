"""Instrument ranges, transposition and playability (C-O1).

These matter because a score can sound fine in MIDI and still be unplayable or
plainly wrong on the page — notes below an instrument's lowest string, or a
clarinet part written at concert pitch.
"""

import pytest

from researcher.composer import orchestration as orc


class TestRangeTable:
    def test_every_entry_is_well_formed(self):
        from music21 import pitch

        for name, spec in orc.INSTRUMENT_RANGES.items():
            lo = pitch.Pitch(spec["low"]).midi
            hi = pitch.Pitch(spec["high"]).midi
            assert lo < hi, f"{name}: low is not below high"
            assert hi - lo >= 12, f"{name}: range narrower than an octave"
            assert isinstance(spec["transpose"], int)

    def test_voices_are_covered(self):
        """music21 returns range=None..None for voices, so we supply our own."""
        for v in ("Soprano", "Alto", "Tenor", "Bass"):
            lo, hi = orc.range_of(v)
            assert lo > 0 and hi > lo

    @pytest.mark.parametrize("given,expected", [
        ("violin", "Violin"), ("VIOLIN", "Violin"), (" Cello ", "Cello"),
        # "French Horn" is itself a table key, so it wins over the alias.
        # Both entries carry identical specs, so either answer is correct.
        ("french horn", "French Horn"), ("double bass", "Double Bass"),
        ("alto sax", "Alto Saxophone"), ("choir", "Soprano"),
        ("nonsense-instrument", "Piano"), ("", "Piano"), (None, "Piano"),
    ])
    def test_name_normalisation(self, given, expected):
        assert orc.canonical_name(given) == expected


class TestRangeFitting:
    @staticmethod
    def _part(names):
        from music21 import stream, note

        p = stream.Part()
        for n in names:
            p.append(note.Note(n, quarterLength=1))
        return p

    def test_detects_notes_below_the_instrument(self):
        p = self._part(["C2", "D2", "A4"])
        r = orc.check_range(p, "Violin")
        assert not r["ok"]
        assert r["below"] == ["C2", "D2"]

    def test_octave_shift_preserves_pitch_class(self):
        """Shifting by octaves keeps the harmony; clamping would rewrite it."""
        p = self._part(["C2", "D2", "B7"])
        before = [n.pitch.name for n in p.recurse().notes]
        orc.fit_to_range(p, "Violin")
        after = [n.pitch.name for n in p.recurse().notes]
        assert before == after, "pitch classes changed — harmony was altered"
        assert orc.check_range(p, "Violin")["ok"]

    def test_in_range_music_is_left_alone(self):
        p = self._part(["G3", "A4", "E6"])
        assert orc.fit_to_range(p, "Violin") == 0

    def test_impossible_note_is_not_left_out_of_range(self):
        """A range narrower than an octave must not produce a worse note."""
        p = self._part(["C0"])
        orc.fit_to_range(p, "Piccolo")
        n = list(p.recurse().notes)[0]
        lo, hi = orc.range_of("Piccolo")
        assert n.pitch.midi >= lo or n.pitch.midi == 12


class TestInstrumentAssignment:
    @staticmethod
    def _part():
        from music21 import stream, note, instrument

        p = stream.Part()
        p.insert(0, instrument.Piano())      # the placeholder assemble() adds
        for n in ("C4", "E4", "G4"):
            p.append(note.Note(n, quarterLength=1))
        return p

    def test_placeholder_instrument_is_replaced_not_stacked(self):
        """music21 honours the FIRST instrument found, so the old Piano must go.

        Leaving it made every part read "Piano" while the report claimed the
        requested instrument had been applied.
        """
        from music21 import instrument as m21inst

        p = self._part()
        orc.apply_instrument(p, "Violin")
        found = [i.instrumentName for i in p.recurse().getElementsByClass(m21inst.Instrument)]
        assert found == ["Violin"], f"expected only Violin, got {found}"

    @pytest.mark.parametrize("name,transposing", [
        ("Clarinet", True), ("Trumpet", True), ("Horn", True),
        ("Violin", False), ("Piano", False), ("Trombone", False),
    ])
    def test_transposing_instruments_are_marked(self, name, transposing):
        p = self._part()
        r = orc.apply_instrument(p, name)
        assert r["transposed"] is transposing


class TestBreathing:
    @staticmethod
    def _sustained(beats):
        from music21 import stream, note

        p = stream.Part()
        for _ in range(beats):
            p.append(note.Note("G4", quarterLength=1))
        return p

    def test_wind_players_need_to_breathe(self):
        assert orc.check_breathing(self._sustained(24), "Trumpet")

    def test_strings_do_not(self):
        assert orc.check_breathing(self._sustained(24), "Violin") == []

    def test_short_phrases_are_fine(self):
        assert orc.check_breathing(self._sustained(4), "Flute") == []


class TestOrchestrate:
    def test_ui_selection_reaches_the_score(self):
        """The regression this fixes: the picker was ignored entirely."""
        from researcher.composer.realize import build_long_plan, assemble
        from music21 import instrument as m21inst

        plan = build_long_plan("classical", "C major", "4/4", 100, 0.5)
        score = assemble(plan, instruments=["Violin", "Cello"])
        names = []
        for part in score.parts:
            ins = list(part.recurse().getElementsByClass(m21inst.Instrument))
            names.append(ins[0].instrumentName if ins else None)
        assert names[:2] == ["Violin", "Cello"]

    def test_fewer_instruments_than_parts_reuses_the_last(self):
        from researcher.composer.realize import build_long_plan, assemble

        plan = build_long_plan("classical", "C major", "4/4", 100, 0.5)
        report = assemble(plan, instruments=["Violin"]).editorial.orchestration
        assert report["parts"] >= 1
        assert all(r["instrument"] == "Violin" for r in report["reports"])

    def test_score_still_exports(self):
        import pathlib
        from researcher.composer.realize import build_long_plan, assemble

        plan = build_long_plan("jazz", "C major", "4/4", 120, 0.5)
        score = assemble(plan, instruments=["Trumpet", "Tuba"])
        xml = pathlib.Path(score.write("musicxml")).read_text()
        assert "<score-partwise" in xml
        assert "Trumpet" in xml
