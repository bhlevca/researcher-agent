"""Tests for deterministic musical realisation.

These pin the properties that separate composed music from random notes. They
run without an LLM, which is the point: the musical guarantees are supposed to
come from code, not from hoping the model behaves.
"""

import pytest

from researcher.composer.realize import (
    Plan, Section, DEVELOPMENTS, realise_progression, develop_motif,
    generate_bass, find_voice_leading_problems, assemble,
)

MOTIF = [("C4", 1.0), ("E4", 0.5), ("G4", 0.5), ("E4", 1.0)]


class TestProgression:
    def test_roman_numerals_become_correct_pitches(self):
        from music21 import chord
        part = realise_progression(["I", "V7"], "C major", 2, "4/4")
        chords = list(part.recurse().getElementsByClass(chord.Chord))
        assert [p.name for p in chords[0].pitches] == ["C", "E", "G"]
        assert [p.name for p in chords[1].pitches] == ["G", "B", "D", "F"]

    def test_progression_fills_exactly_the_requested_bars(self):
        from music21 import chord
        part = realise_progression(["I", "vi", "ii", "V"], "C major", 4, "4/4")
        total = sum(c.quarterLength
                    for c in part.recurse().getElementsByClass(chord.Chord))
        assert total == pytest.approx(16.0)   # 4 bars x 4 beats

    def test_minor_key_is_honoured(self):
        from music21 import chord
        part = realise_progression(["i"], "A minor", 1, "4/4")
        c = list(part.recurse().getElementsByClass(chord.Chord))[0]
        assert [p.name for p in c.pitches] == ["A", "C", "E"]

    def test_unknown_numeral_falls_back_instead_of_crashing(self):
        part = realise_progression(["I", "!!bogus!!", "V"], "C major", 3, "4/4")
        from music21 import chord
        assert len(list(part.recurse().getElementsByClass(chord.Chord))) == 3


class TestMotifDevelopment:
    @pytest.mark.parametrize("how", DEVELOPMENTS)
    def test_every_development_returns_notes(self, how):
        out = list(develop_motif(MOTIF, how, "C major").recurse().notes)
        assert out, f"{how} produced nothing"

    def test_sequence_stays_in_key(self):
        """Chromatic transposition would put F# into C major — a wrong note."""
        for how in ("sequence_up", "sequence_down"):
            names = [n.nameWithOctave
                     for n in develop_motif(MOTIF, how, "C major").recurse().notes]
            assert not [n for n in names if "#" in n or "-" in n], \
                f"{how} left the key: {names}"

    def test_sequence_actually_moves(self):
        orig = [n.nameWithOctave for n in develop_motif(MOTIF, "original", "C major").recurse().notes]
        up = [n.nameWithOctave for n in develop_motif(MOTIF, "sequence_up", "C major").recurse().notes]
        assert orig != up

    def test_augmentation_doubles_duration(self):
        base = sum(n.quarterLength for n in develop_motif(MOTIF, "original", "C major").recurse().notes)
        aug = sum(n.quarterLength for n in develop_motif(MOTIF, "augmentation", "C major").recurse().notes)
        assert aug == pytest.approx(base * 2)

    def test_retrograde_reverses(self):
        orig = [n.nameWithOctave for n in develop_motif(MOTIF, "original", "C major").recurse().notes]
        retro = [n.nameWithOctave for n in develop_motif(MOTIF, "retrograde", "C major").recurse().notes]
        assert retro == orig[::-1]

    def test_inversion_turns_the_contour_around(self):
        """The motif rises; inverted it must fall."""
        inv = [n.pitch.midi for n in develop_motif(MOTIF, "inversion", "C major").recurse().notes]
        assert inv[1] < inv[0], "inversion did not flip direction"


class TestBass:
    def _prog(self):
        return realise_progression(["I", "vi", "ii65", "V7"], "C major", 4, "4/4")

    @pytest.mark.parametrize("genre", ["jazz", "pop", "classical"])
    def test_bass_is_generated_for_each_genre(self, genre):
        assert list(generate_bass(self._prog(), genre, "C major").recurse().notes)

    def test_genres_produce_different_bass_lines(self):
        """A walking bass and a root-on-the-beat bass must not be identical."""
        prog = self._prog()
        jazz = [n.nameWithOctave for n in generate_bass(prog, "jazz", "C major").recurse().notes]
        pop = [n.nameWithOctave for n in generate_bass(prog, "pop", "C major").recurse().notes]
        assert jazz != pop

    def test_jazz_bass_walks_in_quarter_notes(self):
        notes = list(generate_bass(self._prog(), "jazz", "C major").recurse().notes)
        assert all(n.quarterLength == 1.0 for n in notes)


class TestVoiceLeading:
    def test_parallel_fifths_are_detected(self):
        from music21 import note, stream
        a, b = stream.Part(), stream.Part()
        for n in ("C4", "D4"):
            a.append(note.Note(n))
        for n in ("G4", "A4"):
            b.append(note.Note(n))
        assert any("fifth" in p for p in find_voice_leading_problems(a, b))

    def test_clean_motion_reports_nothing(self):
        from music21 import note, stream
        a, b = stream.Part(), stream.Part()
        for n in ("C4", "D4"):
            a.append(note.Note(n))
        for n in ("E4", "F4"):
            b.append(note.Note(n))
        assert find_voice_leading_problems(a, b) == []


class TestAssembly:
    def _plan(self, genre="jazz"):
        return Plan(
            key="C major", time_signature="4/4", tempo=120, genre=genre,
            motif=MOTIF, form=["A", "A", "B", "A"],
            sections={
                "A": Section("A", ["I", "vi", "ii65", "V7"], 4, "original"),
                "B": Section("B", ["IV", "IV", "I", "V7"], 4, "inversion"),
            },
        )

    def test_melody_fills_every_bar(self):
        """A motif stated once leaves the section mostly empty."""
        score = assemble(self._plan())
        mel = list(list(score.parts)[0].recurse().notes)
        beats = sum(n.quarterLength for n in mel)
        assert beats == pytest.approx(64.0)   # 16 bars x 4 beats

    def test_parts_are_not_identical(self):
        """Copying the melody into the bass is not an arrangement."""
        score = assemble(self._plan())
        parts = list(score.parts)
        a = [n.nameWithOctave for n in parts[0].recurse().notes]
        b = [n.nameWithOctave for n in parts[1].recurse().notes]
        assert a[:min(len(a), len(b))] != b[:min(len(a), len(b))]

    def test_no_long_scale_runs(self):
        """A run straight up the scale is the signature of generated filler."""
        score = assemble(self._plan())
        midis = [n.pitch.midi for n in list(score.parts)[0].recurse().notes]
        run = longest = 1
        for i in range(1, len(midis)):
            step = midis[i] - midis[i - 1]
            run = run + 1 if 0 < step <= 2 else 1
            longest = max(longest, run)
        assert longest <= 6, f"scale run of {longest} notes"

    def test_exports_to_musicxml_and_midi(self):
        import pathlib
        score = assemble(self._plan())
        xml = pathlib.Path(score.write("musicxml")).read_text()
        assert "<score-partwise" in xml
        assert pathlib.Path(score.write("midi")).read_bytes()[:4] == b"MThd"

    def test_form_repeats_reuse_the_same_material(self):
        """AABA means the As match — that is what makes form audible."""
        score = assemble(self._plan())
        mel = [n.nameWithOctave for n in list(score.parts)[0].recurse().notes]
        quarter = len(mel) // 4
        assert mel[:quarter] == mel[quarter:quarter * 2]


class TestPolish:
    """Joining specialist material to the deterministic guarantees.

    The specialist writes idiomatic vocabulary but still marches up and down
    the scale, and hands back a bare single line. These pin both repairs.
    """

    SCALE_TUNE = (
        'X:1\nL:1/4\nM:4/4\nK:C\n'
        '"C" c B A G |"F" F E D C |"G7" C D E F |"C" G3 z |'
    )

    @staticmethod
    def _longest_run(part):
        from music21 import note as m21note
        ns = [n.pitch.midi for n in part.recurse().getElementsByClass(m21note.Note)]
        best = run = 1
        prev = 0
        for i in range(1, len(ns)):
            d = ns[i] - ns[i - 1]
            dirn = (d > 0) - (d < 0)
            run = run + 1 if (0 < abs(d) <= 2 and dirn == prev) else (2 if 0 < abs(d) <= 2 else 1)
            prev = dirn
            best = max(best, run)
        return best

    def test_scale_runs_are_broken(self):
        from researcher.composer.realize import polish
        score, report = polish(self.SCALE_TUNE, "jazz", "C major", 120)
        assert report["scale_runs_fixed"] > 0
        assert self._longest_run(list(score.parts)[0]) <= 6

    def test_bare_melody_gains_a_bass(self):
        from researcher.composer.realize import polish
        score, report = polish(self.SCALE_TUNE, "jazz", "C major", 120)
        assert report["bass_added"] is True
        assert report["parts"] == 2

    def test_added_bass_is_not_a_copy_of_the_melody(self):
        from music21 import note as m21note
        from researcher.composer.realize import polish
        score, _ = polish(self.SCALE_TUNE, "jazz", "C major", 120)
        parts = list(score.parts)
        a = [n.nameWithOctave for n in parts[0].recurse().getElementsByClass(m21note.Note)]
        b = [n.nameWithOctave for n in parts[1].recurse().getElementsByClass(m21note.Note)]
        assert a[:min(len(a), len(b))] != b[:min(len(a), len(b))]

    def test_chord_symbols_are_not_counted_as_melody_notes(self):
        """Regression: ChordSymbol objects appear in .notes and silently split
        every run in two, so the detector found nothing on an obvious scale."""
        from music21 import converter, note as m21note
        s = converter.parse(self.SCALE_TUNE, format="abc")
        part = list(s.parts)[0]
        assert len(list(part.recurse().notes)) > \
               len(list(part.recurse().getElementsByClass(m21note.Note)))

    def test_polish_survives_unparseable_input(self):
        from researcher.composer.realize import polish
        score, report = polish("this is not ABC at all", "jazz", "C major", 120)
        assert score is None
        assert report["scale_runs_fixed"] == 0

    def test_tune_without_chord_symbols_gets_no_bass(self):
        from researcher.composer.realize import polish
        plain = 'X:1\nL:1/4\nM:4/4\nK:C\nC D E F | G A B c |'
        score, report = polish(plain, "jazz", "C major", 120)
        assert report["bass_added"] is False


class TestLongForm:
    """Reaching 2-3 minutes without producing 3 minutes of mush."""

    def test_plan_hits_the_requested_duration(self):
        from researcher.composer.realize import build_long_plan, estimate_duration
        for minutes in (2.0, 2.5, 3.0):
            plan = build_long_plan("jazz", "C major", "4/4", 120, minutes)
            got = estimate_duration(plan) / 60
            assert abs(got - minutes) < 0.5, f"asked {minutes}min, planned {got:.1f}min"

    def test_material_recurs_rather_than_being_all_new(self):
        """Length is only listenable if the listener recognises things coming back."""
        from researcher.composer.realize import build_long_plan
        plan = build_long_plan("jazz", "C major", "4/4", 120, 3.0)
        progressions = [tuple(plan.sections[n].progression) for n in plan.form]
        assert len(set(progressions)) < len(progressions), "no material recurs"

    def test_harmony_is_richer_than_plain_triads(self):
        """Sevenths, ninths, secondary dominants — not I-IV-V."""
        from researcher.composer.realize import build_long_plan
        plan = build_long_plan("jazz", "C major", "4/4", 120, 2.5)
        figures = {f for n in plan.form for f in plan.sections[n].progression}
        extended = [f for f in figures if any(c in f for c in ("7", "9", "13", "/"))]
        assert extended, f"only plain triads: {figures}"

    def test_every_figure_actually_realises(self):
        """A Roman numeral the library cannot parse would silently become I."""
        from music21 import roman, key as m21key
        from researcher.composer.realize import GENRE_PROGRESSIONS
        k = m21key.Key("C")
        bad = []
        for genre, sections in GENRE_PROGRESSIONS.items():
            for name, figures in sections.items():
                for fig in figures:
                    try:
                        roman.RomanNumeral(fig, k)
                    except Exception:
                        bad.append(f"{genre}/{name}/{fig}")
        assert not bad, f"unrealisable figures: {bad}"

    def test_long_piece_assembles_and_stays_clean(self):
        from music21 import note as m21note
        from researcher.composer.realize import build_long_plan, assemble
        plan = build_long_plan("classical", "C major", "4/4", 120, 2.5)
        score = assemble(plan)
        mel = [n for n in list(score.parts)[0].recurse().getElementsByClass(m21note.Note)]
        assert len(mel) > 200, "too few notes for a 2.5 minute piece"
        midis = [n.pitch.midi for n in mel]
        run = best = 1
        prev = 0
        for i in range(1, len(midis)):
            d = midis[i] - midis[i - 1]
            dirn = (d > 0) - (d < 0)
            run = run + 1 if (0 < abs(d) <= 2 and dirn == prev) else (2 if 0 < abs(d) <= 2 else 1)
            prev = dirn
            best = max(best, run)
        assert best <= 6, f"scale run of {best} in the long form"


class TestSanitizeAbc:
    """Repairs for the notation errors the specialist actually makes."""

    @staticmethod
    def _clean(abc):
        from researcher.composer.specialist import sanitize_abc
        return sanitize_abc(abc)

    def test_inline_metre_change_is_removed(self):
        """[M:3/4] mid-line leaves the bar wrong on both sides of it."""
        out, notes = self._clean('X:1\nM:4/4\nK:C\n[V:1] B>A z8 |:"C" [M:3/4] G>E A z|')
        assert "[M:3/4]" not in out
        assert any("metre" in n for n in notes)

    def test_broken_rhythm_on_a_rest_is_removed(self):
        """'C4 >z' is meaningless — there is nothing to lengthen."""
        out, notes = self._clean("X:1\nM:4/4\nK:C\nc2 B<A G2 C4 >z|")
        assert ">z" not in out.replace(" ", "")
        assert any("rest" in n for n in notes)

    def test_unbalanced_slur_is_closed(self):
        out, notes = self._clean("X:1\nM:4/4\nK:C\n(D (E F) G |")
        assert out.count("(") == out.count(")")

    def test_header_lines_are_left_alone(self):
        src = "X:1\nT:Tune\nM:4/4\nL:1/8\nK:C\nCDEF|"
        out, _ = self._clean(src)
        for field in ("X:1", "T:Tune", "M:4/4", "L:1/8", "K:C"):
            assert field in out

    def test_clean_input_is_unchanged(self):
        src = 'X:1\nM:4/4\nL:1/8\nK:C\n"C" C2 E2 G2 E2 |'
        out, notes = self._clean(src)
        assert notes == []
        assert "C2 E2 G2 E2" in out
