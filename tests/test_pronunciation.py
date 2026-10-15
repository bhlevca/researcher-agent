"""Tests for pronunciation WER scoring utility."""

import pytest
from researcher.routes.speech import _wer, _pronunciation_score


class TestWER:
    def test_perfect_match(self):
        assert _wer("hello world", "hello world") == 0.0

    def test_case_insensitive(self):
        assert _wer("Hello World", "hello world") == 0.0

    def test_one_word_wrong(self):
        wer = _wer("hello world", "hello earth")
        assert 0 < wer <= 0.5

    def test_completely_wrong(self):
        wer = _wer("hello world", "goodbye planet")
        assert wer >= 0.5

    def test_empty_hypothesis(self):
        assert _wer("hello", "") == 1.0

    def test_empty_reference(self):
        assert _wer("", "") == 0.0

    def test_extra_words(self):
        wer = _wer("hello", "hello world")
        assert wer > 0.0

    def test_punctuation_ignored(self):
        assert _wer("hello, world!", "hello world") == 0.0


class TestPronunciationScore:
    def test_perfect(self):
        assert _pronunciation_score("hello world", "hello world") == 1.0

    def test_empty_hypothesis(self):
        assert _pronunciation_score("hello", "") == 0.0

    def test_empty_whitespace_hypothesis(self):
        assert _pronunciation_score("hello", "   ") == 0.0

    def test_partial_match(self):
        score = _pronunciation_score("the quick brown fox", "the quick brown cat")
        assert 0.5 < score < 1.0

    def test_score_in_range(self):
        score = _pronunciation_score("bonjour", "bon")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Phoneme-level scoring (T-C)
# ---------------------------------------------------------------------------


class TestPhonemeScoring:
    """WER scores RECOGNITION, which is the wrong question for pronunciation.

    It returns 0.00 both for a perfectly pronounced word spelled differently and
    for an entirely wrong word — the learner cannot tell those apart. These tests
    pin the cases that motivated moving to phoneme comparison.
    """

    @staticmethod
    def _score(ref, hyp, lang="French"):
        from researcher.routes.speech import _phoneme_score
        return _phoneme_score(ref, hyp, lang)

    def test_identical_scores_perfect(self):
        assert self._score("bonjour", "bonjour")[0] == 1.0

    def test_same_sound_different_spelling_is_not_punished(self):
        """WER gives these 0.00; they sound identical, so phoneme scoring must not."""
        from researcher.routes.speech import _pronunciation_score
        for ref, hyp in (("bonjour", "bonjoure"), ("merci", "mercy")):
            assert _pronunciation_score(ref, hyp) == 0.0, "WER should fail this"
            assert self._score(ref, hyp)[0] >= 0.95, f"{ref}/{hyp} sounds the same"

    def test_substituted_phoneme_is_penalised(self):
        """/ʒ/ -> /z/ differs by ~one articulatory feature, so pure feature
        distance rated it ~0.98. Segment accuracy is mixed in to catch it."""
        score, detail = self._score("bonjour", "bonzour")
        assert 0.75 <= score < 0.95, f"expected a visible but modest penalty, got {score}"
        assert "/ʒ/" in detail and "/z/" in detail

    def test_wrong_word_scores_below_a_phoneme_slip(self):
        slip = self._score("bonjour", "bonzour")[0]
        wrong = self._score("bonjour", "bonsoir")[0]
        assert wrong < slip, "a wrong word must score below a single-sound slip"

    def test_detail_names_the_substitution(self):
        _, detail = self._score("bonjour", "bonzour")
        assert "->" in detail, "feedback must name the actual substitution"

    @pytest.mark.parametrize("lang", ["fr", "French", "es", "Spanish", "de", "ro"])
    def test_supported_languages_use_phonemes(self, lang):
        assert self._score("casa", "caza", lang) is not None

    @pytest.mark.parametrize("lang", ["auto", "", "en", "Klingon"])
    def test_unsupported_languages_fall_back(self, lang):
        """Returning None is the signal to use WER — it must not raise."""
        assert self._score("bonjour", "bonzour", lang) is None

    def test_ipa_transcription_handles_accents(self):
        from researcher.routes.speech import _to_ipa
        ipa = _to_ipa("français", "French")
        assert ipa and any(c in ipa for c in "ɛɑ̃sfʀ")
