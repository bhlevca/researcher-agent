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
