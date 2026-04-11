"""Tests for pure utility functions in main.py."""

import re
import queue
from researcher.config import _clean_line, _QueueWriter, _SESSION_ID_RE


class TestCleanLine:
    def test_strips_ansi_codes(self):
        assert _clean_line("\x1b[32mHello\x1b[0m") == "Hello"

    def test_strips_box_drawing(self):
        assert _clean_line("╭──── Agent ────╮") == "Agent"

    def test_strips_combined(self):
        line = "\x1b[1m╭── \x1b[33mthinking\x1b[0m ──╮"
        result = _clean_line(line)
        assert "thinking" in result
        assert "╭" not in result
        assert "\x1b" not in result

    def test_plain_text_unchanged(self):
        assert _clean_line("Hello world") == "Hello world"

    def test_empty_input(self):
        assert _clean_line("") == ""

    def test_only_box_chars_returns_empty(self):
        assert _clean_line("╭╮╰╯│─") == ""

    def test_osc_escape(self):
        # OSC sequence: \x1b]...\x07
        assert _clean_line("\x1b]0;title\x07content") == "content"


class TestQueueWriter:
    def test_write_lines_go_to_queue(self):
        q = queue.Queue()
        w = _QueueWriter(q)
        w.write("line one\nline two\n")
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items == ["line one", "line two"]

    def test_flush_sends_partial(self):
        q = queue.Queue()
        w = _QueueWriter(q)
        w.write("partial")
        assert q.empty()  # no newline yet
        w.flush()
        assert q.get_nowait() == "partial"

    def test_empty_lines_skipped(self):
        q = queue.Queue()
        w = _QueueWriter(q)
        w.write("hello\n\n\nworld\n")
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items == ["hello", "world"]

    def test_ansi_cleaned(self):
        q = queue.Queue()
        w = _QueueWriter(q)
        w.write("\x1b[31mred text\x1b[0m\n")
        assert q.get_nowait() == "red text"

    def test_isatty_false(self):
        q = queue.Queue()
        w = _QueueWriter(q)
        assert w.isatty() is False

    def test_encoding(self):
        q = queue.Queue()
        w = _QueueWriter(q)
        assert w.encoding == "utf-8"


class TestSessionIdValidation:
    def test_valid_ids(self):
        assert _SESSION_ID_RE.match("abcdef01")
        assert _SESSION_ID_RE.match("00000000")
        assert _SESSION_ID_RE.match("deadbeef")

    def test_too_short(self):
        assert not _SESSION_ID_RE.match("abc")

    def test_too_long(self):
        assert not _SESSION_ID_RE.match("abcdef012")

    def test_uppercase_rejected(self):
        assert not _SESSION_ID_RE.match("ABCDEF01")

    def test_special_chars_rejected(self):
        assert not _SESSION_ID_RE.match("abcd-ef0")

    def test_empty_rejected(self):
        assert not _SESSION_ID_RE.match("")


class TestProbeModelCapabilities:
    """Test the model capability probe with mocked Ollama responses."""

    def test_tools_detected(self):
        from unittest.mock import patch, MagicMock
        from researcher.config import probe_model_capabilities, _model_caps_cache

        _model_caps_cache.pop("ollama/test-tools", None)  # clear cache
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "template": "{{- if .Tools }}tools block{{ end }}",
            "details": {"family": "qwen2"},
        }
        mock_resp.raise_for_status = lambda: None
        with patch("requests.post", return_value=mock_resp):
            caps = probe_model_capabilities("ollama/test-tools")
        assert caps["supports_tools"] is True
        assert caps["family"] == "qwen2"
        _model_caps_cache.pop("ollama/test-tools", None)

    def test_no_tools_detected(self):
        from unittest.mock import patch, MagicMock
        from researcher.config import probe_model_capabilities, _model_caps_cache

        _model_caps_cache.pop("ollama/test-notools", None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "template": "{{ .Prompt }}",
            "details": {"family": "gemma"},
        }
        mock_resp.raise_for_status = lambda: None
        with patch("requests.post", return_value=mock_resp):
            caps = probe_model_capabilities("ollama/test-notools")
        assert caps["supports_tools"] is False
        assert caps["family"] == "gemma"
        _model_caps_cache.pop("ollama/test-notools", None)

    def test_ollama_unreachable_defaults_to_no_tools(self):
        from unittest.mock import patch
        from researcher.config import probe_model_capabilities, _model_caps_cache

        _model_caps_cache.pop("ollama/test-down", None)
        with patch("requests.post", side_effect=ConnectionError("refused")):
            caps = probe_model_capabilities("ollama/test-down")
        assert caps["supports_tools"] is False
        _model_caps_cache.pop("ollama/test-down", None)

    def test_cache_hit(self):
        from researcher.config import probe_model_capabilities, _model_caps_cache

        _model_caps_cache["ollama/cached"] = {"supports_tools": True, "is_thinking": False, "family": "test"}
        caps = probe_model_capabilities("ollama/cached")
        assert caps["supports_tools"] is True
        _model_caps_cache.pop("ollama/cached", None)


class TestFuzzyGenreMatching:
    """Tests for fuzzy genre matching in the knowledge loader."""

    def setup_method(self):
        # Clear caches so each test gets fresh YAML data
        import researcher.composer.knowledge.loader as loader
        loader._genres_cache = None

    def test_exact_match(self):
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("jazz")
        assert "GENRE (JAZZ):" in block

    def test_exact_match_classical(self):
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("classical")
        assert "GENRE (CLASSICAL):" in block

    def test_substring_style_in_key(self):
        """'jaz' is a substring of 'jazz' → should match jazz."""
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("jaz")
        assert "GENRE (JAZZ):" in block

    def test_substring_key_in_style(self):
        """'jazz' is a substring of 'light jazzy jazz feel' → should match jazz."""
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("light jazzy jazz feel")
        assert "GENRE (JAZZ):" in block

    def test_fuzzy_jazzy(self):
        """'jazzy' contains 'jazz' as substring → should match jazz."""
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("jazzy")
        assert "GENRE (JAZZ):" in block

    def test_fuzzy_bluesy(self):
        """'blues' is substring of 'bluesy' → should match blues."""
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("bluesy")
        assert "GENRE (BLUES):" in block

    def test_fuzzy_rocky(self):
        """'rock' is substring of 'rocky' → should match rock."""
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("rocky")
        assert "GENRE (ROCK):" in block

    def test_fuzzy_folkish(self):
        """'folk' is substring of 'folkish' → should match folk."""
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("folkish")
        assert "GENRE (FOLK):" in block

    def test_case_insensitive(self):
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("JAZZ")
        assert "GENRE (JAZZ):" in block

    def test_whitespace_trimmed(self):
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("  blues  ")
        assert "GENRE (BLUES):" in block

    def test_unknown_falls_back_to_classical(self):
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("xylophonic_grunge_99")
        assert "GENRE (CLASSICAL):" in block

    def test_genre_block_has_required_sections(self):
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("jazz")
        assert "Form:" in block
        assert "Structure:" in block
        assert "Treble rhythm:" in block
        assert "Bass rhythm:" in block
        assert "Voicing:" in block
        assert "Melody style:" in block

    def test_genre_block_has_example_bars(self):
        from researcher.composer.knowledge.loader import get_genre_block
        block = get_genre_block("jazz")
        assert "EXAMPLE BARS" in block


class TestFuzzyProgressionMatching:
    """Tests for fuzzy chord progression matching in tools.py."""

    def test_exact_jazz_progression(self):
        from researcher.composer.tools import _build_progression
        lines = _build_progression("C major", "jazz", 8, "")
        assert "jazz" in lines[0].lower()

    def test_fuzzy_jazzy_progression(self):
        from researcher.composer.tools import _build_progression
        lines = _build_progression("C major", "jazzy", 8, "")
        # Should fuzzy-match to jazz, not fall back to classical
        joined = " ".join(lines)
        # Jazz progressions use m7/maj7/7 chords
        assert "m7" in joined or "maj7" in joined or "7" in joined

    def test_fuzzy_bluesy_progression(self):
        from researcher.composer.tools import _build_progression
        lines = _build_progression("C major", "bluesy", 12, "")
        joined = " ".join(lines)
        # Blues 12-bar uses dominant 7ths
        assert "7" in joined

    def test_unknown_style_with_feel_fallback(self):
        from researcher.composer.tools import _build_progression
        lines = _build_progression("C major", "xyzzy", 8, "jazz")
        # Should use feel='jazz' as fallback
        joined = " ".join(lines)
        assert "m7" in joined or "maj7" in joined or "7" in joined

    def test_unknown_style_no_feel_falls_to_classical(self):
        from researcher.composer.tools import _build_progression
        lines = _build_progression("C major", "xyzzy", 8, "")
        assert "xyzzy" in lines[0].lower()
