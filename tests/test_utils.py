"""Tests for pure utility functions in main.py."""

import re
import queue
from researcher.main import _clean_line, _QueueWriter, _SESSION_ID_RE


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
