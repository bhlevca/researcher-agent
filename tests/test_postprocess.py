"""Tests for the post-processing logic in main.py."""

import re
from unittest.mock import patch
from pathlib import Path

from researcher.main import _postprocess, _extract_usage, STATIC_DIR


class TestPostprocess:
    def test_plain_text_passes_through(self):
        result = _postprocess("Hello world", "")
        assert result == "Hello world"

    def test_empty_response_becomes_fallback(self):
        result = _postprocess("", "")
        assert "could not complete" in result.lower() or len(result) > 0

    def test_strips_result_tags(self):
        result = _postprocess("<result>  </result>Some answer", "")
        assert "<result>" not in result
        assert "Some answer" in result

    def test_strips_trailing_thought_chain(self):
        text = "Final answer here.\nThought: I should check\nAction: search\nAction Input: test"
        result = _postprocess(text, "")
        assert "Thought:" not in result
        assert "Final answer here." in result

    def test_image_from_log_when_missing_in_response(self, tmp_path):
        # Create a fake image file
        gen_dir = STATIC_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        fake_img = gen_dir / "abc123456789.png"
        fake_img.write_bytes(b"fake png")

        try:
            log = "Some output\n![generated image](/static/generated/abc123456789.png)\nMore output"
            result = _postprocess("The agent did something", log)
            assert "/static/generated/abc123456789.png" in result
        finally:
            fake_img.unlink(missing_ok=True)

    def test_invalid_image_reference_removed(self):
        text = "Here is an image: ![pic](/static/generated/nonexistent999.png)"
        result = _postprocess(text, "")
        assert "nonexistent999" not in result


class TestExtractUsage:
    def test_with_token_usage(self):
        mock = type("Result", (), {
            "token_usage": type("TU", (), {
                "total_tokens": 100,
                "prompt_tokens": 40,
                "completion_tokens": 60,
            })()
        })()
        usage = _extract_usage(mock)
        assert usage["total_tokens"] == 100
        assert usage["prompt_tokens"] == 40
        assert usage["completion_tokens"] == 60

    def test_without_token_usage(self):
        mock = type("Result", (), {"token_usage": None})()
        assert _extract_usage(mock) == {}

    def test_no_attribute(self):
        mock = type("Result", (), {})()
        assert _extract_usage(mock) == {}
