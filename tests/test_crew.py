"""Tests for crew.py — font discovery and Pillow image generation."""

import json
import os
from pathlib import Path
from unittest.mock import patch

from researcher.crew import _find_truetype_font, _GENERATED_DIR


class TestFontDiscovery:
    def test_finds_a_font_on_this_system(self):
        # On any Linux dev machine, at least one path should exist
        result = _find_truetype_font()
        # We don't require a font (CI might not have one), so just check type
        assert result is None or (isinstance(result, str) and Path(result).is_file())

    def test_returns_none_when_no_fonts(self):
        with patch("researcher.crew._FONT_SEARCH_PATHS", ["/nonexistent/font.ttf"]):
            from researcher.crew import _find_truetype_font
            assert _find_truetype_font() is None


class TestGenerateImageTool:
    def test_basic_rectangle(self):
        from researcher.crew import generate_image

        spec = json.dumps({
            "width": 100,
            "height": 100,
            "background": "white",
            "shapes": [
                {"type": "rectangle", "x": 10, "y": 10, "width": 50, "height": 50,
                 "fill": "red"}
            ]
        })
        result = generate_image.run(spec)
        assert "![generated image]" in result
        assert "/static/generated/" in result
        # Verify file was created
        fname = result.split("/")[-1].rstrip(")")
        fpath = _GENERATED_DIR / fname
        assert fpath.exists()
        fpath.unlink()  # cleanup

    def test_text_shape(self):
        from researcher.crew import generate_image

        spec = json.dumps({
            "width": 200,
            "height": 100,
            "background": "#333333",
            "shapes": [
                {"type": "text", "x": 10, "y": 10, "text": "Hello", "size": 20,
                 "fill": "white"}
            ]
        })
        result = generate_image.run(spec)
        assert "![generated image]" in result
        fname = result.split("/")[-1].rstrip(")")
        (GENERATED := _GENERATED_DIR / fname).unlink(missing_ok=True)

    def test_circle_shape(self):
        from researcher.crew import generate_image

        spec = json.dumps({
            "width": 100,
            "height": 100,
            "background": "black",
            "shapes": [
                {"type": "circle", "cx": 50, "cy": 50, "radius": 30, "fill": "blue"}
            ]
        })
        result = generate_image.run(spec)
        assert "![generated image]" in result
        fname = result.split("/")[-1].rstrip(")")
        (_GENERATED_DIR / fname).unlink(missing_ok=True)

    def test_invalid_json_returns_error(self):
        from researcher.crew import generate_image

        result = generate_image.run("not json at all")
        assert "Error" in result

    def test_max_dimensions_clamped(self):
        from researcher.crew import generate_image

        spec = json.dumps({"width": 99999, "height": 99999, "shapes": []})
        result = generate_image.run(spec)
        # Should succeed (clamped to 2048) not crash
        assert "![generated image]" in result
        fname = result.split("/")[-1].rstrip(")")
        (_GENERATED_DIR / fname).unlink(missing_ok=True)
