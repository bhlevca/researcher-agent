"""Tests for crew.py — font discovery, Pillow image generation, build_crew, and memory."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import researcher.crew as _crew_mod
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


# ---------------------------------------------------------------------------
# build_crew — task isolation and memory
# ---------------------------------------------------------------------------

class TestBuildCrew:
    """Test build_crew returns a Crew with proper task isolation."""

    def test_simple_topic_single_task(self):
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._decompose = lambda topic: ["simple question"]
        crew = rc.build_crew("simple question")
        assert len(crew.tasks) == 1

    def test_single_task_has_context_empty(self):
        """Even single tasks should have context=[] to prevent memory leaks."""
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._decompose = lambda topic: ["simple question"]
        crew = rc.build_crew("simple question")
        assert crew.tasks[0].context == []

    def test_multi_part_all_tasks_isolated(self):
        """Each task must have context=[] so no output forwarding occurs."""
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._decompose = lambda topic: [
            "Write essay in English (1500 words)",
            "Write essay in German (1500 words)",
            "Write essay in French (1500 words)",
        ]
        crew = rc.build_crew("Write in English, German and French")
        assert len(crew.tasks) == 3
        for task in crew.tasks:
            assert task.context == [], "Task context must be [] to prevent forwarding"

    def test_multi_part_no_original_request_in_description(self):
        """Sub-task descriptions must NOT contain 'ORIGINAL REQUEST'."""
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._decompose = lambda topic: [
            "Write essay in English (1500 words)",
            "Write essay in German (1500 words)",
        ]
        crew = rc.build_crew("Write in English and German")
        for task in crew.tasks:
            assert "ORIGINAL REQUEST" not in task.description

    def test_multi_part_descriptions_contain_part_text(self):
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._decompose = lambda topic: [
            "Write essay in English (1500 words)",
            "Write essay in German (1500 words)",
        ]
        crew = rc.build_crew("Write in English and German")
        assert "English" in crew.tasks[0].description
        assert "German" in crew.tasks[1].description


class TestCrewPostprocess:
    """Test postprocess handles single and list results."""

    def test_single_result_passthrough(self):
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._is_multi_part = False
        mock_result = type("R", (), {"raw": "hello world"})()
        assert rc.postprocess(mock_result) == "hello world"

    def test_multi_part_tasks_output_merged(self):
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._is_multi_part = True
        t1 = type("T", (), {"raw": "Part one"})()
        t2 = type("T", (), {"raw": "Part two"})()
        mock_result = type("R", (), {"tasks_output": [t1, t2]})()
        merged = rc.postprocess(mock_result)
        assert "Part one" in merged
        assert "Part two" in merged
        assert "---" in merged

    def test_multi_part_empty_entries_skipped(self):
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._is_multi_part = True
        t1 = type("T", (), {"raw": "Content"})()
        t2 = type("T", (), {"raw": ""})()
        mock_result = type("R", (), {"tasks_output": [t1, t2]})()
        result = rc.postprocess(mock_result)
        assert result == "Content"


class TestResetMemory:
    """Test reset_memory clears the crew's memory store."""

    def test_reset_calls_memory_reset(self):
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._memory = MagicMock()
        rc.reset_memory()
        rc._memory.reset.assert_called_once()

    def test_reset_without_memory_is_noop(self):
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        # _memory not set yet — should not raise
        rc.reset_memory()

    def test_reset_handles_exception(self):
        from researcher.crew import ResearchCrew
        rc = ResearchCrew()
        rc._memory = MagicMock()
        rc._memory.reset.side_effect = RuntimeError("storage error")
        # Should log warning but not raise
        rc.reset_memory()
