"""Tests for the image critique-and-retry loop and the video backend.

Both features are GPU-bound in production, so everything here mocks the actual
generation and exercises the surrounding logic — which is where the bugs live.
"""

from unittest.mock import patch

import pytest

from researcher import image as image_mod
from researcher import video as video_mod


# ---------------------------------------------------------------------------
# Image tag parsing
# ---------------------------------------------------------------------------


class TestTagToPath:
    def test_markdown_tag(self, tmp_path, monkeypatch):
        monkeypatch.setattr(image_mod, "_GENERATED_DIR", tmp_path)
        (tmp_path / "abc123.png").write_bytes(b"x")
        p = image_mod._tag_to_path("![generated image](/static/generated/abc123.png)")
        assert p is not None and p.name == "abc123.png"

    def test_html_tag(self, tmp_path, monkeypatch):
        monkeypatch.setattr(image_mod, "_GENERATED_DIR", tmp_path)
        (tmp_path / "def456.png").write_bytes(b"x")
        p = image_mod._tag_to_path('<img src="/static/generated/def456.png" alt="a">')
        assert p is not None and p.name == "def456.png"

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(image_mod, "_GENERATED_DIR", tmp_path)
        assert image_mod._tag_to_path("![x](/static/generated/nope.png)") is None

    def test_error_string_returns_none(self):
        assert image_mod._tag_to_path("Error during FLUX generation") is None


# ---------------------------------------------------------------------------
# Critique-and-retry loop
# ---------------------------------------------------------------------------


def _tool_fn():
    """The undecorated function behind the CrewAI tool."""
    return getattr(image_mod.generate_ai_image, "func", image_mod.generate_ai_image)


class TestRefineLoop:
    GOOD_TAG = "![generated image](/static/generated/a.png)"
    SECOND_TAG = "![generated image](/static/generated/b.png)"

    def test_disabled_does_a_single_generation(self, monkeypatch):
        monkeypatch.setattr(image_mod, "IMAGE_REFINE_ROUNDS", 0)
        calls = []
        monkeypatch.setattr(image_mod, "_generate_image_once",
                            lambda p: calls.append(p) or self.GOOD_TAG)
        assert _tool_fn()("a cat") == self.GOOD_TAG
        assert len(calls) == 1

    def test_accepted_first_time_does_not_regenerate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(image_mod, "IMAGE_REFINE_ROUNDS", 2)
        monkeypatch.setattr(image_mod, "_GENERATED_DIR", tmp_path)
        (tmp_path / "a.png").write_bytes(b"x")
        calls = []
        monkeypatch.setattr(image_mod, "_generate_image_once",
                            lambda p: calls.append(p) or self.GOOD_TAG)
        monkeypatch.setattr(image_mod, "_critique_image",
                            lambda path, prompt: (True, "none", prompt))
        assert _tool_fn()("a cat") == self.GOOD_TAG
        assert len(calls) == 1, "an accepted image must not be regenerated"

    def test_retry_uses_the_improved_prompt(self, monkeypatch, tmp_path):
        monkeypatch.setattr(image_mod, "IMAGE_REFINE_ROUNDS", 1)
        monkeypatch.setattr(image_mod, "_GENERATED_DIR", tmp_path)
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        calls = []

        def _gen(p):
            calls.append(p)
            return self.GOOD_TAG if len(calls) == 1 else self.SECOND_TAG

        monkeypatch.setattr(image_mod, "_generate_image_once", _gen)
        monkeypatch.setattr(image_mod, "_critique_image",
                            lambda path, prompt: (False, "no tower", "a cat WITH a tower"))
        out = _tool_fn()("a cat")
        assert calls == ["a cat", "a cat WITH a tower"]
        assert out == self.SECOND_TAG

    def test_no_better_prompt_still_rerolls(self, monkeypatch, tmp_path):
        """A faulted image with no prompt improvement must still be re-rolled.

        This asserts the opposite of what it originally did, and the change came
        from a hardware run: asked for seven marbles, ZImage drew six, and the
        critic correctly said so — but the prompt was already precise, so it had
        no improvement to offer. Bailing there skipped the retry in exactly the
        case the loop exists for. These pipelines are stochastic, so a re-roll is
        a real fix; the round cap bounds the cost.
        """
        monkeypatch.setattr(image_mod, "IMAGE_REFINE_ROUNDS", 3)
        monkeypatch.setattr(image_mod, "_GENERATED_DIR", tmp_path)
        (tmp_path / "a.png").write_bytes(b"x")
        calls = []
        monkeypatch.setattr(image_mod, "_generate_image_once",
                            lambda p: calls.append(p) or self.GOOD_TAG)
        monkeypatch.setattr(image_mod, "_critique_image",
                            lambda path, prompt: (False, "wrong", "a cat"))
        assert _tool_fn()("a cat") == self.GOOD_TAG
        # First render plus one re-roll per configured round, and no more.
        assert len(calls) == 4
        assert all(c == "a cat" for c in calls)

    def test_generation_error_skips_critique(self, monkeypatch):
        monkeypatch.setattr(image_mod, "IMAGE_REFINE_ROUNDS", 2)
        monkeypatch.setattr(image_mod, "_generate_image_once",
                            lambda p: "Error during FLUX generation: boom")
        called = []
        monkeypatch.setattr(image_mod, "_critique_image",
                            lambda *a: called.append(1) or (True, "", ""))
        out = _tool_fn()("a cat")
        assert out.startswith("Error")
        assert not called, "must not critique a failed generation"

    def test_failed_retry_keeps_the_first_image(self, monkeypatch, tmp_path):
        monkeypatch.setattr(image_mod, "IMAGE_REFINE_ROUNDS", 2)
        monkeypatch.setattr(image_mod, "_GENERATED_DIR", tmp_path)
        (tmp_path / "a.png").write_bytes(b"x")
        calls = []

        def _gen(p):
            calls.append(p)
            return self.GOOD_TAG if len(calls) == 1 else "Error: OOM"

        monkeypatch.setattr(image_mod, "_generate_image_once", _gen)
        monkeypatch.setattr(image_mod, "_critique_image",
                            lambda path, prompt: (False, "bad", "better prompt"))
        assert _tool_fn()("a cat") == self.GOOD_TAG

    def test_broken_critic_never_blocks_generation(self, monkeypatch, tmp_path):
        """A missing or erroring vision model must degrade to plain generation."""
        monkeypatch.setattr(image_mod, "_GENERATED_DIR", tmp_path)
        img = tmp_path / "a.png"
        from PIL import Image
        Image.new("RGB", (8, 8), "white").save(img)

        with patch("requests.post", side_effect=OSError("ollama down")):
            good, problems, improved = image_mod._critique_image(img, "a cat")
        assert good is True
        assert improved == "a cat"


# ---------------------------------------------------------------------------
# Video backend
# ---------------------------------------------------------------------------


class TestVideoParams:
    @pytest.mark.parametrize("seconds", [1, 2, 4, 8, 12])
    def test_frame_count_is_8k_plus_1(self, seconds):
        """Video pipelines require frame counts of the form 8k+1."""
        n = video_mod._frame_count(seconds, 16)
        assert (n - 1) % 8 == 0
        assert n >= 9

    def test_unknown_backend_is_ignored(self, monkeypatch):
        monkeypatch.setattr(video_mod, "VIDEO_BACKEND", "ltx")
        assert video_mod.update_video_params({"video_backend": "bogus"})["video_backend"] == "ltx"

    def test_backend_switch_drops_cached_pipeline(self, monkeypatch):
        """Leaving a stale pipeline resident would waste VRAM on a 16GB card."""
        monkeypatch.setattr(video_mod, "VIDEO_BACKEND", "ltx")
        monkeypatch.setattr(video_mod, "_video_pipe", object())
        monkeypatch.setattr(video_mod, "_video_pipe_kind", "ltx")
        video_mod.update_video_params({"video_backend": "wan"})
        assert video_mod._video_pipe is None

    def test_values_are_clamped(self, monkeypatch):
        monkeypatch.setattr(video_mod, "VIDEO_BACKEND", "ltx")
        p = video_mod.update_video_params(
            {"seconds": 999, "fps": 1, "steps": 9999, "guidance": -5}
        )
        assert p["seconds"] <= 15
        assert p["fps"] >= 4
        assert p["steps"] <= 100
        assert p["guidance"] >= 0


class TestVideoGeneration:
    @staticmethod
    def _fn():
        return getattr(video_mod.generate_ai_video, "func", video_mod.generate_ai_video)

    def test_disabled_backend_returns_clear_error(self, monkeypatch):
        monkeypatch.setattr(video_mod, "VIDEO_BACKEND", "none")
        out = self._fn()("anything")
        assert "disabled" in out.lower()

    def test_uncached_model_fails_fast_with_instructions(self, monkeypatch):
        """A chat request must not silently start a multi-GB download."""
        monkeypatch.setattr(video_mod, "VIDEO_BACKEND", "ltx")
        monkeypatch.setattr(video_mod, "_hf_model_cache_present", lambda repo: False)
        out = self._fn()("a fox in snow")
        assert "not downloaded" in out
        assert "hf download" in out

    def test_tool_result_is_the_final_answer(self):
        """Ollama is unloaded during generation, so a post-tool LLM call would
        come back empty — same reason GenerateAIImage sets this."""
        assert video_mod.generate_ai_video.result_as_answer is True

    def test_video_tag_to_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(video_mod, "_GENERATED_DIR", tmp_path)
        (tmp_path / "clip.mp4").write_bytes(b"x")
        tag = '<video src="/static/generated/clip.mp4" controls></video>'
        assert video_mod.video_tag_to_path(tag).name == "clip.mp4"
        assert video_mod.video_tag_to_path("no video here") is None


# ---------------------------------------------------------------------------
# Media prefix routing ("image: …" / "video: …" bypasses the agent)
# ---------------------------------------------------------------------------


class TestMediaPrefix:
    """The prefix must be narrow.

    A false positive is expensive and confusing: an ordinary question would be
    fed to a diffusion model instead of the agent, and the user would get a
    picture back instead of an answer.
    """

    @staticmethod
    def _match(msg):
        from researcher.routes.chat import _match_media_prefix
        return _match_media_prefix(msg)

    @pytest.mark.parametrize("msg,expected", [
        ("video: a fox running through snow", ("video", "a fox running through snow")),
        ("clip: waves at sunset", ("video", "waves at sunset")),
        ("animation: a bouncing ball", ("video", "a bouncing ball")),
        ("image: a red apple", ("image", "a red apple")),
        ("img: a red apple", ("image", "a red apple")),
        ("picture: a barn", ("image", "a barn")),
        ("photo: portrait of a woman", ("image", "portrait of a woman")),
        ("draw: a house", ("image", "a house")),
    ])
    def test_recognised_prefixes(self, msg, expected):
        assert self._match(msg) == expected

    @pytest.mark.parametrize("msg", [
        "What is the capital of France?",
        "Can you make a video about physics later?",       # 'video' not at the start
        "my video: notes from the meeting",                 # prefix not leading
        "Tell me about image compression",
        "",
    ])
    def test_ordinary_messages_are_not_hijacked(self, msg):
        assert self._match(msg) is None

    @pytest.mark.parametrize("msg", ["video:", "image:   ", "photo:"])
    def test_prefix_without_a_prompt_is_ignored(self, msg):
        """A bare prefix has nothing to generate, so fall through to the agent."""
        assert self._match(msg) is None

    def test_case_and_spacing_are_tolerated(self):
        assert self._match("  VIDEO :  a storm  ") == ("video", "a storm")
        assert self._match("Image:A cat") == ("image", "A cat")

    def test_multiline_prompt_is_kept_whole(self):
        kind, prompt = self._match("image: a castle\non a hill at dawn")
        assert kind == "image"
        assert "on a hill at dawn" in prompt

    def test_colon_inside_the_prompt_survives(self):
        kind, prompt = self._match("image: a sign reading: OPEN")
        assert kind == "image"
        assert prompt == "a sign reading: OPEN"


# ---------------------------------------------------------------------------
# Per-backend video defaults
# ---------------------------------------------------------------------------


class TestBackendDefaults:
    """Each video model has its own trained resolution and frame rate.

    Carrying one model's numbers onto the other is what made LTX clips limp, so
    switching backend must re-derive anything the user has not pinned.
    """

    @pytest.fixture(autouse=True)
    def _restore(self):
        v = video_mod
        saved = (v.VIDEO_BACKEND, v._VIDEO_WIDTH, v._VIDEO_HEIGHT, v._VIDEO_FPS,
                 v._VIDEO_SECONDS, v._VIDEO_STEPS, v._VIDEO_GUIDANCE, set(v._PINNED))
        yield
        (v.VIDEO_BACKEND, v._VIDEO_WIDTH, v._VIDEO_HEIGHT, v._VIDEO_FPS,
         v._VIDEO_SECONDS, v._VIDEO_STEPS, v._VIDEO_GUIDANCE) = saved[:7]
        v._PINNED = saved[7]

    def test_each_backend_has_its_own_native_settings(self):
        wan = video_mod._BACKEND_DEFAULTS["wan"]
        ltx = video_mod._BACKEND_DEFAULTS["ltx"]
        assert wan != ltx
        # Wan takes no frame_rate argument and trains at 24; LTX conditions on 25.
        assert wan["fps"] == 24 and ltx["fps"] == 25
        assert wan["guidance"] == 5.0 and ltx["guidance"] == 3.0

    def test_switching_backend_rederives_unpinned_settings(self):
        v = video_mod
        v._PINNED = set()
        # Force a real transition: re-selecting the current backend is a no-op
        # by design, so start from the other one.
        v.update_video_params({"video_backend": "ltx"})
        v.update_video_params({"video_backend": "wan"})
        assert (v._VIDEO_WIDTH, v._VIDEO_FPS) == (832, 24)
        v.update_video_params({"video_backend": "ltx"})
        assert (v._VIDEO_WIDTH, v._VIDEO_FPS) == (704, 25)

    def test_reselecting_the_same_backend_changes_nothing(self):
        v = video_mod
        v._PINNED = set()
        v.update_video_params({"video_backend": "wan"})
        v._VIDEO_WIDTH = 640
        v.update_video_params({"video_backend": "wan"})
        assert v._VIDEO_WIDTH == 640, "a no-op switch clobbered a runtime value"

    def test_pinned_settings_survive_a_backend_switch(self):
        """An explicit env var must not be silently overwritten."""
        v = video_mod
        v._PINNED = {"width", "fps"}
        v.update_video_params({"video_backend": "wan"})
        v._VIDEO_WIDTH, v._VIDEO_FPS = 640, 12
        v.update_video_params({"video_backend": "ltx"})
        assert v._VIDEO_WIDTH == 640, "pinned width was overwritten"
        assert v._VIDEO_FPS == 12, "pinned fps was overwritten"
        # unpinned values should still follow the backend
        assert v._VIDEO_GUIDANCE == 3.0

    def test_frame_count_is_legal_for_both_vaes(self):
        """LTX needs 8k+1; Wan's VAE compresses 4x temporally, needing 4k+1."""
        for seconds, fps in ((2.5, 25), (3.0, 24), (1.0, 24), (5.0, 25)):
            n = video_mod._frame_count(seconds, fps)
            assert (n - 1) % 8 == 0, f"{n} frames is not 8k+1"
            assert (n - 1) % 4 == 0, f"{n} frames is not 4k+1"
            assert n >= 9

    def test_wan_is_the_default_backend(self):
        assert video_mod._BACKEND_DEFAULTS.get("wan") is not None
        assert "Wan2.2" in video_mod._WAN_MODEL
