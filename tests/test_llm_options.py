"""Thinking, and how many layers go on the card.

Both of these were wrong in three places at once, because the three crews each
built their own Ollama options dict.

**Thinking.** The switch they used — ``chat_template_kwargs:
{"enable_thinking": ...}`` — is silently ignored by Ollama. Measured against
qwen3.5:9b with ``num_predict: 24`` and the prompt "Say exactly: READY":

    thinking left on (default)            ->  ''        (empty)
    chat_template_kwargs enable_thinking  ->  ''        (ignored)
    think: False                          ->  'READY'

So thinking was on for every thinking-capable model regardless of intent, and
the model spent its whole output budget in a channel Ollama returns
separately. gemma4:26b produced 2031 characters of thinking, zero of response,
``done_reason: length``. From the outside that is an agent that writes a plan
and stops — which is exactly what was reported.

**Layers.** ``num_gpu: 99`` pinned every layer to the GPU. gemma4:26b then
asks for a 17 GB buffer on a 15.5 GiB card: ``cudaMalloc failed: out of
memory``, or a lopsided split that ``ollama ps`` shows as ``21%/79% CPU/GPU``.
"""

import pytest

from researcher.llm_options import (
    extra_body,
    is_thinking_model,
    ollama_options,
)


class TestRecognisingThinkingModels:
    @pytest.mark.parametrize("model", [
        "ollama/qwen3.5:9b",
        "ollama/qwen3.6:27b",
        "ollama/deepseek-r1:14b",
        "ollama/gemma4:26b",
        "ollama/gemma4:12b",
        "ollama/gpt-oss:20b",
        "GEMMA4:26B",                  # case must not matter
    ])
    def test_thinking_models_are_recognised(self, model):
        assert is_thinking_model(model)

    @pytest.mark.parametrize("model", [
        "ollama/mistral-small3.2:24b",
        "ollama/phi4:14b",
        "ollama/qwen2.5-coder:14b",
    ])
    def test_plain_models_are_not(self, model):
        assert not is_thinking_model(model)

    def test_an_empty_name_is_safe(self):
        assert not is_thinking_model("")
        assert not is_thinking_model(None)


class TestTheThinkingSwitch:
    def test_thinking_is_off_by_default(self):
        """Every crew here needs tokens it can read."""
        assert extra_body("ollama/gemma4:26b")["think"] is False

    def test_thinking_can_be_asked_for(self):
        """Conversational tutoring wants the thinking panel."""
        assert extra_body("ollama/qwen3.5:9b", thinking=True)["think"] is True

    def test_a_plain_model_gets_no_think_key_at_all(self):
        """Sending it to a model with no thinking channel is just noise."""
        assert "think" not in extra_body("ollama/mistral-small3.2:24b")

    def test_the_switch_that_does_not_work_is_not_used(self):
        """chat_template_kwargs is ignored by Ollama; it must not reappear."""
        for thinking in (True, False):
            body = extra_body("ollama/qwen3.5:9b", thinking=thinking)
            assert "chat_template_kwargs" not in body


class TestLayerPlacement:
    def test_num_gpu_is_not_forced(self):
        """Ollama fits the model; 99 layers does not fit a 26B on 15.5 GiB."""
        assert "num_gpu" not in ollama_options()
        assert "num_gpu" not in extra_body("ollama/gemma4:26b")["options"]

    def test_it_can_still_be_set_explicitly(self):
        assert ollama_options(num_gpu=99)["num_gpu"] == 99


class TestTheOptionsThemselves:
    def test_user_params_are_honoured(self):
        opts = ollama_options({"num_ctx": 8192, "num_predict": 512,
                               "top_k": 20, "top_p": 0.5, "min_p": 0.1,
                               "repeat_penalty": 1.3})
        assert opts["num_ctx"] == 8192
        assert opts["num_predict"] == 512
        assert opts["top_k"] == 20
        assert opts["top_p"] == 0.5
        assert opts["min_p"] == 0.1
        assert opts["repeat_penalty"] == 1.3

    def test_a_zero_seed_means_random_and_is_omitted(self):
        """Sending seed=0 pins every answer to the same one."""
        assert "seed" not in ollama_options({"seed": 0})

    def test_a_real_seed_is_passed(self):
        assert ollama_options({"seed": 42})["seed"] == 42

    def test_overrides_win_over_params(self):
        opts = ollama_options({"num_predict": 100}, num_predict=16384)
        assert opts["num_predict"] == 16384

    def test_string_params_are_coerced(self):
        """They arrive from JSON in the settings panel."""
        opts = ollama_options({"num_ctx": "4096", "top_p": "0.8"})
        assert opts["num_ctx"] == 4096
        assert opts["top_p"] == 0.8


class TestEveryCrewUsesTheWorkingSwitch:
    """The bug existed in triplicate; this is what stops it coming back."""

    @pytest.mark.parametrize("module", [
        "researcher/crew.py",
        "researcher/tutor/crew.py",
        "researcher/composer/crew.py",
    ])
    def test_no_crew_sends_the_ignored_switch(self, module):
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / module)
        body = src.read_text()
        # Mentioning it in a comment that explains the bug is fine.
        code = "\n".join(ln for ln in body.splitlines()
                         if not ln.strip().startswith("#"))
        assert "chat_template_kwargs" not in code, (
            f"{module} uses a switch Ollama ignores")

    @pytest.mark.parametrize("module", [
        "researcher/crew.py",
        "researcher/tutor/crew.py",
        "researcher/composer/crew.py",
    ])
    def test_no_crew_pins_every_layer_to_the_gpu(self, module):
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / module)
        code = "\n".join(ln for ln in src.read_text().splitlines()
                         if not ln.strip().startswith("#"))
        assert "num_gpu" not in code, f"{module} forces layer placement"

    def test_the_research_crew_turns_thinking_off(self):
        """It runs ReAct, so it needs an Action line in the visible output."""
        import os

        os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
        from researcher.crew import ResearchCrew

        llm = ResearchCrew._make_llm("ollama/gemma4:26b",
                                     ResearchCrew.DEFAULT_LLM_PARAMS)
        body = llm.additional_params.get("extra_body", {})
        assert body.get("think") is False

    def test_the_composer_turns_thinking_off(self):
        """Its output has to be ABC notation we can parse."""
        import os

        os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
        from researcher.composer.crew import ComposerCrew

        llm = ComposerCrew._make_llm("ollama/gemma4:26b",
                                     ComposerCrew.DEFAULT_LLM_PARAMS)
        assert llm.additional_params["extra_body"].get("think") is False

    def test_the_tutor_keeps_its_explicit_control(self):
        """Thinking on for conversation, off for quiz and story JSON."""
        import os

        os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
        from researcher.tutor.crew import TutorCrew

        on = TutorCrew._make_llm("ollama/qwen3.5:9b",
                                 TutorCrew.DEFAULT_LLM_PARAMS)
        off = TutorCrew._make_llm("ollama/qwen3.5:9b",
                                  TutorCrew.DEFAULT_LLM_PARAMS, thinking=False)
        assert on.additional_params["extra_body"].get("think") is True
        assert off.additional_params["extra_body"].get("think") is False
