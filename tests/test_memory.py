"""Tests for memory and planning configuration — ensuring no OPENAI_API_KEY dependency.

These tests verify that:
1. The Agent does NOT have memory=True (which creates Memory() with default gpt-4o-mini)
2. The Crew has memory=False (intentional — avoids OpenAI dependency and context bloat)
3. PlanningConfig is properly bounded (max_attempts is not None/unlimited)
"""

import os
from unittest.mock import patch

import pytest


# Prevent CrewAI from reaching out during import
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("SERPER_API_KEY", "test-key-not-real")


@pytest.fixture
def crew_instance():
    """Build a ResearchCrew and return the Crew object.

    Patches the Ollama base_url to a dummy so the LLM
    object can be created without a running Ollama server.
    """
    from researcher.crew import ResearchCrew

    rc = ResearchCrew(model="ollama/qwen3.5:9b")
    return rc


class TestAgentMemoryNotBoolean:
    """Agent must NOT use memory=True (creates Memory with gpt-4o-mini default)."""

    def test_agent_memory_is_not_true(self, crew_instance):
        """If agent.memory is True (bool), CrewAI creates Memory() which
        defaults to gpt-4o-mini and requires OPENAI_API_KEY."""
        agent = crew_instance.researcher()
        # memory should be None (falls back to crew) or a Memory instance — never True
        assert agent.memory is not True, (
            "Agent memory=True triggers Memory() with default gpt-4o-mini LLM. "
            "Remove memory=True from Agent; let it fall back to Crew's Memory."
        )

    def test_agent_memory_is_not_bare_memory(self, crew_instance):
        """If someone sets memory to a Memory instance on the agent,
        it must have an explicit LLM (not the default gpt-4o-mini)."""
        from crewai.memory.unified_memory import Memory

        agent = crew_instance.researcher()
        if isinstance(agent.memory, Memory):
            # The LLM field should not be the default string "gpt-4o-mini"
            assert agent.memory.llm != "gpt-4o-mini", (
                "Agent's Memory uses default gpt-4o-mini. "
                "Pass llm=ollama_llm to avoid OPENAI_API_KEY dependency."
            )


class TestCrewMemoryDisabled:
    """Crew memory is intentionally False to avoid OpenAI dependency and context bloat.

    Memory recall/save is handled via monkey-patches in crew.py instead.
    """

    def test_crew_memory_is_false(self, crew_instance):
        crew = crew_instance.crew()
        assert crew.memory is False, (
            "Crew memory should be False (intentional design). "
            "Memory operations are handled via monkey-patched recall/save."
        )

    def test_crew_does_not_create_memory_instance(self, crew_instance):
        crew = crew_instance.crew()
        mem = getattr(crew, "_memory", None)
        assert mem is None or mem is False, (
            f"Crew should not have a Memory instance, got {type(mem)}"
        )


class TestPlanningConfigBounded:
    """PlanningConfig must have bounded max_attempts to prevent infinite loops."""

    def test_agent_has_planning_config(self, crew_instance):
        agent = crew_instance.researcher()
        assert (
            agent.planning_config is not None
        ), "Agent should have a PlanningConfig for structured reasoning"

    def test_planning_max_attempts_is_bounded(self, crew_instance):
        """max_attempts=None means unlimited — must be set to prevent loops."""
        agent = crew_instance.researcher()
        pc = agent.planning_config
        assert pc is not None
        assert pc.max_attempts is not None, (
            "PlanningConfig.max_attempts is None (unlimited). "
            "Set a limit to prevent infinite reasoning loops with local models."
        )
        assert (
            1 <= pc.max_attempts <= 10
        ), f"max_attempts={pc.max_attempts} is outside reasonable range [1, 10]"

    def test_planning_max_steps_is_bounded(self, crew_instance):
        agent = crew_instance.researcher()
        pc = agent.planning_config
        assert pc is not None
        assert pc.max_steps is not None
        assert (
            1 <= pc.max_steps <= 20
        ), f"max_steps={pc.max_steps} is outside reasonable range [1, 20]"

    def test_planning_prompt_contains_ready_signal(self, crew_instance):
        """The plan prompt must instruct the model to produce the READY signal."""
        agent = crew_instance.researcher()
        pc = agent.planning_config
        assert pc is not None
        if pc.plan_prompt:
            assert (
                "READY: I am ready to execute the task." in pc.plan_prompt
            ), "Plan prompt must include the exact READY signal that CrewAI checks for"


class TestNoOpenAIKeyRequired:
    """Smoke test: building the crew must not raise OPENAI_API_KEY errors."""

    def test_crew_builds_without_openai_key(self):
        """Ensure the full crew can be built with no OPENAI_API_KEY set."""
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            # Re-set required test env vars
            os.environ["CREWAI_TRACING_ENABLED"] = "false"
            os.environ["SERPER_API_KEY"] = "test-key-not-real"
            from researcher.crew import ResearchCrew

            # This should not raise
            rc = ResearchCrew(model="ollama/qwen3.5:9b")
            crew = rc.crew()
            assert crew is not None
