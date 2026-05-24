"""
Tests for researcher.patches — CrewAI compatibility patches.

Each test verifies that:
  (a) the patch is correctly applied (the patched behaviour works), AND
  (b) where possible, that the upstream behaviour that necessitates the
      patch still exists (so the test fails when the patch becomes redundant,
      prompting a clean-up).

Run with: .venv/bin/python -m pytest tests/test_patches.py -v
"""

import pytest


# Import patches module to ensure patches are applied
import researcher.patches as patches


# ---------------------------------------------------------------------------
# P1 — ReAct tool calling: Ollama provider returns False
# ---------------------------------------------------------------------------

class TestP1ReactToolCalling:

    def test_ollama_provider_returns_false(self):
        """Ollama instances must report supports_function_calling=False."""
        from crewai.llms.providers.openai_compatible.completion import (
            OpenAICompatibleCompletion,
        )
        from unittest.mock import MagicMock

        instance = MagicMock(spec=OpenAICompatibleCompletion)
        instance.provider = "ollama"
        instance._allow_fc = False

        result = OpenAICompatibleCompletion.supports_function_calling(instance)
        assert result is False, (
            "P1 patch broken: Ollama provider should return False for "
            "supports_function_calling to force ReAct mode."
        )

    def test_non_ollama_provider_not_affected(self):
        """Non-Ollama providers (e.g. openai) must not be affected by the patch."""
        from crewai.llms.providers.openai_compatible.completion import (
            OpenAICompatibleCompletion,
        )
        from unittest.mock import MagicMock, patch as mock_patch

        instance = MagicMock(spec=OpenAICompatibleCompletion)
        instance.provider = "openai"
        instance._allow_fc = False

        # The original method would decide based on provider capabilities;
        # our patch must delegate to it for non-ollama providers.
        with mock_patch.object(
            patches, "_orig_supports_fc", return_value=True
        ):
            result = OpenAICompatibleCompletion.supports_function_calling(instance)
        assert result is True

    def test_allow_fc_flag_exempts_instance(self):
        """Instances with _allow_fc=True (planning LLM) must bypass the patch."""
        from crewai.llms.providers.openai_compatible.completion import (
            OpenAICompatibleCompletion,
        )
        from unittest.mock import MagicMock, patch as mock_patch

        instance = MagicMock(spec=OpenAICompatibleCompletion)
        instance.provider = "ollama"
        instance._allow_fc = True

        with mock_patch.object(
            patches, "_orig_supports_fc", return_value=True
        ):
            result = OpenAICompatibleCompletion.supports_function_calling(instance)
        assert result is True, "_allow_fc=True should bypass the Ollama restriction"


# ---------------------------------------------------------------------------
# P2 — Lenient READY detection
# ---------------------------------------------------------------------------

class TestP2ReadyDetection:

    @pytest.mark.parametrize("text, expected", [
        ("READY: I am ready to execute the task.", True),   # exact upstream string
        ("READY", True),                                    # qwen3.5 short form
        ("**READY**", True),                               # bold markdown
        ("* READY:", True),                                # bullet prefix
        ("READY: Let's proceed", True),                    # trailing text
        ("Not ready yet", False),
        ("", False),
        ("I need more information", False),
    ])
    def test_lenient_ready(self, text, expected):
        assert patches._lenient_ready(text) is expected

    def test_parse_planning_response_ready(self):
        """_parse_planning_response must accept qwen3.5-style READY."""
        from crewai.utilities.reasoning_handler import AgentReasoning

        plan, ready = AgentReasoning._parse_planning_response("**READY**")
        assert ready is True, (
            "P2 patch broken: _parse_planning_response should accept **READY**"
        )

    def test_parse_planning_response_not_ready(self):
        from crewai.utilities.reasoning_handler import AgentReasoning

        plan, ready = AgentReasoning._parse_planning_response("Still thinking...")
        assert ready is False

    def test_parse_planning_response_empty(self):
        from crewai.utilities.reasoning_handler import AgentReasoning

        plan, ready = AgentReasoning._parse_planning_response("")
        assert ready is False
        assert "No plan" in plan

    def test_plan_stacking_prevented(self):
        """handle_reasoning must strip the previous plan before adding a new one."""
        import crewai.agent.utils as utils_mod
        from unittest.mock import MagicMock, patch as mock_patch

        task = MagicMock()
        task.description = "Do something\n\nPlanning:\nOld plan text"
        agent = MagicMock()

        # Simulate handle_reasoning appending a new plan
        def fake_orig(a, t):
            t.description = t.description + "\n\nPlanning:\nNew plan text"

        with mock_patch.object(patches, "_orig_handle_reasoning", fake_orig):
            utils_mod.handle_reasoning(agent, task)

        assert task.description.count("\n\nPlanning:\n") == 1, (
            "P2 patch broken: old plan should be stripped before adding new one"
        )
        assert "New plan text" in task.description
        assert "Old plan text" not in task.description

    def test_plan_capped_at_max_chars(self):
        """Plans longer than _MAX_PLAN_CHARS must be truncated."""
        import crewai.agent.utils as utils_mod
        from unittest.mock import MagicMock, patch as mock_patch

        long_plan = "x" * (patches._MAX_PLAN_CHARS + 500)
        task = MagicMock()
        task.description = "Task desc"
        agent = MagicMock()

        def fake_orig(a, t):
            t.description = t.description + "\n\nPlanning:\n" + long_plan

        with mock_patch.object(patches, "_orig_handle_reasoning", fake_orig):
            utils_mod.handle_reasoning(agent, task)

        _, plan_part = task.description.split("\n\nPlanning:\n", 1)
        assert len(plan_part) <= patches._MAX_PLAN_CHARS + len("\n[plan truncated]")
        assert "[plan truncated]" in plan_part


# ---------------------------------------------------------------------------
# P3 — Context-length error patterns and invoke-loop wrapping
# ---------------------------------------------------------------------------

class TestP3ContextLength:

    def test_extra_patterns_in_context_limit_errors(self):
        """All Ollama-specific error strings must be present in CONTEXT_LIMIT_ERRORS."""
        from crewai.utilities.exceptions.context_window_exceeding_exception import (
            CONTEXT_LIMIT_ERRORS,
        )
        for pattern in patches._EXTRA_CTX_PATTERNS:
            assert pattern in CONTEXT_LIMIT_ERRORS, (
                f"P3 patch broken: '{pattern}' missing from CONTEXT_LIMIT_ERRORS"
            )

    def test_invoke_loop_methods_are_wrapped(self):
        """All three invoke-loop methods must be wrapped by our patch."""
        import crewai.agents.crew_agent_executor as exe_mod

        for method_name in (
            "_invoke_loop_native_tools",
            "_invoke_loop_react",
            "_invoke_loop_native_no_tools",
        ):
            method = getattr(exe_mod.CrewAgentExecutor, method_name)
            assert hasattr(method, "__wrapped__") or "wrapper" in str(method), (
                f"P3 patch broken: {method_name} does not appear to be wrapped"
            )


# ---------------------------------------------------------------------------
# P4 — Memory recall capping
# ---------------------------------------------------------------------------

class TestP4MemoryRecall:

    def test_memory_recall_is_patched(self):
        """UnifiedMemory.recall must be our patched version."""
        from crewai.memory.unified_memory import Memory

        # The patched method's closure references _orig_memory_recall
        assert Memory.recall is not patches._orig_memory_recall, (
            "P4 patch broken: Memory.recall was not replaced"
        )

    def test_query_truncated_when_too_long(self):
        """Queries longer than 4000 chars must be truncated before recall."""
        from unittest.mock import MagicMock, patch as mock_patch
        from crewai.memory.unified_memory import Memory

        captured = {}

        def fake_orig(self, query, limit=10, **kwargs):
            captured["query"] = query
            captured["limit"] = limit
            return []

        instance = MagicMock(spec=Memory)
        long_query = "q" * 5000

        with mock_patch.object(patches, "_orig_memory_recall", fake_orig):
            Memory.recall(instance, long_query, limit=10)

        assert len(captured["query"]) == 4000, (
            "P4 patch broken: long query was not truncated to 4000 chars"
        )

    def test_results_capped_in_shallow_mode(self):
        """In shallow mode, at most _MEMORY_RECALL_LIMIT entries are requested."""
        from unittest.mock import MagicMock, patch as mock_patch
        from crewai.memory.unified_memory import Memory

        captured = {}

        def fake_orig(self, query, limit=10, **kwargs):
            captured["limit"] = limit
            return []

        instance = MagicMock(spec=Memory)
        patches.memory_depth = "shallow"

        with mock_patch.object(patches, "_orig_memory_recall", fake_orig):
            Memory.recall(instance, "query")

        assert captured["limit"] == patches._MEMORY_RECALL_LIMIT

    def test_entry_content_truncated(self):
        """Entries longer than _MAX_MEMORY_ENTRY_CHARS must be truncated."""
        from unittest.mock import MagicMock, patch as mock_patch
        from crewai.memory.unified_memory import Memory

        long_content = "c" * (patches._MAX_MEMORY_ENTRY_CHARS + 200)
        mock_entry = MagicMock()
        mock_entry.record.content = long_content

        def fake_orig(self, query, limit=10, **kwargs):
            return [mock_entry]

        instance = MagicMock(spec=Memory)
        patches.memory_depth = "shallow"

        with mock_patch.object(patches, "_orig_memory_recall", fake_orig):
            results = Memory.recall(instance, "query")

        assert len(results[0].record.content) <= patches._MAX_MEMORY_ENTRY_CHARS + 1  # +1 for "…"
        assert results[0].record.content.endswith("…")


# ---------------------------------------------------------------------------
# P5 — Memory save truncation
# ---------------------------------------------------------------------------

class TestP5MemorySave:

    def test_save_to_memory_is_patched(self):
        """BaseAgentExecutor._save_to_memory must be our patched version."""
        from crewai.agents.agent_builder.base_agent_executor import BaseAgentExecutor

        assert BaseAgentExecutor._save_to_memory is not patches._orig_save_to_memory, (
            "P5 patch broken: _save_to_memory was not replaced"
        )

    def test_no_memory_skips_gracefully(self):
        """When agent has no memory, _save_to_memory must return without error."""
        from crewai.agents.agent_builder.base_agent_executor import BaseAgentExecutor
        from unittest.mock import MagicMock

        executor = MagicMock()  # no spec — BaseAgentExecutor uses __slots__-like attrs
        executor.agent.memory = None
        executor.crew = None
        executor.task = MagicMock()

        output = MagicMock()
        output.text = "some result"

        # Should not raise
        BaseAgentExecutor._save_to_memory(executor, output)

    def test_content_truncated_before_extraction(self):
        """Raw content fed to extract_memories must not exceed _MAX_SAVE_CONTENT_CHARS."""
        from crewai.agents.agent_builder.base_agent_executor import BaseAgentExecutor
        from unittest.mock import MagicMock, patch as mock_patch

        executor = MagicMock()  # no spec — see above
        executor.agent.memory = MagicMock()
        executor.agent.memory.read_only = False
        executor.agent.role = "researcher"
        executor.crew = None
        executor.task = MagicMock()
        executor.task.description = "Task desc"
        executor.agent.memory.extract_memories.return_value = []

        long_result = "r" * 10000
        output = MagicMock()
        output.text = long_result

        with mock_patch("crewai.utilities.string_utils.sanitize_tool_name", return_value="x"):
            BaseAgentExecutor._save_to_memory(executor, output)

        call_args = executor.agent.memory.extract_memories.call_args
        raw_content = call_args[0][0]
        assert len(raw_content) <= patches._MAX_SAVE_CONTENT_CHARS, (
            f"P5 patch broken: content fed to extract_memories ({len(raw_content)} chars) "
            f"exceeds limit ({patches._MAX_SAVE_CONTENT_CHARS})"
        )
