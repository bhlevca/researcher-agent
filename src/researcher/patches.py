"""
CrewAI / litellm / Ollama compatibility patches.

All patches in this file are applied once at import time.  They work around
bugs or incompatibilities between CrewAI 1.14.5, litellm, and local Ollama
models (primarily qwen3.5:9b).

VERSION GUARD
-------------
If the crewai version changes, the assertion below will fail loudly at startup.
Before upgrading crewai, review each patch section, verify whether the upstream
bug is still present, and update TESTED_CREWAI_VERSION once confirmed.

HOW TO REMOVE A PATCH
----------------------
1. Confirm the upstream bug is fixed (run the corresponding test_patches.py test).
2. Delete the patch function and its call in apply_all().
3. Update the version guard to the new tested version.
"""

import re as _re
import logging
import functools

import crewai

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version guard — update this string after verifying all patches on a new version.
# ---------------------------------------------------------------------------
TESTED_CREWAI_VERSION = "1.14.5"

assert crewai.__version__ == TESTED_CREWAI_VERSION, (
    f"CrewAI version mismatch: patches were written for {TESTED_CREWAI_VERSION} "
    f"but {crewai.__version__} is installed.  Review each patch in patches.py, "
    f"verify it is still needed, then update TESTED_CREWAI_VERSION."
)


# ---------------------------------------------------------------------------
# P1 — Force ReAct text-based tool calling for Ollama models.
#
# Problem: CrewAI calls llm.supports_function_calling() → True for Ollama via
# litellm.  CrewAI then sends OpenAI-style tool schemas and expects structured
# tool_call JSON back.  qwen3.5 (and most Ollama models) return PLAIN TEXT
# describing the tool call, so CrewAI treats it as a "final answer" and never
# executes the tool.
#
# Fix: Override supports_function_calling() → False for Ollama, forcing CrewAI
# into ReAct mode (Action: / Action Input: text pattern), which works reliably.
#
# Note: CrewAI's LLM.__new__ routes "ollama/" models to
# OpenAICompatibleCompletion, not LLM itself.  Patching LLM has no effect.
# Instances with _allow_fc=True (e.g. the planning LLM) are exempted.
# ---------------------------------------------------------------------------
from crewai.llms.providers.openai_compatible.completion import (  # noqa: E402
    OpenAICompatibleCompletion as _OAICompat,
)

_orig_supports_fc = _OAICompat.supports_function_calling


def _apply_p1_react_tool_calling():
    def _patched_supports_fc(self):
        if getattr(self, "_allow_fc", False):
            return _orig_supports_fc(self)
        if (getattr(self, "provider", "") or "") == "ollama":
            return False
        return _orig_supports_fc(self)

    _OAICompat.supports_function_calling = _patched_supports_fc
    _logger.info("P1 applied: Ollama → ReAct tool calling (supports_function_calling=False)")


# ---------------------------------------------------------------------------
# P2 — Lenient READY detection for qwen3.5 planning responses.
#
# Problem: CrewAI's planning requires the exact string
#   "READY: I am ready to execute the task."
# but qwen3.5 outputs "READY", "**READY**", "* READY:", etc.
# Both the text-parsing path and the function-calling fallback path are affected.
#
# Fix: Patch _parse_planning_response and _call_with_function to use a regex
# that accepts all common READY formats.
#
# Also prevents plan stacking: each planning round appends to task.description,
# so on re-execution old plans accumulate.  Strip them before adding new ones,
# and cap plan length so verbose LLM tables don't blow up the context.
# ---------------------------------------------------------------------------
from crewai.utilities.reasoning_handler import AgentReasoning  # noqa: E402
import crewai.agent.utils as _agent_utils_mod  # noqa: E402

_READY_RE = _re.compile(r"(?m)^\s*(?:\*\s*)*\*{0,2}READY\*{0,2}\s*:?[^\n]*$")
_MAX_PLAN_CHARS = 2000

_orig_call_with_function = AgentReasoning._call_with_function
_orig_handle_reasoning = _agent_utils_mod.handle_reasoning


def _lenient_ready(text: str) -> bool:
    if "READY: I am ready to execute the task." in text:
        return True
    return bool(_READY_RE.search(text))


def _apply_p2_ready_detection():
    @staticmethod
    def _patched_parse(response: str) -> tuple[str, bool]:
        if not response:
            return "No plan was generated.", False
        return response, _lenient_ready(response)

    def _patched_call_with_function(self, prompt, plan_type):
        plan, steps, ready = _orig_call_with_function(self, prompt, plan_type)
        if not ready:
            ready = _lenient_ready(plan)
        return plan, steps, ready

    def _capped_handle_reasoning(agent, task):
        if "\n\nPlanning:\n" in task.description:
            task.description = task.description.split("\n\nPlanning:\n")[0]
        _orig_handle_reasoning(agent, task)
        if "\n\nPlanning:\n" in task.description:
            base, plan_text = task.description.split("\n\nPlanning:\n", 1)
            if len(plan_text) > _MAX_PLAN_CHARS:
                plan_text = plan_text[:_MAX_PLAN_CHARS] + "\n[plan truncated]"
            task.description = base + "\n\nPlanning:\n" + plan_text

    AgentReasoning._parse_planning_response = _patched_parse
    AgentReasoning._call_with_function = _patched_call_with_function
    _agent_utils_mod.handle_reasoning = _capped_handle_reasoning
    _logger.info(
        "P2 applied: lenient READY detection, no plan stacking, max %d chars", _MAX_PLAN_CHARS
    )


# ---------------------------------------------------------------------------
# P3 — Context-length errors handled before litellm re-raise.
#
# Problem 1: CrewAI's invoke-loop methods re-raise litellm errors BEFORE
# calling is_context_length_exceeded(), so Ollama context-length errors bypass
# the summarise-and-retry logic of respect_context_window=True.
#
# Problem 2: Ollama's error string "the input length exceeds the context length"
# does not match any pattern in CONTEXT_LIMIT_ERRORS (wrong word order vs.
# "context length exceeded").
#
# Fix: Add missing patterns to CONTEXT_LIMIT_ERRORS, then wrap all three
# invoke-loop methods so context-length errors are caught first.
# ---------------------------------------------------------------------------
from crewai.utilities.exceptions.context_window_exceeding_exception import (  # noqa: E402
    CONTEXT_LIMIT_ERRORS,
)
from crewai.utilities.agent_utils import is_context_length_exceeded  # noqa: E402
import crewai.agents.crew_agent_executor as _executor_mod  # noqa: E402

_EXTRA_CTX_PATTERNS = [
    "exceeds the context length",  # Ollama via litellm
    "input is too long",
    "prompt is too long",
]

_orig_invoke_loop_native_tools = _executor_mod.CrewAgentExecutor._invoke_loop_native_tools
_orig_invoke_loop_react = _executor_mod.CrewAgentExecutor._invoke_loop_react
_orig_invoke_loop_native_no_tools = _executor_mod.CrewAgentExecutor._invoke_loop_native_no_tools


def _apply_p3_context_length():
    for pattern in _EXTRA_CTX_PATTERNS:
        if pattern not in CONTEXT_LIMIT_ERRORS:
            CONTEXT_LIMIT_ERRORS.append(pattern)

    def _patch_exception_handler(original_method):
        @functools.wraps(original_method)
        def wrapper(self, *args, **kwargs):
            try:
                return original_method(self, *args, **kwargs)
            except Exception as e:
                if is_context_length_exceeded(e):
                    from crewai.utilities.agent_utils import handle_context_length
                    _logger.warning("Context length exceeded — attempting to summarise…")
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=self._printer,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        i18n=self._i18n,
                        verbose=self.agent.verbose,
                    )
                    return original_method(self, *args, **kwargs)
                raise
        return wrapper

    _executor_mod.CrewAgentExecutor._invoke_loop_native_tools = _patch_exception_handler(
        _orig_invoke_loop_native_tools
    )
    _executor_mod.CrewAgentExecutor._invoke_loop_react = _patch_exception_handler(
        _orig_invoke_loop_react
    )
    _executor_mod.CrewAgentExecutor._invoke_loop_native_no_tools = _patch_exception_handler(
        _orig_invoke_loop_native_no_tools
    )
    _logger.info("P3 applied: context-length errors handled before litellm re-raise")


# ---------------------------------------------------------------------------
# P4 — Memory recall: depth-switchable, capped results, truncated entries.
#
# Problem: Default depth="deep" triggers RecallFlow (multiple LLM calls, ~50s).
# Default limit=5 with large stored entries bloats the context window.
# Long query strings overflow the embedding model's context (nomic-embed-text).
#
# Fix: Expose a runtime-switchable memory_depth flag ("shallow"/"deep").
# In shallow mode cap at 5 entries, truncate each to 1000 chars, truncate
# query to 4000 chars.
# ---------------------------------------------------------------------------
from crewai.memory.unified_memory import Memory as _UnifiedMemory  # noqa: E402

_orig_memory_recall = _UnifiedMemory.recall

_MAX_MEMORY_ENTRY_CHARS = 1000
_MEMORY_RECALL_LIMIT = 5

# Imported by routes/models.py to expose /memory-depth toggle
memory_depth: str = "shallow"


def _apply_p4_memory_recall():
    def _capped_recall(self, query, limit=10, **kwargs):
        import researcher.patches as _patches_mod
        depth = _patches_mod.memory_depth
        kwargs["depth"] = depth
        cap = _MEMORY_RECALL_LIMIT if depth == "shallow" else min(limit, 5)
        char_limit = _MAX_MEMORY_ENTRY_CHARS if depth == "shallow" else 2000
        if isinstance(query, str) and len(query) > 4000:
            query = query[:4000]
        matches = _orig_memory_recall(self, query, limit=cap, **kwargs)
        for m in matches:
            if len(m.record.content) > char_limit:
                m.record.content = m.record.content[:char_limit] + "…"
        return matches

    _UnifiedMemory.recall = _capped_recall
    _logger.info(
        "P4 applied: memory recall depth=%s, max %d entries, %d chars",
        memory_depth, _MEMORY_RECALL_LIMIT, _MAX_MEMORY_ENTRY_CHARS,
    )


# ---------------------------------------------------------------------------
# P5 — Truncate memory save content to prevent LLM context overflow.
#
# Problem: _save_to_memory feeds full task.description + expected_output +
# full result to extract_memories() → the combined text easily exceeds the
# LLM context window → memory_save_failed errors on every turn.
#
# Fix: Build a brief summary (task desc ≤500 chars + result ≤2000 chars)
# and cap the total at 6000 chars before calling extract_memories().
# ---------------------------------------------------------------------------
from crewai.agents.agent_builder.base_agent_executor import (  # noqa: E402
    BaseAgentExecutor as _BaseAgentExecutor,
)

_orig_save_to_memory = _BaseAgentExecutor._save_to_memory
_MAX_SAVE_CONTENT_CHARS = 6000


def _apply_p5_memory_save():
    def _patched_save_to_memory(self, output) -> None:
        from crewai.utilities.string_utils import sanitize_tool_name

        memory = getattr(self.agent, "memory", None) or (
            getattr(self.crew, "_memory", None) if self.crew else None
        )
        if memory is None or not self.task or getattr(memory, "read_only", False):
            return
        if f"Action: {sanitize_tool_name('Delegate work to coworker')}" in output.text:
            return
        try:
            desc = (self.task.description or "")[:500]
            result_text = (output.text or "")[:2000]
            raw = f"Task: {desc}\nAgent: {self.agent.role}\nResult: {result_text}"
            if len(raw) > _MAX_SAVE_CONTENT_CHARS:
                raw = raw[:_MAX_SAVE_CONTENT_CHARS]
            extracted = memory.extract_memories(raw)
            if extracted:
                memory.remember_many(extracted, agent_role=self.agent.role)
        except Exception as e:
            self.agent._logger.log("error", f"Failed to save to memory: {e}")

    _BaseAgentExecutor._save_to_memory = _patched_save_to_memory
    _logger.info("P5 applied: _save_to_memory content capped at %d chars", _MAX_SAVE_CONTENT_CHARS)


# ---------------------------------------------------------------------------
# Entry point — called once at module import time.
# ---------------------------------------------------------------------------

def apply_all():
    _apply_p1_react_tool_calling()
    _apply_p2_ready_detection()
    _apply_p3_context_length()
    _apply_p4_memory_recall()
    _apply_p5_memory_save()


apply_all()
