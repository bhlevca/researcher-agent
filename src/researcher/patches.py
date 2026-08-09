"""
CrewAI / litellm / Ollama compatibility patches.

All patches in this file are applied once at import time.  They work around
bugs or incompatibilities between CrewAI 1.14.5, litellm, and local Ollama
models (primarily qwen3.5:9b).

VERSION GUARD
-------------
Guarding on an exact version string is too brittle: a routine `uv lock --upgrade`
turns a working install into a hard startup crash even when every patch still
applies cleanly.  What actually matters is not the version number but whether the
symbols each patch reaches into still exist.

So the guard has two levels:

* **Structural check (fatal).**  Every attribute a patch monkey-patches is
  verified to exist before anything is applied.  A missing symbol means the patch
  would silently no-op or explode later, so that *does* stop startup — with a
  message naming the patch and the symbol.
* **Version range (warning only).**  Outside the tested range you get a logged
  warning telling you to re-verify, not a crash.

HOW TO REMOVE A PATCH
----------------------
1. Confirm the upstream bug is fixed (run the corresponding test_patches.py test).
2. Delete the patch function, its call in apply_all(), and its _PATCH_TARGETS entry.
3. Widen TESTED_CREWAI_RANGE to the version you verified against.
"""

import os
import re as _re
import logging
import functools

import crewai

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version range — widen the upper bound after verifying all patches on a new
# release.  Being outside this range warns; it does not stop the app.
# ---------------------------------------------------------------------------
TESTED_CREWAI_RANGE = ("1.14.5", "1.15.14")


def _version_tuple(v: str) -> tuple:
    """Parse a version string into a comparable tuple, ignoring any suffix."""
    parts = []
    for chunk in v.split(".")[:3]:
        digits = _re.match(r"\d+", chunk)
        parts.append(int(digits.group()) if digits else 0)
    return tuple(parts)


def _check_version_range() -> None:
    """Warn (never raise) when the installed crewai is outside the tested range."""
    low, high = TESTED_CREWAI_RANGE
    installed = crewai.__version__
    if not (_version_tuple(low) <= _version_tuple(installed) <= _version_tuple(high)):
        _logger.warning(
            "CrewAI %s is outside the tested range %s–%s. All patch targets were "
            "verified present, so startup continues — but re-check patches.py "
            "against this release and widen TESTED_CREWAI_RANGE once confirmed.",
            installed, low, high,
        )


# ---------------------------------------------------------------------------
# Structural guard — (patch id, module path, class name or None, attribute).
# Every symbol a patch reaches into is listed here and verified before any
# patch is applied.  This is what actually protects against an upstream refactor.
# ---------------------------------------------------------------------------
_PATCH_TARGETS = [
    ("P1", "crewai.llms.providers.openai_compatible.completion",
     "OpenAICompatibleCompletion", "supports_function_calling"),
    ("P2", "crewai.utilities.reasoning_handler", "AgentReasoning", "_call_with_function"),
    ("P2", "crewai.utilities.reasoning_handler", "AgentReasoning", "_parse_planning_response"),
    ("P2", "crewai.agent.utils", None, "handle_reasoning"),
    ("P3", "crewai.utilities.exceptions.context_window_exceeding_exception",
     None, "CONTEXT_LIMIT_ERRORS"),
    ("P3", "crewai.utilities.agent_utils", None, "is_context_length_exceeded"),
    ("P3", "crewai.utilities.agent_utils", None, "handle_context_length"),
    ("P3", "crewai.agents.crew_agent_executor", "CrewAgentExecutor", "_invoke_loop_native_tools"),
    ("P3", "crewai.agents.crew_agent_executor", "CrewAgentExecutor", "_invoke_loop_react"),
    ("P3", "crewai.agents.crew_agent_executor", "CrewAgentExecutor", "_invoke_loop_native_no_tools"),
    ("P4", "crewai.memory.unified_memory", "Memory", "recall"),
    ("P5", "crewai.agents.agent_builder.base_agent_executor", "BaseAgentExecutor", "_save_to_memory"),
]


def verify_patch_targets() -> list[str]:
    """Return a list of human-readable descriptions of any missing patch targets.

    An empty list means every symbol the patches depend on is still present.
    """
    import importlib

    missing: list[str] = []
    for patch_id, module_path, class_name, attr in _PATCH_TARGETS:
        location = f"{module_path}.{class_name}.{attr}" if class_name else f"{module_path}.{attr}"
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            missing.append(f"{patch_id}: cannot import {module_path} ({e})")
            continue
        owner = module
        if class_name:
            owner = getattr(module, class_name, None)
            if owner is None:
                missing.append(f"{patch_id}: {module_path}.{class_name} no longer exists")
                continue
        if not hasattr(owner, attr):
            missing.append(f"{patch_id}: {location} no longer exists")
    return missing


def _check_patch_targets() -> None:
    """Raise a clear, actionable error if any patch target has disappeared."""
    missing = verify_patch_targets()
    if missing:
        detail = "\n  - ".join(missing)
        raise RuntimeError(
            f"CrewAI {crewai.__version__} has moved or removed symbols that "
            f"researcher.patches depends on:\n  - {detail}\n"
            f"Review each affected patch in patches.py — the upstream bug may be "
            f"fixed (delete the patch) or the API may have moved (update the target)."
        )


# ---------------------------------------------------------------------------
# Executor selection — REACHABILITY, not just existence.
#
# CrewAI 1.15.14 changed Agent's default executor from CrewAgentExecutor to the
# Flow-based crewai.experimental.AgentExecutor.  The P3 patches wrap
# CrewAgentExecutor._invoke_loop_* — those methods still *exist* on the new
# version, so the structural check above passes, but they are never *called*
# under the new default.  The effect is silent: context-length errors stop being
# summarised and retried, and the ReAct loop behaves differently.
#
# Agents therefore pin the executor the patches were written and tested against.
# Set CREWAI_EXECUTOR=experimental to opt into the new one once P3 is ported.
# ---------------------------------------------------------------------------
from crewai.agents.crew_agent_executor import CrewAgentExecutor as _CrewAgentExecutor  # noqa: E402

_EXECUTOR_CHOICE = os.getenv("CREWAI_EXECUTOR", "classic").strip().lower()


def get_executor_class():
    """Return the executor class agents should use.

    Defaults to the classic CrewAgentExecutor, which every patch here targets.
    Returns None to mean "use whatever CrewAI defaults to" when the caller has
    explicitly opted into the experimental executor.
    """
    if _EXECUTOR_CHOICE == "experimental":
        return None
    return _CrewAgentExecutor


def executor_kwargs() -> dict:
    """Agent(**executor_kwargs()) pins the executor the patches target.

    Empty when the caller has opted into CrewAI's own default via
    CREWAI_EXECUTOR=experimental.
    """
    cls = get_executor_class()
    return {"executor_class": cls} if cls is not None else {}


def _check_executor_reachability() -> None:
    """Warn when CrewAI's default executor is not the one the patches target."""
    try:
        from crewai import Agent

        default_cls = Agent.model_fields["executor_class"].default
    except Exception:  # pragma: no cover - defensive
        return
    if default_cls is not _CrewAgentExecutor and _EXECUTOR_CHOICE != "experimental":
        _logger.info(
            "CrewAI's default executor is %s, but patches P3/P5 target %s. "
            "Agents pin the latter via patches.get_executor_class(); set "
            "CREWAI_EXECUTOR=experimental to opt out once P3 is ported.",
            default_cls.__name__, _CrewAgentExecutor.__name__,
        )


_check_patch_targets()
_check_version_range()
_check_executor_reachability()


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
