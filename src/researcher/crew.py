import os
import gc
import re as _re
import json as _json
import logging
import threading
from pathlib import Path
from crewai import Agent, Crew, Process, Task, LLM
from crewai.memory import Memory
from crewai.agent.planning_config import PlanningConfig
from crewai.project import CrewBase, agent, crew, task

from researcher.tools import (
    serper_search_wrapped,
    ddg_search_wrapped,
    generate_image,
)
from researcher.image import (
    generate_ai_image,
    preload_sd,
    preload_zimage,
    _image_was_generated,
    IMAGE_BACKEND,
    OLLAMA_IMAGE_URL,
)
from researcher.config import probe_model_capabilities, TOOL_CAPABLE_MODEL

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Monkey-patch: Force ReAct text-based tool calling for Ollama models.
#
# Problem: CrewAI calls llm.supports_function_calling() which returns True
# for qwen3.5 via litellm/Ollama.  In native mode, CrewAI sends OpenAI-style
# tool schemas and expects structured tool_call JSON objects back.  But
# qwen3.5 (and many Ollama models) returns PLAIN TEXT describing tool calls
# instead of structured objects — so CrewAI treats it as "final answer"
# and never actually executes the tool.
#
# Fix: Override supports_function_calling() to return False for Ollama
# models, forcing CrewAI into ReAct mode (Action: / Action Input: text
# pattern) which works reliably with these models.
#
# NOTE: CrewAI's LLM.__new__ routes "ollama/" models to
# OpenAICompatibleCompletion (not the LLM class itself).  The runtime
# instance has provider="ollama" and model="qwen3.5:9b" (prefix stripped).
# We must patch OpenAICompatibleCompletion — patching LLM has no effect.
# ---------------------------------------------------------------------------
from crewai.llms.providers.openai_compatible.completion import (  # noqa: E402
    OpenAICompatibleCompletion as _OAICompat,
)

_orig_supports_fc = _OAICompat.supports_function_calling


def _patched_supports_fc(self):
    """Return False for Ollama models to force ReAct text-based tool calling.

    Instances with ``_allow_fc = True`` (e.g. the planning LLM) are
    exempted because the planning JSON-schema tool works fine with Ollama.
    """
    if getattr(self, "_allow_fc", False):
        return _orig_supports_fc(self)
    provider = getattr(self, "provider", "") or ""
    if provider == "ollama":
        return False
    return _orig_supports_fc(self)


_OAICompat.supports_function_calling = _patched_supports_fc
_logger.info("Patched OpenAICompatibleCompletion.supports_function_calling: Ollama → ReAct")
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Monkey-patch: CrewAI's READY detection requires the EXACT substring
#   "READY: I am ready to execute the task."
# but qwen3.5 writes "READY", "**READY**", etc.  Patch both the text-parsing
# path (_parse_planning_response) and the function-calling fallback path
# (_call_with_function) to use lenient detection.
# ---------------------------------------------------------------------------
_READY_RE = _re.compile(r"(?m)^\s*(?:\*\s*)*\*{0,2}READY\*{0,2}\s*:?[^\n]*$")


def _lenient_ready(text: str) -> bool:
    """Return True if *text* contains a READY signal in any common format."""
    if "READY: I am ready to execute the task." in text:
        return True
    return bool(_READY_RE.search(text))


from crewai.utilities.reasoning_handler import AgentReasoning  # noqa: E402


# 1) Text-parsing path
@staticmethod
def _patched_parse(response: str) -> tuple[str, bool]:
    if not response:
        return "No plan was generated.", False
    return response, _lenient_ready(response)


AgentReasoning._parse_planning_response = _patched_parse

# 2) Function-calling path (wrap to fix ready flag on fallback)
_orig_call_with_function = AgentReasoning._call_with_function


def _patched_call_with_function(self, prompt, plan_type):
    plan, steps, ready = _orig_call_with_function(self, prompt, plan_type)
    if not ready:
        ready = _lenient_ready(plan)
    return plan, steps, ready


AgentReasoning._call_with_function = _patched_call_with_function

# 3) Prevent plan stacking: each planning round appends to task.description
#    but on re-execution the old plans stay → exponential growth.
#    Also cap plan length so the LLM's verbose tables/matrices don't blow up context.
_MAX_PLAN_CHARS = 2000

import crewai.agent.utils as _agent_utils_mod  # noqa: E402

_orig_handle_reasoning = _agent_utils_mod.handle_reasoning


def _capped_handle_reasoning(agent, task):
    """Strip stale plans, run planning, then cap the appended plan text."""
    # Remove any previously appended plan text — only keep the newest
    if "\n\nPlanning:\n" in task.description:
        task.description = task.description.split("\n\nPlanning:\n")[0]
    _orig_handle_reasoning(agent, task)
    # Now trim if the plan is too long
    if "\n\nPlanning:\n" in task.description:
        base, plan_text = task.description.split("\n\nPlanning:\n", 1)
        if len(plan_text) > _MAX_PLAN_CHARS:
            plan_text = plan_text[:_MAX_PLAN_CHARS] + "\n[plan truncated]"
        task.description = base + "\n\nPlanning:\n" + plan_text


_agent_utils_mod.handle_reasoning = _capped_handle_reasoning
_logger.info("Patched handle_reasoning: no stacking, max %d chars", _MAX_PLAN_CHARS)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Monkey-patch: CrewAI executor bug — litellm errors are re-raised BEFORE
# is_context_length_exceeded() is checked.  This means Ollama context-length
# errors (which come through litellm) bypass the summarise-and-retry logic
# of respect_context_window=True.
#
# Additionally, the error string "the input length exceeds the context length"
# (from Ollama) does NOT match any pattern in CONTEXT_LIMIT_ERRORS because
# the word order is different ("exceeds the context length" vs "context length
# exceeded").
#
# Fix both: add the missing pattern, then swap the check order so context-
# length errors are caught before the generic litellm re-raise.
# ---------------------------------------------------------------------------
from crewai.utilities.exceptions.context_window_exceeding_exception import (  # noqa: E402
    CONTEXT_LIMIT_ERRORS,
)

_EXTRA_CTX_PATTERNS = [
    "exceeds the context length",  # Ollama via litellm
    "input is too long",  # some providers
    "prompt is too long",  # some providers
]
for _p in _EXTRA_CTX_PATTERNS:
    if _p not in CONTEXT_LIMIT_ERRORS:
        CONTEXT_LIMIT_ERRORS.append(_p)

from crewai.utilities.agent_utils import is_context_length_exceeded  # noqa: E402

import crewai.agents.crew_agent_executor as _executor_mod  # noqa: E402

# Store original methods
_orig_invoke_loop_native_tools = (
    _executor_mod.CrewAgentExecutor._invoke_loop_native_tools
)
_orig_invoke_loop_react = _executor_mod.CrewAgentExecutor._invoke_loop_react
_orig_invoke_loop_native_no_tools = (
    _executor_mod.CrewAgentExecutor._invoke_loop_native_no_tools
)


def _patch_exception_handler(original_method):
    """Wrap an invoke-loop method so context-length errors are checked
    BEFORE the blanket litellm re-raise."""
    import functools

    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        try:
            return original_method(self, *args, **kwargs)
        except Exception as e:
            # Check context length FIRST (before litellm re-raise)
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
                # Retry once after summarisation
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
_logger.info(
    "Patched CrewAgentExecutor: context-length errors now handled before litellm re-raise"
)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Monkey-patch: memory recall + save fixes for Ollama context limits.
#
# RECALL issues:
#   - Default depth="deep" triggers RecallFlow (multiple LLM calls → 50s).
#     Switch to "shallow" (direct vector search → 1-2s).
#   - Default limit=5 with large stored entries bloats the prompt.
#     Cap at 3 entries, truncate each to 500 chars.
#
# SAVE issues:
#   - _save_to_memory feeds full task description + expected output + full
#     result to extract_memories() → LLM context overflow → memory_save_failed.
#     Truncate the raw content to a reasonable size before extraction.
# ---------------------------------------------------------------------------
_MAX_MEMORY_ENTRY_CHARS = 1000
_MEMORY_RECALL_LIMIT = 5
_MAX_SAVE_CONTENT_CHARS = 6000  # ~1500 tokens — enough for LLM to extract key facts

# Runtime-switchable memory depth: "shallow" (fast, default) or "deep" (slow but thorough)
memory_depth: str = "shallow"

from crewai.memory.unified_memory import Memory as _UnifiedMemory  # noqa: E402

# --- Recall: depth-switchable + capped ---
_orig_memory_recall = _UnifiedMemory.recall


def _capped_recall(self, query, limit=10, **kwargs):
    """Recall with runtime-configurable depth, capped results, and truncation."""
    import researcher.crew as _crew_mod

    depth = _crew_mod.memory_depth
    kwargs["depth"] = depth
    cap = _MEMORY_RECALL_LIMIT if depth == "shallow" else min(limit, 5)
    char_limit = _MAX_MEMORY_ENTRY_CHARS if depth == "shallow" else 2000
    # Truncate the query to prevent embedding model context overflow
    # (nomic-embed-text has 8192 token context; ~4 chars/token → 30k chars safe)
    if isinstance(query, str) and len(query) > 4000:
        query = query[:4000]
    matches = _orig_memory_recall(self, query, limit=cap, **kwargs)
    for m in matches:
        if len(m.record.content) > char_limit:
            m.record.content = m.record.content[:char_limit] + "…"
    return matches


_UnifiedMemory.recall = _capped_recall
_logger.info(
    "Patched Memory.recall: depth=%s, %d entries, %d chars max",
    memory_depth,
    _MEMORY_RECALL_LIMIT,
    _MAX_MEMORY_ENTRY_CHARS,
)

# --- Save: truncate before LLM extraction ---
from crewai.agents.agent_builder.base_agent_executor import (  # noqa: E402
    BaseAgentExecutor as _BaseAgentExecutor,
)

_orig_save_to_memory = _BaseAgentExecutor._save_to_memory


def _patched_save_to_memory(self, output) -> None:
    """Save to memory with truncated content to avoid LLM context overflow."""
    from crewai.utilities.string_utils import sanitize_tool_name

    memory = getattr(self.agent, "memory", None) or (
        getattr(self.crew, "_memory", None) if self.crew else None
    )
    if memory is None or not self.task or getattr(memory, "read_only", False):
        return
    if f"Action: {sanitize_tool_name('Delegate work to coworker')}" in output.text:
        return
    try:
        # Build a BRIEF summary for extraction — don't feed the full essay
        desc = (self.task.description or "")[:500]
        result_text = (output.text or "")[:2000]
        raw = f"Task: {desc}\n" f"Agent: {self.agent.role}\n" f"Result: {result_text}"
        # Cap total size
        if len(raw) > _MAX_SAVE_CONTENT_CHARS:
            raw = raw[:_MAX_SAVE_CONTENT_CHARS]
        extracted = memory.extract_memories(raw)
        if extracted:
            memory.remember_many(extracted, agent_role=self.agent.role)
    except Exception as e:
        self.agent._logger.log("error", f"Failed to save to memory: {e}")


_BaseAgentExecutor._save_to_memory = _patched_save_to_memory
_logger.info(
    "Patched _save_to_memory: content capped at %d chars", _MAX_SAVE_CONTENT_CHARS
)
# ---------------------------------------------------------------------------


@CrewBase
class ResearchCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Default LLM parameter values (exposed via /llm-params API)
    DEFAULT_LLM_PARAMS: dict = {
        "temperature": 0.4,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.0,
        "seed": 0,
        "num_predict": 4096,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "planning_max_attempts": 3,
    }

    def __init__(self, model: str | None = None):
        self._model_name = model or os.getenv("MODEL", "ollama/qwen3.5:9b")
        self._llm_params: dict = dict(self.DEFAULT_LLM_PARAMS)
        self.ollama_llm = self._make_llm(self._model_name, self._llm_params)
        self._model_caps = probe_model_capabilities(self._model_name)
        self._is_multi_part = False

    def get_llm_params(self) -> dict:
        """Return current LLM parameters."""
        return dict(self._llm_params)

    def update_llm_params(self, params: dict) -> dict:
        """Update LLM parameters and rebuild the LLM instance."""
        for k, v in params.items():
            if k in self.DEFAULT_LLM_PARAMS:
                self._llm_params[k] = v
        self.ollama_llm = self._make_llm(self._model_name, self._llm_params)
        return dict(self._llm_params)

    @staticmethod
    def _make_llm(model: str, params: dict | None = None) -> LLM:
        if params is None:
            params = ResearchCrew.DEFAULT_LLM_PARAMS
        is_thinking_model = any(k in model.lower() for k in ("qwen3", "deepseek-r1"))
        # Build extra_body with Ollama options (CrewAI 1.12 no longer accepts config=)
        seed = int(params.get("seed", 0))
        ollama_options = {
            "num_ctx": int(params.get("num_ctx", 32768)),
            "num_gpu": 99,
            "num_predict": int(params.get("num_predict", 4096)),
            "top_k": int(params.get("top_k", 40)),
            "top_p": float(params.get("top_p", 0.9)),
            "min_p": float(params.get("min_p", 0.0)),
            "repeat_penalty": float(params.get("repeat_penalty", 1.1)),
        }
        if seed > 0:
            ollama_options["seed"] = seed
        extra_body: dict = {"options": ollama_options}
        if is_thinking_model:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        return LLM(
            model=model,
            base_url="http://localhost:11434",
            temperature=float(params.get("temperature", 0.4)),
            frequency_penalty=float(params.get("frequency_penalty", 0.0)),
            presence_penalty=float(params.get("presence_penalty", 0.0)),
            extra_body=extra_body,
        )

    @staticmethod
    def _make_planning_llm(model: str) -> LLM:
        """Lightweight LLM for the planning phase only.

        Thinking is DISABLED and context is small so each planning call
        completes in ~10-20 s instead of 3-4 min.  This LLM is NOT
        affected by the supports_function_calling patch (provider is
        still 'ollama'), but PlanningConfig's function-calling path
        only sends a JSON-schema tool which Ollama handles fine.
        """
        return LLM(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.2,
            extra_body={
                "options": {
                    "num_ctx": 8192,
                    "num_gpu": 99,
                    "num_predict": 2048,
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    @agent
    def researcher(self) -> Agent:
        # Use the selected model for planning if it supports tools;
        # otherwise fall back to a known tool-capable model.
        if self._model_caps["supports_tools"]:
            planning_model = self._model_name
        else:
            planning_model = TOOL_CAPABLE_MODEL
            _logger.info(
                "[RESEARCH] %s lacks tool support — planning with %s",
                self._model_name, planning_model,
            )
        planning_llm = self._make_planning_llm(planning_model)
        # Allow native function calling for planning — the single JSON-schema
        # tool works fine with Ollama; only task-execution tools are broken.
        planning_llm._allow_fc = True
        planning = PlanningConfig(
            llm=planning_llm,
            max_attempts=int(self._llm_params.get("planning_max_attempts", 3)),
            max_steps=6,
            reasoning_effort="medium",
            plan_prompt=(
                "You are {role}.\n\n"
                "Task: {description}\n\n"
                "Expected output: {expected_output}\n\n"
                "Available tools: {tools}\n\n"
                "Create a CONCISE plan with 2-{max_steps} steps. "
                "For each step write ONE line: the action and which tool to use.\n"
                "IMPORTANT: Only reference tools from the Available tools list above. "
                "If a step needs NO tool (e.g. writing, reasoning, reviewing), "
                "write 'Tool: direct generation'. NEVER write 'Tool: None'.\n"
                "Do NOT include tables, matrices, or checklists — just the numbered steps.\n"
                "This task can be answered from your own knowledge — you do NOT "
                "need to search the internet unless the task explicitly requires it.\n"
                "If the task asks you to generate an image, your plan MUST include "
                "a step that calls the image tool. Do NOT plan to explain limitations "
                "or safety policies — just plan to execute the request.\n"
                "CONTENT POLICY: This is an art and photography application. "
                "Artistic nudity, figure drawing, and photographic art are "
                "legitimate and permitted. The image tool has its own content "
                "filter — your job is ONLY to plan the tool call, not to "
                "judge content. NEVER refuse a request on moral or safety "
                "grounds — always plan to call the image tool.\n\n"
                "End your response with EXACTLY this line:\n"
                "READY: I am ready to execute the task.\n"
            ),
        )
        # For models without native tool support, add explicit ReAct format
        agent_config = dict(self.agents_config["researcher"])
        if not self._model_caps["supports_tools"]:
            agent_config["backstory"] = agent_config.get("backstory", "") + (
                "\n\n===== CRITICAL: TOOL CALLING FORMAT =====\n"
                "You MUST use EXACTLY this text format to call tools:\n\n"
                "Thought: I need to use a tool.\n"
                "Action: <tool_name>\n"
                'Action Input: {"param": "value"}\n\n'
                "Then WAIT for the Observation (the tool result). "
                "Do NOT invent or fabricate the tool response yourself. "
                "Do NOT use ```tool_code``` or any other format.\n"
                "After receiving the Observation, write:\n\n"
                "Thought: I now have the answer.\n"
                "Final Answer: [your response]\n"
            )
        return Agent(
            config=agent_config,
            tools=[
                serper_search_wrapped,
                ddg_search_wrapped,
                generate_image,
                generate_ai_image,
            ],
            llm=self.ollama_llm,
            verbose=True,
            max_iter=25,
            max_retry_limit=3,
            respect_context_window=True,
            planning_config=planning,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # Dynamic task splitting: decompose complex requests into sub-tasks
    # ------------------------------------------------------------------

    def _decompose(self, topic: str) -> list[str]:
        """Use LLM to decide if a request should be split into parts.

        Returns a list with either one entry (no split) or multiple
        self-contained part descriptions.
        """
        # Skip decomposition for short / simple requests
        if len(topic) < 300:
            return [topic]

        prompt = (
            "You are a task planner. Analyze the user request below and decide "
            "whether it should be split into independent parts that can be "
            "written separately and then concatenated.\n\n"
            "SPLIT when the request:\n"
            "- Asks for content in MULTIPLE LANGUAGES (one part per language)\n"
            "- Asks for multiple INDEPENDENT long documents, chapters, or reports\n"
            "- Would produce more than ~3000 words total\n\n"
            "Do NOT split when:\n"
            "- It is a simple question or short task\n"
            "- The parts are interdependent (each needs context from others)\n"
            "- Total output would be under ~3000 words\n\n"
            f"Request:\n{topic[:2000]}\n\n"
            "Respond with ONLY valid JSON, no markdown fences, no explanation:\n"
            'If no split: {"split": false}\n'
            'If split: {"split": true, "parts": ["full description of '
            'part 1", "full description of part 2", ...]}\n\n'
            "CRITICAL RULES FOR EACH PART DESCRIPTION:\n"
            "- Each part must be COMPLETELY SELF-CONTAINED. It must include "
            "ALL requirements: format, length, style, topic, sources, etc.\n"
            "- Each part must specify ONLY what that part should produce.\n"
            "- Do NOT mention the other parts or the overall request.\n"
            "- Do NOT say 'this is part N of M'.\n"
            "- The person executing each part will ONLY see that part's "
            "description — they will have NO knowledge of other parts.\n"
            "- Copy ALL relevant constraints (word count, formatting, "
            "academic level, equation requirements, etc.) into EACH part."
        )

        # Use a lightweight LLM config (no thinking, small context)
        decompose_llm = LLM(
            model=self._model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            extra_body={
                "options": {
                    "num_ctx": 8192,
                    "num_gpu": 99,
                    "num_predict": 2048,
                }
            },
        )

        try:
            response = decompose_llm.call([{"role": "user", "content": prompt}])
            # Strip <think> tags and markdown fences
            text = _re.sub(r"<think>[\s\S]*?</think>\s*", "", response).strip()
            text = _re.sub(r"^```(?:json)?\s*", "", text)
            text = _re.sub(r"\s*```$", "", text)
            data = _json.loads(text)
            if (
                data.get("split")
                and isinstance(data.get("parts"), list)
                and len(data["parts"]) > 1
            ):
                _logger.info("Decomposed request into %d sub-tasks", len(data["parts"]))
                return data["parts"]
        except Exception as e:
            _logger.warning("Decomposition failed (running as single task): %s", e)
        return [topic]

    _EMBEDDER_CONFIG = {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "url": "http://localhost:11434/api/embeddings",
        },
    }

    def build_crew(self, topic: str) -> "Crew":
        """Build a Crew tailored for *topic*.

        Simple requests get a single task; complex ones are automatically
        decomposed into independent sub-tasks.  Each sub-task has
        ``context=[]`` so CrewAI does NOT forward previous task output
        into the next task — every task only sees its own description.
        """
        parts = self._decompose(topic)
        self._is_multi_part = len(parts) > 1

        agent_instance = self.researcher()
        exp = self.tasks_config["research_task"]["expected_output"]

        if self._is_multi_part:
            _logger.info("Building multi-part crew with %d sub-tasks", len(parts))

        tasks = []
        for part in parts:
            desc = (
                f'Analyze the user\'s request: "{part}".\n'
                "First, determine if you can answer from your own knowledge.\n"
                "If the request includes '=== ATTACHED FILE CONTENT' that IS the "
                "actual text extracted from the user's uploaded file(s). It is part "
                "of your prompt — read it directly and use it as your PRIMARY source. "
                "Do NOT say you cannot read or access the file.\n"
                "If not answerable from your knowledge or attached files, determine "
                "if it's a system question (use LocalSystemCheck) "
                "or a world question (use InternetSearch).\n\n"
                "CRITICAL TOOL EXECUTION RULE:\n"
                "If you decide to use a tool, you MUST actually execute it using "
                "the Action/Action Input format. NEVER write the tool name or "
                "input as plain text in your Final Answer. If your Final Answer "
                "contains text like 'InternetSearch — ...' or 'Input: {...}' "
                "that means you FAILED to execute the tool. You must STOP and "
                "actually call the tool first.\n\n"
                "LENGTH AND THOROUGHNESS:\n"
                "When the user specifies a page count, word count, or says "
                '"thorough" / "detailed", you MUST produce the full requested '
                "length. One page = 500 words of dense text. Do NOT summarise "
                "or condense. Fill the requested length with substantive content."
            )
            tasks.append(
                Task(
                    description=desc,
                    expected_output=exp,
                    agent=agent_instance,
                    context=[],
                )
            )

        return Crew(
            agents=[agent_instance],
            tasks=tasks,
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    def postprocess(self, result) -> str:
        """Merge sub-task outputs when the request was split.

        For single-task runs this is a no-op passthrough.
        """
        if self._is_multi_part and hasattr(result, "tasks_output"):
            outputs = []
            for t in result.tasks_output:
                raw = (t.raw or "").strip()
                if raw:
                    outputs.append(raw)
            if outputs:
                return "\n\n---\n\n".join(outputs)
        return result.raw if hasattr(result, "raw") else str(result)

    def reset_memory(self) -> None:
        """Clear the crew's memory store and kickoff task outputs so stale
        data cannot leak into future requests."""
        mem = getattr(self, "_memory", None)
        if mem is not None:
            try:
                mem.reset()
                _logger.info("Crew memory reset")
            except Exception as e:
                _logger.warning("Failed to reset memory: %s", e)

        # Also wipe the kickoff task outputs sqlite db (and stale WAL/SHM)
        try:
            from crewai.utilities.paths import db_storage_path as _db_storage_path

            db_dir = Path(_db_storage_path())
            for pattern in ("latest_kickoff_task_outputs.db*",):
                for f in db_dir.glob(pattern):
                    f.unlink(missing_ok=True)
        except Exception as e:
            _logger.warning("Failed to clean kickoff outputs: %s", e)
