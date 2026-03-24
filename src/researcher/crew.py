import os
import gc
import re as _re
import json as _json
import uuid as _uuid
import logging
import threading
from pathlib import Path
from crewai import Agent, Crew, Process, Task, LLM
from crewai.memory import Memory
from crewai.agent.planning_config import PlanningConfig
from crewai.project import CrewBase, agent, crew, task

from crewai.tools import tool
from crewai_tools import SerperDevTool

from langchain_community.tools import DuckDuckGoSearchRun
from PIL import Image, ImageDraw, ImageFont

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monkey-patch: CrewAI's READY detection requires the EXACT substring
#   "READY: I am ready to execute the task."
# but qwen3.5 writes "READY", "**READY**", etc.  Patch both the text-parsing
# path (_parse_planning_response) and the function-calling fallback path
# (_call_with_function) to use lenient detection.
# ---------------------------------------------------------------------------
_READY_RE = _re.compile(r'(?m)^\s*(?:\*\s*)*\*{0,2}READY\*{0,2}\s*:?[^\n]*$')


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
    "exceeds the context length",   # Ollama via litellm
    "input is too long",            # some providers
    "prompt is too long",           # some providers
]
for _p in _EXTRA_CTX_PATTERNS:
    if _p not in CONTEXT_LIMIT_ERRORS:
        CONTEXT_LIMIT_ERRORS.append(_p)

from crewai.utilities.agent_utils import is_context_length_exceeded  # noqa: E402

import crewai.agents.crew_agent_executor as _executor_mod  # noqa: E402

# Store original methods
_orig_invoke_loop_native_tools = _executor_mod.CrewAgentExecutor._invoke_loop_native_tools
_orig_invoke_loop_react = _executor_mod.CrewAgentExecutor._invoke_loop_react
_orig_invoke_loop_native_no_tools = _executor_mod.CrewAgentExecutor._invoke_loop_native_no_tools

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
_logger.info("Patched CrewAgentExecutor: context-length errors now handled before litellm re-raise")
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
_logger.info("Patched Memory.recall: depth=%s, %d entries, %d chars max",
             memory_depth,
             _MEMORY_RECALL_LIMIT, _MAX_MEMORY_ENTRY_CHARS)

# --- Save: truncate before LLM extraction ---
from crewai.agents.agent_builder.base_agent_executor_mixin import (  # noqa: E402
    CrewAgentExecutorMixin,
)

_orig_save_to_memory = CrewAgentExecutorMixin._save_to_memory


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
        raw = (
            f"Task: {desc}\n"
            f"Agent: {self.agent.role}\n"
            f"Result: {result_text}"
        )
        # Cap total size
        if len(raw) > _MAX_SAVE_CONTENT_CHARS:
            raw = raw[:_MAX_SAVE_CONTENT_CHARS]
        extracted = memory.extract_memories(raw)
        if extracted:
            memory.remember_many(extracted, agent_role=self.agent.role)
    except Exception as e:
        self.agent._logger.log("error", f"Failed to save to memory: {e}")


CrewAgentExecutorMixin._save_to_memory = _patched_save_to_memory
_logger.info("Patched _save_to_memory: content capped at %d chars", _MAX_SAVE_CONTENT_CHARS)
# ---------------------------------------------------------------------------

# --- Smart TrueType font discovery ---
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian / Ubuntu / openSUSE
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",  # Fedora / RHEL
    "/usr/share/fonts/truetype/DejaVuSans.ttf",  # openSUSE alt
    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # Generic Linux
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _find_truetype_font() -> str | None:
    for p in _FONT_SEARCH_PATHS:
        if Path(p).is_file():
            return p
    _logger.warning(
        "No TrueType font found in standard paths; "
        "text rendering will use Pillow default bitmap font"
    )
    return None


_TRUETYPE_FONT_PATH = _find_truetype_font()

# Instantiate once, reuse across calls
_ddg_search = DuckDuckGoSearchRun()
_serper_raw = SerperDevTool()

_GENERATED_DIR = Path(__file__).parent / "static" / "generated"
_GENERATED_DIR.mkdir(parents=True, exist_ok=True)


@tool("InternetSearch")
def serper_search_wrapped(search_query: str) -> str:
    """Search the internet using Google (Serper). Input must be a single search query string."""
    # LLMs sometimes send JSON or arrays instead of a plain string — handle gracefully
    q = search_query.strip()
    if q.startswith("[") or q.startswith("{"):
        try:
            parsed = _json.loads(q)
            if isinstance(parsed, list):
                results = []
                for item in parsed:
                    sq = (
                        item.get("search_query", "")
                        if isinstance(item, dict)
                        else str(item)
                    )
                    if sq:
                        results.append(_serper_raw.run(search_query=sq))
                return "\n---\n".join(results)
            elif isinstance(parsed, dict) and "search_query" in parsed:
                q = parsed["search_query"]
        except (_json.JSONDecodeError, Exception):
            pass
    return _serper_raw.run(search_query=q)


@tool("DuckDuckGoSearch")
def ddg_search_wrapped(query: str) -> str:
    """Search the web using DuckDuckGo. Use as fallback if Serper fails. Input must be a single search query string."""
    q = query.strip()
    if q.startswith("[") or q.startswith("{"):
        try:
            parsed = _json.loads(q)
            if isinstance(parsed, list):
                results = []
                for item in parsed:
                    sq = (
                        item.get("query", item.get("search_query", ""))
                        if isinstance(item, dict)
                        else str(item)
                    )
                    if sq:
                        results.append(_ddg_search.run(sq))
                return "\n---\n".join(results)
            elif isinstance(parsed, dict):
                q = parsed.get("query", parsed.get("search_query", q))
        except (_json.JSONDecodeError, Exception):
            pass
    return _ddg_search.run(q)


@tool("GenerateImage")
def generate_image(instructions: str) -> str:
    """Draw geometric shapes. Input: JSON string with width, height, background, shapes array.
    Shape types: rectangle, circle, triangle, polygon, line, text. Copy the returned image tag into Final Answer.
    """
    try:
        spec = _json.loads(instructions)
    except _json.JSONDecodeError:
        return "Error: instructions must be valid JSON"
    w = min(int(spec.get("width", 512)), 2048)
    h = min(int(spec.get("height", 512)), 2048)
    bg = spec.get("background", "white")
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for shape in spec.get("shapes", []):
        t = shape.get("type", "")
        fill = shape.get("fill", None)
        outline = shape.get("outline", None)
        if t == "rectangle":
            x, y = int(shape.get("x", 0)), int(shape.get("y", 0))
            sw, sh = int(shape.get("width", 100)), int(shape.get("height", 100))
            draw.rectangle([x, y, x + sw, y + sh], fill=fill, outline=outline)
        elif t == "circle":
            cx, cy = int(shape.get("cx", 100)), int(shape.get("cy", 100))
            r = int(shape.get("radius", 50))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline)
        elif t == "line":
            x1, y1 = int(shape.get("x1", 0)), int(shape.get("y1", 0))
            x2, y2 = int(shape.get("x2", 100)), int(shape.get("y2", 100))
            lw = int(shape.get("width", 2))
            draw.line([x1, y1, x2, y2], fill=fill or "black", width=lw)
        elif t in ("triangle", "polygon"):
            pts = shape.get("points", shape.get("vertices", []))
            if pts and len(pts) >= 3:
                flat = [tuple(p) for p in pts]
                draw.polygon(flat, fill=fill, outline=outline)
        elif t == "text":
            x, y = int(shape.get("x", 10)), int(shape.get("y", 10))
            txt = str(shape.get("text", ""))
            size = int(shape.get("size", 20))
            if _TRUETYPE_FONT_PATH:
                try:
                    font = ImageFont.truetype(_TRUETYPE_FONT_PATH, size)
                except (IOError, OSError):
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
            draw.text((x, y), txt, fill=fill or "black", font=font)
    fname = f"{_uuid.uuid4().hex[:12]}.png"
    img.save(_GENERATED_DIR / fname)
    return f"![generated image](/static/generated/{fname})"


# --------------- Stable Diffusion AI image generation ---------------
_sd_pipe = None
_sd_lock = threading.Lock()  # guards lazy pipeline init
_vram_lock = threading.Lock()  # serialises GPU-heavy inference
_SD_MODEL_ID = os.getenv("SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")


def _get_sd_pipe():
    """Lazy-load SD pipeline. Uses CPU offload so VRAM is only used during generation."""
    global _sd_pipe
    if _sd_pipe is None:
        with _sd_lock:
            if _sd_pipe is None:  # double-check
                import torch
                from diffusers import StableDiffusionPipeline

                import sys as _sys
                import warnings
                print("[SD] Loading Stable Diffusion pipeline…", flush=True)
                # Suppress the harmless "position_ids UNEXPECTED" load report
                _real_stdout = _sys.stdout
                _sys.stdout = open(os.devnull, "w")
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*position_ids.*")
                        _sd_pipe = StableDiffusionPipeline.from_pretrained(
                            _SD_MODEL_ID,
                            torch_dtype=torch.float16,
                            safety_checker=None,
                            requires_safety_checker=False,
                        )
                finally:
                    _sys.stdout.close()
                    _sys.stdout = _real_stdout
                _sd_pipe.enable_model_cpu_offload()
                print("[SD] Pipeline ready.", flush=True)
    return _sd_pipe


def preload_sd():
    """Call from the server to warm up the SD pipeline in a background thread."""
    threading.Thread(target=_get_sd_pipe, daemon=True).start()


@tool("GenerateAIImage")
def generate_ai_image(prompt: str) -> str:
    """Generate a realistic image from a text prompt using Stable Diffusion AI.
    Use for animals, landscapes, people, objects, scenes. Add style keywords like photorealistic, 8k.
    Copy the returned image tag EXACTLY into your Final Answer."""
    import sys as _sys

    _sys.stderr.write(f"[SD] generate_ai_image called: {prompt[:100]}\n")
    _sys.stderr.flush()
    try:
        pipe = _get_sd_pipe()
        _sys.stderr.write("[SD] Pipeline acquired, acquiring VRAM lock...\n")
        _sys.stderr.flush()
        with _vram_lock:
            _sys.stderr.write("[SD] Starting inference...\n")
            _sys.stderr.flush()
            result = pipe(
                prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                width=512,
                height=512,
            )
            img = result.images[0]
            # Free GPU memory inside the lock so nothing else grabs VRAM first
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        fname = f"{_uuid.uuid4().hex[:12]}.png"
        img.save(_GENERATED_DIR / fname)
        tag = f"![generated image](/static/generated/{fname})"
        _sys.stderr.write(f"[SD] Done: {tag}\n")
        _sys.stderr.flush()
        return tag
    except Exception as e:
        _sys.stderr.write(f"[SD] ERROR: {e}\n")
        _sys.stderr.flush()
        return f"Error generating image: {e}"


@CrewBase
class ResearchCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, model: str | None = None):
        self._model_name = model or os.getenv("MODEL", "ollama/qwen3.5:9b")
        self.ollama_llm = self._make_llm(self._model_name)
        self._is_multi_part = False

    @staticmethod
    def _make_llm(model: str) -> LLM:
        is_thinking_model = any(k in model.lower() for k in ("qwen3", "deepseek-r1"))
        extra: dict = {}
        if is_thinking_model:
            # Let thinking models reason internally; CrewAI strips <think> tags.
            # Forcing enable_thinking=False on qwen3 can cause empty responses.
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        return LLM(
            model=model,
            base_url="http://localhost:11434",
            config={
                "options": {
                    "num_ctx": 196608,
                    "temperature": 0.4,
                    "num_gpu": 99,
                    "num_predict": 131072,
                }
            },
            **extra,
        )

    @agent
    def researcher(self) -> Agent:
        planning = PlanningConfig(
            max_attempts=3,
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
                "need to search the internet unless the task explicitly requires it.\n\n"
                "End your response with EXACTLY this line:\n"
                "READY: I am ready to execute the task.\n"
            ),
        )
        return Agent(
            config=self.agents_config["researcher"],
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
        embedder_config = {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "url": "http://localhost:11434/api/embeddings",
            },
        }
        memory = Memory(
            llm=self.ollama_llm,
            embedder=embedder_config,
        )
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=memory,
            verbose=True,
            embedder=embedder_config,
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
            "Each part description must be a COMPLETE, self-contained task "
            "description with all relevant context, requirements, and "
            "formatting instructions from the original request."
        )

        # Use a lightweight LLM config (no thinking, small context)
        decompose_llm = LLM(
            model=self._model_name,
            base_url="http://localhost:11434",
            config={
                "options": {
                    "num_ctx": 8192,
                    "temperature": 0.1,
                    "num_gpu": 99,
                    "num_predict": 2048,
                }
            },
        )

        try:
            response = decompose_llm.call(
                [{"role": "user", "content": prompt}]
            )
            # Strip <think> tags and markdown fences
            text = _re.sub(r'<think>[\s\S]*?</think>\s*', '', response).strip()
            text = _re.sub(r'^```(?:json)?\s*', '', text)
            text = _re.sub(r'\s*```$', '', text)
            data = _json.loads(text)
            if (
                data.get("split")
                and isinstance(data.get("parts"), list)
                and len(data["parts"]) > 1
            ):
                _logger.info(
                    "Decomposed request into %d sub-tasks", len(data["parts"])
                )
                return data["parts"]
        except Exception as e:
            _logger.warning(
                "Decomposition failed (running as single task): %s", e
            )
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
        decomposed into independent sub-tasks so each gets a full
        generation budget (num_predict tokens).
        """
        parts = self._decompose(topic)
        self._is_multi_part = len(parts) > 1

        agent_instance = self.researcher()

        if not self._is_multi_part:
            # ---- single task (same as original) ----
            memory = Memory(llm=self.ollama_llm, embedder=self._EMBEDDER_CONFIG)
            desc = (
                f'Analyze the user\'s request: "{topic}".\n'
                "First, determine if you can answer from your own knowledge.\n"
                "If not, determine if it's a system question (use LocalSystemCheck) "
                "or a world question (use InternetSearch).\n\n"
                "LENGTH AND THOROUGHNESS:\n"
                'When the user specifies a page count, word count, or says '
                '"thorough" / "detailed", you MUST produce the full requested '
                "length. One page = 500 words of dense text. Do NOT summarise "
                "or condense. Fill the requested length with substantive content."
            )
            exp = self.tasks_config["research_task"]["expected_output"]
            tasks = [Task(description=desc, expected_output=exp,
                          agent=agent_instance)]
            return Crew(
                agents=[agent_instance],
                tasks=tasks,
                process=Process.sequential,
                memory=memory,
                verbose=True,
                embedder=self._EMBEDDER_CONFIG,
            )
        else:
            # ---- multi-part: one sub-task per part ----
            _logger.info("Building multi-part crew with %d sub-tasks", len(parts))
            memory = Memory(llm=self.ollama_llm, embedder=self._EMBEDDER_CONFIG)
            tasks = []
            for i, part in enumerate(parts, 1):
                desc = (
                    f'Analyze the user\'s request: "{part}".\n'
                    "First, determine if you can answer from your own knowledge.\n"
                    "If not, determine if it's a system question (use LocalSystemCheck) "
                    "or a world question (use InternetSearch).\n\n"
                    f"ORIGINAL REQUEST (for context only):\n{topic}\n\n"
                    "LENGTH AND THOROUGHNESS:\n"
                    'When the user specifies a page count, word count, or says '
                    '"thorough" / "detailed", you MUST produce the full requested '
                    "length. One page = 500 words of dense text. Do NOT summarise "
                    "or condense. Fill the requested length with substantive content."
                )
                exp = self.tasks_config["research_task"]["expected_output"]
                tasks.append(
                    Task(description=desc, expected_output=exp,
                         agent=agent_instance)
                )
            return Crew(
                agents=[agent_instance],
                tasks=tasks,
                process=Process.sequential,
                memory=memory,
                verbose=True,
                embedder=self._EMBEDDER_CONFIG,
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
