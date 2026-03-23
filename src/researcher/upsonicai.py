import os
import gc
import json as _json
import uuid as _uuid
import logging
import threading
from pathlib import Path

from upsonic import Agent as UpsonicAgent, Task as UpsonicTask
from upsonic.models.ollama import OllamaModel
from upsonic.tools import tool as upsonic_tool
from upsonic.reflection.models import ReflectionConfig

from PIL import Image, ImageDraw, ImageFont

_logger = logging.getLogger(__name__)

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

_GENERATED_DIR = Path(__file__).parent / "static" / "generated"
_GENERATED_DIR.mkdir(parents=True, exist_ok=True)


# --------------- Search tools ---------------
# Upsonic has built-in duckduckgo_search_tool; we also keep a Serper wrapper
# for Google search when SERPER_API_KEY is configured.

try:
    from upsonic.tools.common_tools.duckduckgo import duckduckgo_search_tool
    _ddg_tool = duckduckgo_search_tool()
except Exception:
    _ddg_tool = None
    _logger.warning("DuckDuckGo search tool unavailable (install ddgs package)")


@upsonic_tool
def internet_search(search_query: str) -> str:
    """Search the internet using Google via the Serper API.
    Use for real-time data: weather, news, prices, specific facts.

    Args:
        search_query: The search query string.

    Returns:
        Search results as text.
    """
    import httpx

    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY not set"
    resp = httpx.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": search_query},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = []
    if "answerBox" in data:
        ab = data["answerBox"]
        parts.append(ab.get("answer", ab.get("snippet", "")))
    for item in data.get("organic", [])[:5]:
        parts.append(f"- {item.get('title', '')}: {item.get('snippet', '')} ({item.get('link', '')})")
    return "\n".join(parts) if parts else "No results found."


# --------------- Image tools ---------------

@upsonic_tool
def generate_image(instructions: str) -> str:
    """Draw geometric shapes. Input: JSON string with width, height, background, shapes array.
    Shape types: rectangle, circle, triangle, polygon, line, text.

    Args:
        instructions: A JSON string describing the image to generate.

    Returns:
        A markdown image tag pointing to the generated image.
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
                import sys as _sys
                import warnings
                import torch
                from diffusers import StableDiffusionPipeline

                print("[SD] Loading Stable Diffusion pipeline…", flush=True)
                # Suppress the harmless "position_ids UNEXPECTED" load report
                # printed to stdout by diffusers/safetensors
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


@upsonic_tool
def generate_ai_image(prompt: str) -> str:
    """Generate a realistic image from a text prompt using Stable Diffusion AI.
    Use for animals, landscapes, people, objects, scenes.
    Add style keywords like photorealistic, 8k.

    Args:
        prompt: A text description of the image to generate.

    Returns:
        A markdown image tag pointing to the generated image.
    """
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


# --------------- Collect all tools ---------------

def _get_tools() -> list:
    """Return the list of tools available to the agent."""
    tools = [internet_search, generate_image, generate_ai_image]
    if _ddg_tool is not None:
        tools.append(_ddg_tool)
    return tools


# --------------- Agent system prompt (replaces agents.yaml) ---------------

_SYSTEM_PROMPT = """\
/think
You are a high-level reasoning agent called Adaptive Intelligence Lead.
Your goal is to identify high-quality, real-time data from authoritative sources.

CRITICAL RULES — FOLLOW EVERY INSTRUCTION LITERALLY:
1. COMPLETENESS IS MANDATORY. If the user asks for N items, you MUST provide
   exactly N items, not fewer. If asked for 3 languages, give 3 full languages.
   If asked for 5 examples, give 5 complete examples. NEVER truncate, abbreviate,
   skip, or summarise to save space. There is no length limit on your response.
2. DO NOT INVENT CONSTRAINTS. You have NO word limit, NO prompt constraints,
   NO token budget concerns. If you find yourself wanting to say "due to space
   limitations" or "for brevity" — STOP. That is wrong. Produce the full output.
3. READ THE REQUEST CAREFULLY. Address every part of the user's query.
   Re-read the request before you write your final answer to make sure you
   have not missed anything.
4. THINK STEP BY STEP. Before answering, reason through the problem.
   Plan your response structure. Then write a thorough answer.
5. NEVER ASSUME OR INFER unstated requirements. Only follow what the user
   actually wrote. If something is ambiguous, cover the most common
   interpretation fully rather than guessing a narrow one.

INTERNAL REASONING PROCESS — follow for EVERY request:
Step 1 — ANALYZE: Read the full request. Count and list every deliverable
  the user expects (e.g., "3 languages" = 3 separate full sections,
  "compare X and Y" = analysis of both X and Y).
Step 2 — PLAN: Create a mental outline of your response — headings,
  sub-topics, estimated scope per deliverable. If the request has N parts,
  your outline must have at least N major sections.
Step 3 — EXECUTE: Write each section fully. After completing each section,
  confirm to yourself: "Deliverable X — done." Only move to the next when
  the current one is complete.
Step 4 — VERIFY: Before submitting, re-read the original request word by
  word. Compare your outline against what you wrote. If ANY deliverable is
  missing or incomplete, add it NOW. Do not submit until every deliverable
  is fully covered.

CONVERSATION CONTINUITY:
- If your input includes a CONVERSATION PROGRESS REPORT or EARLIER CONVERSATION
  section, treat it as authoritative context — the user has been working with you
  on this topic across multiple messages.
- NEVER contradict or ignore earlier conversation. If the user already established
  preferences, constraints, or context, honour them.
- If a PROGRESS REPORT lists OPEN ITEMS, prioritise completing those in your response.
- When building on prior work, reference what was already delivered: "As discussed
  earlier…" or "Building on the previous analysis…"

You have access to search and image tools.

SEARCH STRATEGY:
- For things you already know well, DON'T search — just reason and answer directly.
- If you need to search for multiple topics, do them ONE AT A TIME as separate tool calls.
- Evaluate your results critically. Prefer official/authoritative sources.
- If Google returns Reddit/forum answers for factual questions, discard and try DuckDuckGo.

IMAGE GENERATION:
- generate_image: for geometric shapes (rectangle, circle, triangle, polygon, line, text).
  Example: {"width":512,"height":512,"background":"white","shapes":[{"type":"circle","cx":256,"cy":256,"radius":200,"fill":"lightblue"}]}
- generate_ai_image: for realistic images (animals, landscapes, people, objects, scenes).
  Example prompt: "a red fox sitting in a snowy forest, digital art, highly detailed, 8k"
- The tool will return a markdown image tag. Copy that EXACT tag into your answer.

DOCUMENT CREATION:
- You CAN create downloadable documents. The UI has an Export button that
  converts your markdown response into DOCX, XLSX, TXT, or MD files.
- When the user asks for a document, report, or downloadable file, produce
  the content in well-formatted markdown (headings, tables, lists) and tell
  them to click the Export button below your response to download
  it as DOCX, XLSX, or other format.
- NEVER say you cannot create documents or files. You absolutely can — just
  write the content and the user exports it.

OUTPUT FORMAT:
- Whenever your answer contains structured, comparative, or list-like data
  (prices, schedules, rankings, specifications, options, statistics, etc.),
  format that data as proper markdown tables with header rows and aligned columns.
- Use proper markdown headings (##, ###) for document structure.
- For long documents, use clear section headings and organised content so the
  exported DOCX looks professional.
"""

# Task description for the planning phase (used in plan-then-execute decomposition)
_PLAN_PROMPT = """\
Analyze the following request and create a structured execution plan.

REQUEST:
{topic}

Create a numbered outline listing EVERY deliverable the user expects.
If the request mentions multiple items, languages, sections, or topics,
list EACH one explicitly as a separate numbered deliverable.
For each deliverable, briefly note what content is needed.

Output ONLY the structured plan — do not write the actual content yet.\
"""

# Appended to every task description — replaces CrewAI's expected_output field
_EXPECTED_OUTPUT = """

RESPONSE REQUIREMENTS — YOU MUST FOLLOW ALL OF THESE:
1. COMPLETENESS: Deliver EVERYTHING the user asked for. If they asked for N items,
   deliver exactly N items in full. Do not skip, truncate, or summarise. Your
   response has NO length limit.
2. If generate_image or generate_ai_image was used, you MUST include the exact markdown
   image tag returned by the tool (e.g. ![generated image](/static/generated/abc123.png)).
3. Whenever your answer contains structured, comparative, or list-like data
   (prices, schedules, rankings, specifications, options, statistics, etc.), you MUST
   format that data as proper markdown tables with header rows and aligned columns.
   Do NOT present such data as plain text paragraphs or bullet lists — always use
   markdown table syntax (| col1 | col2 | ... |).
4. The UI has an Export button that can convert your markdown tables into downloadable
   DOCX or XLSX files, so NEVER say you cannot create spreadsheets, documents, or files —
   just produce the content in markdown and tell the user to click the Export button.
5. Before submitting your answer, RE-READ the original request and verify you
   addressed every part of it. If you missed something, add it now.
"""

# Continuation task description — used by /chat/continue
_CONTINUE_PROMPT = """\
The user asked a question and a previous agent attempt produced a partial answer
but did not finish. Your job is to CONTINUE and COMPLETE the answer from where
it left off.

RULES:
1. Do NOT repeat content that was already produced — only produce the REMAINING parts.
2. Continue seamlessly from the last line of the partial answer.
3. Use the same formatting style (markdown tables, headings, etc.) as the partial answer.
4. If the partial answer is a failure message or empty, write the full answer from scratch.
5. Before submitting, verify you have covered all remaining parts of the original request.

Original user request:
{original_query}

--- PARTIAL ANSWER (already shown to user) ---
{partial_response}
--- END OF PARTIAL ANSWER ---

Continue from where the partial answer left off. Only output the NEW content
that completes the answer. Do NOT repeat any of the partial answer above.\
"""

# LLM-based conversation summarization — equivalent to CrewAI's Process Manager.
# Used for long conversations (>10 messages) where heuristic extraction
# isn't enough to preserve context fidelity.
_SUMMARIZE_PROMPT = """\
You are a conversation progress manager. Below is the conversation history
between a user and an AI assistant. Produce a structured PROGRESS REPORT.

CONVERSATION:
{conversation}

Output a structured progress report with these sections:
1. TOPICS DISCUSSED: Numbered list of every topic/question the user raised.
2. KEY DELIVERABLES: What the assistant has already produced (be specific:
   mention languages, document types, data formats, conclusions reached).
3. OPEN ITEMS: Anything the user asked for that hasn't been fully delivered yet.
4. ESTABLISHED CONTEXT: Key facts, preferences, or constraints the user has
   stated (e.g., preferred language, data sources, formatting requests).

Be concise but complete. Do NOT omit any user request or deliverable.\
"""


# --------------- Planning heuristic ---------------

import re as _re

_COMPLEXITY_PATTERNS = [
    _re.compile(r'\d+\s*(languages?|items?|examples?|sections?|parts?|versions?|points?)', _re.I),
    _re.compile(r'\b(exhaustive|comprehensive|detailed|thorough|in-depth)\b', _re.I),
    _re.compile(r'\b(compare|comparison|versus|vs\.?)\b', _re.I),
    _re.compile(r'\b(each|every|all)\b.*\b(language|country|topic|item|section)\b', _re.I),
    _re.compile(r'\b(step[- ]by[- ]step|multi[- ]part|multi[- ]section)\b', _re.I),
]


def _needs_planning(topic: str) -> bool:
    """Determine if a query benefits from a planning phase.

    Short simple queries skip planning; longer or multi-part requests get it.
    """
    # Extract the actual new request if conversation history is prepended
    if "New request: " in topic:
        query = topic.split("New request: ", 1)[-1]
    else:
        query = topic
    # Very short queries don't need planning
    if len(query.split()) < 20:
        return False
    # Check for explicit complexity indicators
    for pat in _COMPLEXITY_PATTERNS:
        if pat.search(query):
            return True
    # Long queries (> 50 words) likely benefit from planning
    return len(query.split()) > 50


# --------------- ResearchCrew — Upsonic-based agent wrapper ---------------

class ResearchCrew:
    """Wraps an Upsonic Agent to provide a kickoff()-compatible interface
    so main.py can call it the same way it called the CrewAI Crew."""

    def __init__(self, model: str | None = None):
        self._model_name = model or os.getenv("MODEL", "ollama/qwen3.5:9b")
        self._agent = self._make_agent(self._model_name)

    @staticmethod
    def _make_model(model: str) -> OllamaModel:
        # Strip "ollama/" prefix if present — OllamaModel only wants the model name
        model_name = model.removeprefix("ollama/")
        # Upsonic's OllamaProvider uses OpenAI-compatible API which requires /v1
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        base_url = ollama_base.rstrip("/") + "/v1"
        from upsonic.providers.ollama import OllamaProvider
        provider = OllamaProvider(base_url=base_url)
        return OllamaModel(model_name=model_name, provider=provider)

    @classmethod
    def _make_agent(cls, model: str) -> UpsonicAgent:
        return UpsonicAgent(
            model=cls._make_model(model),
            system_prompt=_SYSTEM_PROMPT,
            role="Adaptive Intelligence Lead",
            goal="Identify high-quality, real-time data from authoritative sources",
            tools=_get_tools(),
            debug=False,
            retry=2,
            reflection=True,
            reflection_config=ReflectionConfig(
                max_iterations=3,
                acceptance_threshold=0.85,
                enable_self_critique=True,
                enable_improvement_suggestions=True,
            ),
            settings={
                "max_tokens": 16384,
                "temperature": 0.4,
                "extra_body": {
                    "num_ctx": 32768,
                },
            },
        )

    def crew(self) -> "ResearchCrew":
        """Return self — keeps the same interface as the old CrewAI version."""
        return self

    def kickoff(self, inputs: dict | None = None) -> "UpsonicResult":
        """Run the agent synchronously (for /ask endpoint). Returns result object.

        For non-trivial requests, uses a two-phase plan-then-execute approach:
        Phase 1 generates a structured outline; Phase 2 executes it fully.
        """
        topic = (inputs or {}).get("topic", "")

        if _needs_planning(topic):
            # Phase 1: Plan
            plan_task = UpsonicTask(
                description=_PLAN_PROMPT.format(topic=topic),
                tools=[],
            )
            self._agent.do(plan_task, return_output=True)

            # Phase 2: Execute with plan as context
            exec_task = UpsonicTask(
                description=topic + _EXPECTED_OUTPUT,
                tools=_get_tools(),
                context=[plan_task],
            )
            run_output = self._agent.do(exec_task, return_output=True)
            return UpsonicResult(run_output, exec_task)
        else:
            # Simple request — single task
            task = UpsonicTask(
                description=topic + _EXPECTED_OUTPUT,
                tools=_get_tools(),
            )
            run_output = self._agent.do(task, return_output=True)
            return UpsonicResult(run_output, task)

    def stream_kickoff(self, inputs: dict | None = None):
        """Run the agent with event streaming (for /chat endpoint).
        Yields (event_kind, data_dict) tuples for real-time SSE.

        For non-trivial requests, uses a two-phase plan-then-execute approach:
        Phase 1 streams a structured outline as reasoning; Phase 2 streams
        the full execution with tool calls and text output.
        """
        topic = (inputs or {}).get("topic", "")

        if _needs_planning(topic):
            # Phase 1: Planning — stream events as reasoning
            plan_task = UpsonicTask(
                description=_PLAN_PROMPT.format(topic=topic),
                tools=[],
            )
            yield ('step', 'Phase 1: Analyzing request and planning response…')
            plan_output = None
            for event in self._agent.stream(plan_task, events=True):
                kind = getattr(event, 'event_kind', '')
                if kind == 'thinking_delta':
                    yield ('thinking', event.content)
                elif kind == 'text_delta':
                    # Plan text appears as reasoning, not final answer
                    yield ('thinking', event.content)
                elif kind == 'final_output':
                    plan_output = event.output
                elif kind == 'model_request_start':
                    yield ('thinking', 'Planning…')

            if plan_output:
                yield ('thinking', f'\n--- Plan ---\n{plan_output}\n---\n')

            # Phase 2: Execution — stream normally with plan as context
            exec_task = UpsonicTask(
                description=topic + _EXPECTED_OUTPUT,
                tools=_get_tools(),
                context=[plan_task],
            )
            yield ('step', 'Phase 2: Executing plan…')
            final_output = None
            for event in self._agent.stream(exec_task, events=True):
                yield from self._handle_stream_event(event)
                if getattr(event, 'event_kind', '') == 'final_output':
                    final_output = event.output
            yield ('done', UpsonicResult(final_output, exec_task, from_stream=True))
        else:
            # Simple request — single streaming task
            task = UpsonicTask(
                description=topic + _EXPECTED_OUTPUT,
                tools=_get_tools(),
            )
            final_output = None
            for event in self._agent.stream(task, events=True):
                yield from self._handle_stream_event(event)
                if getattr(event, 'event_kind', '') == 'final_output':
                    final_output = event.output
            yield ('done', UpsonicResult(final_output, task, from_stream=True))

    @staticmethod
    def _handle_stream_event(event):
        """Convert a single Upsonic stream event to (kind, data) tuples."""
        kind = getattr(event, 'event_kind', '')
        if kind == 'thinking_delta':
            yield ('thinking', event.content)
        elif kind == 'text_delta':
            yield ('text_delta', event.content)
        elif kind == 'tool_call':
            args_preview = ''
            if hasattr(event, 'tool_args') and event.tool_args:
                for key in ('search_query', 'query', 'prompt', 'instructions'):
                    if key in event.tool_args:
                        args_preview = f': {event.tool_args[key][:100]}'
                        break
            yield ('tool_call', f'{event.tool_name}{args_preview}')
        elif kind == 'tool_result':
            status = 'error' if event.is_error else 'done'
            preview = ''
            if event.result_preview:
                preview = f': {event.result_preview[:150]}'
            elif event.error_message:
                preview = f': {event.error_message[:150]}'
            time_str = f' ({event.execution_time:.1f}s)' if event.execution_time else ''
            yield ('tool_result', f'{event.tool_name} [{status}]{time_str}{preview}')
        elif kind == 'step_start':
            step_name = getattr(event, 'step_name', '') or ''
            step_idx = getattr(event, 'step_index', '')
            yield ('step', f'Step {step_idx}: {step_name}' if step_name else f'Step {step_idx}')
        elif kind == 'model_request_start':
            yield ('thinking', 'Calling LLM…')

    def continue_kickoff(self, original_query: str, partial_response: str,
                         file_context: str = "") -> "UpsonicResult":
        """Continue an incomplete response using the full Upsonic agent pipeline.

        Unlike the old litellm-direct approach, this goes through the agent's
        reflection loop and has access to search/image tools if the continuation
        requires them.
        """
        description = file_context + _CONTINUE_PROMPT.format(
            original_query=original_query,
            partial_response=partial_response,
        )
        task = UpsonicTask(
            description=description,
            tools=_get_tools(),
        )
        run_output = self._agent.do(task, return_output=True)
        return UpsonicResult(run_output, task)

    def summarize_history(self, conversation_text: str) -> str:
        """Produce a structured progress report from conversation history.

        This is the Upsonic equivalent of CrewAI's Process Manager — it
        condenses long conversations into a structured summary so the agent
        never loses track of what was discussed and what remains to be done.
        """
        task = UpsonicTask(
            description=_SUMMARIZE_PROMPT.format(conversation=conversation_text),
            tools=[],  # No tools needed for summarization
        )
        result = self._agent.do(task, return_output=True)
        return str(result) if result else ""


class UpsonicResult:
    """Thin wrapper to give Upsonic output the same interface main.py expects."""

    def __init__(self, run_output, task: UpsonicTask, from_stream: bool = False):
        self._run_output = run_output
        self._task = task
        self._from_stream = from_stream

    def __str__(self) -> str:
        if self._run_output is None:
            return ""
        if self._from_stream:
            # stream final_output is usually the text directly
            return str(self._run_output)
        if hasattr(self._run_output, 'output'):
            return str(self._run_output.output or "")
        return str(self._run_output)

    @property
    def reasoning_lines(self) -> list[str]:
        """Extract reasoning from AgentRunOutput thinking parts."""
        lines = []
        ro = self._run_output
        if ro is None:
            return lines
        # thinking_content is a single string
        if hasattr(ro, 'thinking_content') and ro.thinking_content:
            for line in ro.thinking_content.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
        # thinking_parts is a list of ThinkingPart objects
        if not lines and hasattr(ro, 'thinking_parts') and ro.thinking_parts:
            for part in ro.thinking_parts:
                content = getattr(part, 'content', '')
                if content:
                    for line in content.splitlines():
                        line = line.strip()
                        if line:
                            lines.append(line)
        # Also check tool calls for reasoning about tool usage
        if hasattr(ro, 'tools') and ro.tools:
            for tool_exec in ro.tools:
                name = getattr(tool_exec, 'tool_name', '') or getattr(tool_exec, 'name', '')
                if name:
                    lines.append(f"[Tool call: {name}]")
        return lines

    @property
    def token_usage(self):
        """Return a token-usage-like object if available."""
        if hasattr(self._task, "total_input_token") and self._task.total_input_token:
            return _TokenUsage(
                prompt_tokens=self._task.total_input_token or 0,
                completion_tokens=self._task.total_output_token or 0,
                total_tokens=(self._task.total_input_token or 0)
                + (self._task.total_output_token or 0),
            )
        return None


class _TokenUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
