import os
import gc
import json as _json
import uuid as _uuid
import logging
import threading
from pathlib import Path
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from crewai.tools import tool
from crewai_tools import SerperDevTool

from langchain_community.tools import DuckDuckGoSearchRun
from PIL import Image, ImageDraw, ImageFont

_logger = logging.getLogger(__name__)

# --- Smart TrueType font discovery ---
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Debian / Ubuntu / openSUSE
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",  # Fedora / RHEL
    "/usr/share/fonts/truetype/DejaVuSans.ttf",           # openSUSE alt
    "/usr/share/fonts/TTF/DejaVuSans.ttf",                # Arch
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",             # Generic Linux
    "/System/Library/Fonts/Helvetica.ttc",                 # macOS
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
_serper_tool = SerperDevTool()

_GENERATED_DIR = Path(__file__).parent / "static" / "generated"
_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

@tool("DuckDuckGoSearch")
def ddg_search_wrapped(query: str):
    """Search the web using DuckDuckGo. Use as fallback if Serper fails."""
    return _ddg_search.run(query)

@tool("GenerateImage")
def generate_image(instructions: str) -> str:
    """Draw geometric shapes. Input: JSON string with width, height, background, shapes array.
    Shape types: rectangle, circle, triangle, polygon, line, text. Copy the returned image tag into Final Answer."""
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
_sd_lock = threading.Lock()      # guards lazy pipeline init
_vram_lock = threading.Lock()    # serialises GPU-heavy inference
_SD_MODEL_ID = os.getenv("SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")

def _get_sd_pipe():
    """Lazy-load SD pipeline. Uses CPU offload so VRAM is only used during generation."""
    global _sd_pipe
    if _sd_pipe is None:
        with _sd_lock:
            if _sd_pipe is None:  # double-check
                import torch
                from diffusers import StableDiffusionPipeline
                print("[SD] Loading Stable Diffusion pipeline…", flush=True)
                _sd_pipe = StableDiffusionPipeline.from_pretrained(
                    _SD_MODEL_ID,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
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
class ResearchCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, model: str | None = None):
        self._model_name = model or os.getenv("MODEL", "ollama/qwen3.5:9b")
        self.ollama_llm = self._make_llm(self._model_name)

    @staticmethod
    def _make_llm(model: str) -> LLM:
        return LLM(
            model=model,
            base_url="http://localhost:11434",
            config={
                "options": {
                    "num_ctx": 4096,
                    "temperature": 0.1,
                    "num_gpu": 99,
                    "num_predict": 1024,
                }
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[_serper_tool, ddg_search_wrapped, generate_image, generate_ai_image],
            llm=self.ollama_llm,
            verbose=True,
            max_iter=8,
            memory=False,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )