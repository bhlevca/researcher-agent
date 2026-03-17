import os
import json as _json
import uuid as _uuid
import threading
from pathlib import Path
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from crewai.tools import tool
from crewai_tools import SerperDevTool

from langchain_community.tools import DuckDuckGoSearchRun
from PIL import Image, ImageDraw, ImageFont

# Instantiate once, reuse across calls
_ddg_search = DuckDuckGoSearchRun()
_serper_tool = SerperDevTool()

_GENERATED_DIR = Path(__file__).parent / "static" / "generated"
_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

@tool("DuckDuckGoSearch")
def ddg_search_wrapped(query: str):
    """Search the web using DuckDuckGo for general information and fallback data."""
    return _ddg_search.run(query)

@tool("GenerateImage")
def generate_image(instructions: str) -> str:
    """Generate an image with shapes and text. The instructions parameter must be a JSON string with:
    - width, height: image dimensions in pixels (default 512)
    - background: color string like "white", "#3498db", "lightblue" (default "white")
    - shapes: list of shape objects. Each has a "type" and properties:
      Rectangle: {"type":"rectangle","x":10,"y":10,"width":200,"height":100,"fill":"blue","outline":"black"}
      Circle: {"type":"circle","cx":256,"cy":256,"radius":100,"fill":"red","outline":"black"}
      Triangle: {"type":"triangle","points":[[256,56],[56,456],[456,456]],"fill":"green","outline":"black"}
      Polygon: {"type":"polygon","points":[[x1,y1],[x2,y2],[x3,y3],...],"fill":"yellow","outline":"black"}
      Line: {"type":"line","x1":0,"y1":0,"x2":200,"y2":200,"fill":"black","width":3}
      Text: {"type":"text","x":10,"y":10,"text":"Hello","fill":"black","size":24}
    Triangle inside a circle example: {"width":512,"height":512,"background":"white","shapes":[{"type":"circle","cx":256,"cy":256,"radius":200,"fill":"lightblue","outline":"black"},{"type":"triangle","points":[[256,80],[100,400],[412,400]],"fill":"red","outline":"black"}]}
    IMPORTANT: Copy the tool's return value exactly into your Final Answer.
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
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except (IOError, OSError):
                font = ImageFont.load_default()
            draw.text((x, y), txt, fill=fill or "black", font=font)
    fname = f"{_uuid.uuid4().hex[:12]}.png"
    img.save(_GENERATED_DIR / fname)
    return f"![generated image](/static/generated/{fname})"

# --------------- Stable Diffusion AI image generation ---------------
_sd_pipe = None
_sd_lock = threading.Lock()
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
    """Generate a realistic or artistic image from a text description using Stable Diffusion AI.
    Use this tool when the user asks for complex images like animals, trees, landscapes, people,
    logos, or anything that cannot be drawn with simple geometric shapes.
    The prompt parameter should be a detailed text description of the desired image.
    Good prompts: "a red fox sitting in a snowy forest, digital art, highly detailed"
    Bad prompts: "fox" (too vague)
    Add style keywords for better results: "photorealistic", "digital art", "oil painting",
    "watercolor", "cartoon", "3D render", "highly detailed", "8k".
    The tool returns a markdown image tag. Copy it EXACTLY into your Final Answer.
    """
    try:
        pipe = _get_sd_pipe()
        result = pipe(
            prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=512,
            height=512,
        )
        img = result.images[0]
        fname = f"{_uuid.uuid4().hex[:12]}.png"
        img.save(_GENERATED_DIR / fname)
        return f"![generated image](/static/generated/{fname})"
    except Exception as e:
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
            max_iter=5,
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