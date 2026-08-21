"""Text-to-video generation (LTX-Video, Wan).

Deliberately mirrors ``image.py``: a runtime-switchable backend, lazy pipeline
loading, and the same ``_gpu_inference_ctx`` VRAM choreography — video needs
that discipline more than images do, since a video pipeline plus Ollama will not
fit in 16 GB together.

Model IDs are configurable because the good open video models move fast; the
defaults are the ones that fit a 16 GB card.
"""

import os
import gc
import re
import uuid as _uuid
import logging
from pathlib import Path

from crewai.tools import tool

# Reuse the image module's VRAM context manager rather than duplicating it: it
# already unloads Ollama, waits for VRAM, and works around the cuDNN conflict.
from researcher.image import (
    _gpu_inference_ctx,
    _GENERATED_DIR,
    _hf_model_cache_present,
)

_logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

VIDEO_BACKEND = os.getenv("VIDEO_BACKEND", "ltx").lower()
_VALID_VIDEO_BACKENDS = {"ltx", "wan", "none"}

# LTX-Video 2B: fast, but visibly weak on anatomy under motion. Kept as a
# lightweight fallback rather than the default.
_LTX_MODEL = os.getenv("LTX_MODEL", "Lightricks/LTX-Video")
# Wan 2.2 TI2V-5B is the quality default. ~4.66B params against LTX 2B, stored
# fp32 so it halves to ~9.3 GiB in bf16 — small enough that the transformer sits
# entirely in VRAM and runs without per-step PCIe streaming.
#
# Bigger Wan variants were evaluated and rejected for this machine: the A14B
# mixture-of-experts carries TWO 26.6 GiB transformers, ~58.5 GiB of weights,
# which does not fit in 64 GB of system RAM once CUDA and the app are resident.
_WAN_MODEL = os.getenv("WAN_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

# Each model has its own trained resolution, rate and step count. Using one
# model's numbers on the other is what made the first LTX clips limp, so the
# defaults are per-backend and only overridden when the env var is set explicitly.
_BACKEND_DEFAULTS = {
    # LTX conditions on frame_rate and was trained at 25fps; VAE compresses 32x
    # spatially, so dimensions want to be multiples of 32.
    "ltx": {"width": 704, "height": 512, "fps": 25, "seconds": 2.5,
            "steps": 50, "guidance": 3.0},
    # Wan 2.2 VAE compresses 16x spatially and 4x temporally. The pipeline's own
    # defaults are 832x480 / 81 frames / 50 steps / guidance 5.0, and it takes no
    # frame_rate argument — motion is implied by the training distribution.
    "wan": {"width": 832, "height": 480, "fps": 24, "seconds": 3.0,
            "steps": 50, "guidance": 5.0},
}


def _backend_default(key: str, backend: str | None = None):
    return _BACKEND_DEFAULTS.get(backend or VIDEO_BACKEND, _BACKEND_DEFAULTS["wan"])[key]


# An env var that is set explicitly always wins; anything unset follows the
# active backend. _PINNED records which the user pinned, so switching backend
# re-derives the rest instead of carrying one model's numbers onto another.
_PINNED = {k for k in ("width", "height", "fps", "seconds", "steps", "guidance")
           if os.getenv(f"VIDEO_{k.upper()}")}

_VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", _backend_default("width")))
_VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", _backend_default("height")))
# For LTX this is conditioning, not just playback speed: it tells the model how
# much movement belongs between frames, and is passed to the pipeline as well as
# used for export. Wan takes no frame_rate argument and infers motion from its
# training distribution, so there it only sets the export rate.
_VIDEO_FPS = int(os.getenv("VIDEO_FPS", _backend_default("fps")))
# Memory scales with the FRAME COUNT — frames are denoised jointly — so raising
# fps shortens the clip that fits, rather than costing nothing.
_VIDEO_SECONDS = float(os.getenv("VIDEO_SECONDS", _backend_default("seconds")))
_VIDEO_STEPS = int(os.getenv("VIDEO_STEPS", _backend_default("steps")))
_VIDEO_GUIDANCE = float(os.getenv("VIDEO_GUIDANCE", _backend_default("guidance")))

# Most video pipelines require frame counts of the form 8k+1.
def _frame_count(seconds: float, fps: int) -> int:
    raw = max(9, int(seconds * fps))
    return ((raw - 1) // 8) * 8 + 1


_video_pipe = None
_video_pipe_kind: str | None = None


def get_video_params() -> dict:
    """Current runtime video parameters."""
    return {
        "video_backend": VIDEO_BACKEND,
        "width": _VIDEO_WIDTH,
        "height": _VIDEO_HEIGHT,
        "fps": _VIDEO_FPS,
        "seconds": _VIDEO_SECONDS,
        "steps": _VIDEO_STEPS,
        "guidance": _VIDEO_GUIDANCE,
        "model": _LTX_MODEL if VIDEO_BACKEND == "ltx" else _WAN_MODEL,
    }


def update_video_params(params: dict) -> dict:
    """Update runtime video parameters. Only known keys are accepted."""
    global VIDEO_BACKEND, _VIDEO_WIDTH, _VIDEO_HEIGHT, _VIDEO_FPS
    global _VIDEO_SECONDS, _VIDEO_STEPS, _VIDEO_GUIDANCE, _video_pipe, _video_pipe_kind

    if "video_backend" in params:
        val = str(params["video_backend"]).lower()
        if val in _VALID_VIDEO_BACKENDS:
            if val != VIDEO_BACKEND:
                # Switching backend invalidates the cached pipeline; dropping the
                # reference matters on a 16 GB card.
                _video_pipe = None
                _video_pipe_kind = None
                gc.collect()
                VIDEO_BACKEND = val
                # Carry over only what the user pinned; everything else moves to
                # the new model's native settings.
                if val in _BACKEND_DEFAULTS:
                    if "width" not in _PINNED:
                        _VIDEO_WIDTH = _backend_default("width", val)
                    if "height" not in _PINNED:
                        _VIDEO_HEIGHT = _backend_default("height", val)
                    if "fps" not in _PINNED:
                        _VIDEO_FPS = _backend_default("fps", val)
                    if "seconds" not in _PINNED:
                        _VIDEO_SECONDS = _backend_default("seconds", val)
                    if "steps" not in _PINNED:
                        _VIDEO_STEPS = _backend_default("steps", val)
                    if "guidance" not in _PINNED:
                        _VIDEO_GUIDANCE = _backend_default("guidance", val)
            VIDEO_BACKEND = val
    if "width" in params:
        _VIDEO_WIDTH = max(256, int(params["width"]))
    if "height" in params:
        _VIDEO_HEIGHT = max(256, int(params["height"]))
    if "fps" in params:
        _VIDEO_FPS = max(4, min(30, int(params["fps"])))
    if "seconds" in params:
        _VIDEO_SECONDS = max(1.0, min(15.0, float(params["seconds"])))
    if "steps" in params:
        _VIDEO_STEPS = max(4, min(100, int(params["steps"])))
    if "guidance" in params:
        _VIDEO_GUIDANCE = max(0.0, min(15.0, float(params["guidance"])))
    return get_video_params()


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------


def _load_pipe(kind: str):
    """Load (and cache) the pipeline for *kind*."""
    global _video_pipe, _video_pipe_kind
    if _video_pipe is not None and _video_pipe_kind == kind:
        return _video_pipe

    import sys as _sys
    import torch

    if kind == "ltx":
        from diffusers import LTXPipeline as _Pipe
        model_id = _LTX_MODEL
    elif kind == "wan":
        from diffusers import WanPipeline as _Pipe
        model_id = _WAN_MODEL
    else:
        raise ValueError(f"Unknown video backend: {kind}")

    _sys.stderr.write(f"[VIDEO] Loading {kind} pipeline ({model_id})…\n")
    _sys.stderr.flush()
    pipe = _Pipe.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # Offload strategy, in order of preference.
    #
    # These pipelines total ~26 GiB on disk, but the components run one at a
    # time: encode the prompt with T5, free it, then denoise, then decode. The
    # binding constraint is therefore the LARGEST single component (T5-XXL,
    # ~9.5 GiB in bf16), not the sum — and that fits in 15.5 GiB.
    #
    # So prefer model-level offload, which keeps whole modules resident and
    # swaps them once per phase. Sequential offload streams weights layer by
    # layer over PCIe on EVERY step; it needs less VRAM but is far slower, so it
    # is the fallback for when model offload runs out of memory.
    try:
        pipe.enable_model_cpu_offload()
        _sys.stderr.write("[VIDEO] Model CPU offload enabled\n")
    except Exception as e:
        _sys.stderr.write(f"[VIDEO] Model offload unavailable ({e}); using sequential\n")
        try:
            pipe.enable_sequential_cpu_offload()
            _sys.stderr.write("[VIDEO] Sequential CPU offload enabled (slower)\n")
        except Exception:
            pipe.to("cuda")
    # VAE tiling is what keeps the DECODE from OOMing: without it the VAE
    # reconstructs every frame at full resolution in one allocation, which blows
    # 15.5 GiB even when the denoise itself fit comfortably.
    #
    # These helpers are NOT on WanPipeline — they live on the VAE object. The
    # previous loop called them on the pipeline and swallowed the AttributeError,
    # so tiling was silently never enabled. Try both, and say which worked.
    for name, target in (("vae", getattr(pipe, "vae", None)), ("pipe", pipe)):
        if target is None:
            continue
        for meth in ("enable_tiling", "enable_slicing",
                     "enable_vae_tiling", "enable_vae_slicing"):
            fn = getattr(target, meth, None)
            if callable(fn):
                try:
                    fn()
                    _sys.stderr.write(f"[VIDEO] {name}.{meth}() enabled\n")
                except Exception as e:
                    _sys.stderr.write(f"[VIDEO] {name}.{meth}() failed: {e}\n")
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        _sys.stderr.write(
            f"[VIDEO] vae tiling={getattr(vae, 'use_tiling', '?')} "
            f"slicing={getattr(vae, 'use_slicing', '?')}\n"
        )

    _video_pipe, _video_pipe_kind = pipe, kind
    _sys.stderr.write("[VIDEO] Pipeline ready.\n")
    _sys.stderr.flush()
    return pipe


# Off by default. Unlike the image pipelines, warming this costs ~16 GiB of
# SYSTEM RAM (model CPU offload keeps weights in RAM and moves modules to the GPU
# only during inference, so it is not VRAM that is held). That is a real slice of
# 64 GB to reserve for a one-time saving on the first request, so it is opt-in.
VIDEO_WARMUP = os.getenv("VIDEO_WARMUP", "0") == "1"


def preload_video():
    """Warm the video pipeline in the background (mirrors preload_zimage)."""
    import threading

    if not VIDEO_WARMUP or VIDEO_BACKEND not in ("ltx", "wan"):
        return
    if not _hf_model_cache_present(_LTX_MODEL if VIDEO_BACKEND == "ltx" else _WAN_MODEL):
        _logger.info("[VIDEO] Warm-up skipped: weights not downloaded yet")
        return

    def _work():
        try:
            _load_pipe(VIDEO_BACKEND)
        except Exception as e:
            _logger.warning("[VIDEO] Preload failed: %s", e)

    threading.Thread(target=_work, daemon=True).start()


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _export(frames, fps: int) -> str:
    """Write frames to an MP4 the browser can actually play."""
    from diffusers.utils import export_to_video

    fname = f"{_uuid.uuid4().hex[:12]}.mp4"
    out = _GENERATED_DIR / fname
    export_to_video(frames, str(out), fps=fps)
    return fname


def _generate_video_once(prompt: str) -> str:
    """Render one clip and return an HTML <video> tag, or an error string."""
    import sys as _sys
    import torch

    backend = VIDEO_BACKEND.lower()
    if backend == "none":
        return "Error: video generation is disabled (VIDEO_BACKEND=none)."
    if backend not in _VALID_VIDEO_BACKENDS:
        return f"Error: Unknown VIDEO_BACKEND '{backend}'. Use 'ltx', 'wan' or 'none'."

    # Fail fast rather than kicking off a multi-gigabyte download inside a chat
    # request, where it would look like a hang and then time out.
    model_id = _LTX_MODEL if backend == "ltx" else _WAN_MODEL
    if not _hf_model_cache_present(model_id):
        return (
            f"Error: the video model '{model_id}' is not downloaded yet.\n"
            f"Fetch ONLY the diffusers components (~27 GiB). Do not download the "
            f"whole repo: it also hosts standalone checkpoints for other tools "
            f"(LTX-Video is 236 GiB in full).\n"
            f"    hf download {model_id} \\\n"
            f"        --include 'text_encoder/*' 'transformer/*' 'vae/*' "
            f"'tokenizer/*' 'scheduler/*' '*.json'\n"
            f"Or set VIDEO_BACKEND=none to disable video generation."
        )

    num_frames = _frame_count(_VIDEO_SECONDS, _VIDEO_FPS)
    _sys.stderr.write(
        f"[VIDEO] {backend}: {_VIDEO_WIDTH}x{_VIDEO_HEIGHT}, {num_frames} frames "
        f"@ {_VIDEO_FPS}fps, {_VIDEO_STEPS} steps — {prompt[:80]}\n"
    )
    _sys.stderr.flush()

    with _gpu_inference_ctx("VIDEO", efficient_attention=True):
        try:
            pipe = _load_pipe(backend)
            # frame_rate is conditioning, not metadata: the model uses it to decide
            # how much movement belongs between consecutive frames. Omitting it
            # leaves the pipeline default (25) while we export at something else,
            # which is what made the first clips look like they were limping.
            extra = {}
            import inspect as _inspect
            if "frame_rate" in _inspect.signature(pipe.__call__).parameters:
                extra["frame_rate"] = _VIDEO_FPS

            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    **extra,
                    negative_prompt=os.getenv(
                        "VIDEO_NEGATIVE_PROMPT",
                        "blurry, distorted, low quality, watermark, text",
                    ),
                    width=_VIDEO_WIDTH,
                    height=_VIDEO_HEIGHT,
                    num_frames=num_frames,
                    num_inference_steps=_VIDEO_STEPS,
                    guidance_scale=_VIDEO_GUIDANCE,
                )
            frames = result.frames[0]
            fname = _export(frames, _VIDEO_FPS)
        except torch.cuda.OutOfMemoryError:
            # Rather than dead-ending, walk down a cascade: fewer frames first
            # (the dominant term), then smaller frames. Mirrors the OOM retry the
            # ZImage path already does.
            # An OOM mid-forward leaves activations and offload hooks holding
            # VRAM: the first retry saw only 6.9 MB free and could not allocate
            # even 33 MB. Dropping the cached pipeline and collecting twice
            # actually returns the memory, at the cost of reloading weights.
            global _video_pipe, _video_pipe_kind
            _video_pipe = None
            _video_pipe_kind = None
            del pipe
            for _ in range(2):
                gc.collect()
                torch.cuda.empty_cache()
            _sys.stderr.write(
                f"[VIDEO] Released pipeline after OOM; "
                f"{torch.cuda.mem_get_info()[0]/2**30:.2f} GiB free\n"
            )
            pipe = _load_pipe(backend)

            for scale_frames, scale_dim in ((0.6, 1.0), (0.6, 0.75), (0.4, 0.75)):
                torch.cuda.empty_cache()
                gc.collect()
                nf = max(9, int(num_frames * scale_frames))
                nf = ((nf - 1) // 8) * 8 + 1
                w = (int(_VIDEO_WIDTH * scale_dim) // 32) * 32
                h = (int(_VIDEO_HEIGHT * scale_dim) // 32) * 32
                _sys.stderr.write(
                    f"[VIDEO] OOM — retrying at {w}x{h}, {nf} frames\n"
                )
                _sys.stderr.flush()
                try:
                    with torch.inference_mode():
                        result = pipe(
                            prompt=prompt, **extra,
                            negative_prompt=os.getenv(
                                "VIDEO_NEGATIVE_PROMPT",
                                "blurry, distorted, low quality, watermark, text",
                            ),
                            width=w, height=h, num_frames=nf,
                            num_inference_steps=_VIDEO_STEPS,
                            guidance_scale=_VIDEO_GUIDANCE,
                        )
                    fname = _export(result.frames[0], _VIDEO_FPS)
                    _sys.stderr.write(f"[VIDEO] Recovered at {w}x{h}/{nf}f\n")
                    return (
                        f'<video src="/static/generated/{fname}" controls loop muted '
                        f'playsinline style="max-width:100%;border-radius:0.5rem;"></video>'
                    )
                except torch.cuda.OutOfMemoryError:
                    continue
                except Exception as e:
                    return f"Error generating video: {e}"
            return (
                "Error: ran out of VRAM generating video even after reducing size. "
                "Lower VIDEO_SECONDS or VIDEO_WIDTH/VIDEO_HEIGHT, or switch "
                "VIDEO_BACKEND to 'wan' (1.3B), which needs less memory."
            )
        except Exception as e:
            _sys.stderr.write(f"[VIDEO] ERROR: {e}\n")
            return f"Error generating video: {e}"
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _sys.stderr.write(f"[VIDEO] Done: {fname}\n")
    _sys.stderr.flush()
    return (
        f'<video src="/static/generated/{fname}" controls loop muted playsinline '
        f'style="max-width:100%;border-radius:0.5rem;"></video>'
    )


@tool("GenerateAIVideo")
def generate_ai_video(prompt: str) -> str:
    """Generate a short video clip from a text prompt. Use for motion, b-roll,
    animated scenes. Describe the subject, the action, and the camera movement.
    Copy the returned video tag EXACTLY into your Final Answer."""
    return _generate_video_once(prompt)


# The video tag IS the final answer — same reasoning as GenerateAIImage: Ollama
# was just unloaded and reloaded, so a post-tool LLM call returns empty.
generate_ai_video.result_as_answer = True


_VIDEO_PATH_RE = re.compile(r'/static/generated/([A-Za-z0-9_.-]+\.mp4)')


def video_tag_to_path(tag: str) -> Path | None:
    """Extract the on-disk path from a returned video tag."""
    m = _VIDEO_PATH_RE.search(tag or "")
    if not m:
        return None
    p = _GENERATED_DIR / m.group(1)
    return p if p.is_file() else None
