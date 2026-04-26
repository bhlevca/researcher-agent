"""AI image generation pipelines (Stable Diffusion, ZImage-Turbo).

Handles lazy pipeline loading, VRAM management, Ollama unload/reload,
OOM retry cascade, and high-resolution tiled generation with overlap
blending.  Extracted from crew.py during v1.1.0 refactoring.
"""

import math
import os
import gc
import json
import threading
import uuid as _uuid
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from crewai.tools import tool

_logger = logging.getLogger(__name__)

_GENERATED_DIR = Path(__file__).parent / "static" / "generated"
_GENERATED_DIR.mkdir(parents=True, exist_ok=True)


# --------------- Tiled image generation / stitching ---------------

def _compute_tile_grid(
    target_w: int,
    target_h: int,
    tile_w: int,
    tile_h: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Return top-left (x, y) positions for each tile in the grid.

    Tiles are spaced so that consecutive tiles overlap by *overlap* pixels.
    The last column/row is nudged left/up so the grid exactly covers the
    target dimensions without exceeding them.

    Returns a list of (x, y) tuples.
    """
    stride_x = tile_w - overlap
    stride_y = tile_h - overlap

    # Number of tiles needed in each axis
    cols = max(1, math.ceil((target_w - overlap) / stride_x))
    rows = max(1, math.ceil((target_h - overlap) / stride_y))

    positions: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            x = col * stride_x
            y = row * stride_y
            # Clamp last tile so it doesn't exceed target
            x = min(x, target_w - tile_w)
            y = min(y, target_h - tile_h)
            positions.append((x, y))
    return positions


def _make_blend_mask(
    tile_w: int,
    tile_h: int,
    overlap: int,
    feather_left: bool = True,
    feather_right: bool = True,
    feather_top: bool = True,
    feather_bottom: bool = True,
) -> np.ndarray:
    """Create a 2-D cosine feathering mask for a single tile.

    The mask is 1.0 in the interior and ramps from 0→1 over *overlap*
    pixels along edges marked for feathering.  Border tiles should set
    the outer-edge flags to False so the boundary stays at full weight.
    """
    mask = np.ones((tile_h, tile_w), dtype=np.float32)
    if overlap <= 0:
        return mask

    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap))  # 0→1

    # Left / right edges
    if feather_left:
        mask[:, :overlap] *= ramp[np.newaxis, :]
    if feather_right:
        mask[:, -overlap:] *= ramp[np.newaxis, ::-1]
    # Top / bottom edges
    if feather_top:
        mask[:overlap, :] *= ramp[:, np.newaxis]
    if feather_bottom:
        mask[-overlap:, :] *= ramp[::-1][:, np.newaxis]

    return mask


def stitch_tiles(
    tiles: list[Image.Image],
    positions: list[tuple[int, int]],
    target_w: int,
    target_h: int,
    overlap: int,
) -> Image.Image:
    """Blend and composite *tiles* into a single (target_w × target_h) image.

    Each tile is placed at its (x, y) position from *positions*.
    Overlapping regions are blended using cosine alpha feathering so that
    seams are invisible.  Border tiles only feather interior edges.
    """
    if len(tiles) != len(positions):
        raise ValueError("tiles and positions must have the same length")

    # Accumulator arrays (float64 for precision)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.float64)
    weight = np.zeros((target_h, target_w), dtype=np.float64)

    for tile_img, (x, y) in zip(tiles, positions):
        tile_arr = np.asarray(tile_img.convert("RGB"), dtype=np.float64)
        th, tw = tile_arr.shape[:2]
        # Only feather edges that overlap with adjacent tiles;
        # border edges (touching the canvas boundary) stay at full weight.
        mask = _make_blend_mask(
            tw, th, overlap,
            feather_left=(x > 0),
            feather_right=(x + tw < target_w),
            feather_top=(y > 0),
            feather_bottom=(y + th < target_h),
        ).astype(np.float64)

        canvas[y : y + th, x : x + tw] += tile_arr * mask[:, :, np.newaxis]
        weight[y : y + th, x : x + tw] += mask

    # Normalise — avoid division by zero
    weight = np.maximum(weight, 1e-8)
    canvas /= weight[:, :, np.newaxis]

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def _tile_position_label(
    x: int, y: int, cols: int, rows: int, stride_x: int, stride_y: int
) -> str:
    """Return a short positional hint like 'top-left', 'center', etc."""
    col = round(x / stride_x) if stride_x else 0
    row = round(y / stride_y) if stride_y else 0

    v = "top" if row == 0 else ("bottom" if row >= rows - 1 else "middle")
    h = "left" if col == 0 else ("right" if col >= cols - 1 else "center")

    if rows == 1 and cols == 1:
        return ""
    if rows == 1:
        return h
    if cols == 1:
        return v
    return f"{v}-{h}"

# --------------- Stable Diffusion AI image generation ---------------
_sd_pipe = None
_sd_lock = threading.Lock()  # guards lazy pipeline init
_vram_lock = threading.Lock()  # serialises GPU-heavy inference

# Image backend selection
IMAGE_BACKEND = os.getenv("IMAGE_BACKEND", "sd")  # "sd" or "zimage"

# ZImage inference settings (turbo / FLUX-distilled models need few steps)
_ZIMAGE_STEPS = int(os.getenv("ZIMAGE_STEPS", "4"))
_ZIMAGE_GUIDANCE = float(os.getenv("ZIMAGE_GUIDANCE", "0.0"))
_ZIMAGE_MAX_SEQ = int(os.getenv("ZIMAGE_MAX_SEQ", "512"))

# High-resolution tiled generation (v1.5.0)
# Defaults from env vars, but overridable at runtime via get/set_image_params().
ZIMAGE_HIRES = os.getenv("ZIMAGE_HIRES", "0") == "1"
_ZIMAGE_HIRES_WIDTH = int(os.getenv("ZIMAGE_HIRES_WIDTH", "2048"))
_ZIMAGE_HIRES_HEIGHT = int(os.getenv("ZIMAGE_HIRES_HEIGHT", "2048"))
_ZIMAGE_TILE_OVERLAP = int(os.getenv("ZIMAGE_TILE_OVERLAP", "192"))
_ZIMAGE_HIRES_STRENGTH = float(os.getenv("ZIMAGE_HIRES_STRENGTH", "0.35"))

# Default image params dict — mirrors the module vars above.
# Keys here are what the API / UI use; values are the mutable module state.
DEFAULT_IMAGE_PARAMS: dict = {
    "hires_enabled": False,
    "hires_width": 2048,
    "hires_height": 2048,
    "tile_overlap": 192,
    "hires_strength": 0.35,
    "tile_width": int(os.getenv("ZIMAGE_WIDTH", "512")),
    "tile_height": int(os.getenv("ZIMAGE_HEIGHT", "512")),
}


def get_image_params() -> dict:
    """Return the current runtime image generation parameters."""
    return {
        "hires_enabled": ZIMAGE_HIRES,
        "hires_width": _ZIMAGE_HIRES_WIDTH,
        "hires_height": _ZIMAGE_HIRES_HEIGHT,
        "tile_overlap": _ZIMAGE_TILE_OVERLAP,
        "hires_strength": _ZIMAGE_HIRES_STRENGTH,
        "tile_width": int(os.getenv("ZIMAGE_WIDTH", "512")),
        "tile_height": int(os.getenv("ZIMAGE_HEIGHT", "512")),
    }


def update_image_params(params: dict) -> dict:
    """Update runtime image parameters. Only known keys are accepted."""
    global ZIMAGE_HIRES, _ZIMAGE_HIRES_WIDTH, _ZIMAGE_HIRES_HEIGHT, _ZIMAGE_TILE_OVERLAP, _ZIMAGE_HIRES_STRENGTH
    if "hires_enabled" in params:
        ZIMAGE_HIRES = bool(params["hires_enabled"])
    if "hires_width" in params:
        _ZIMAGE_HIRES_WIDTH = max(256, int(params["hires_width"]))
    if "hires_height" in params:
        _ZIMAGE_HIRES_HEIGHT = max(256, int(params["hires_height"]))
    if "tile_overlap" in params:
        _ZIMAGE_TILE_OVERLAP = max(0, min(256, int(params["tile_overlap"])))
    if "hires_strength" in params:
        _ZIMAGE_HIRES_STRENGTH = max(0.05, min(1.0, float(params["hires_strength"])))
    # tile_width / tile_height are read from ZIMAGE_WIDTH/HEIGHT env vars
    # and used as the native pipeline resolution — not changeable at runtime
    # because changing them would require re-validating VRAM constraints.
    return get_image_params()


def load_image_params_from_json(saved: str | None) -> dict:
    """Apply image params from a JSON string (e.g. DB column). Returns current params."""
    if saved:
        try:
            update_image_params(json.loads(saved))
        except (json.JSONDecodeError, TypeError):
            pass
    return get_image_params()


# OLLAMA_IMAGE_URL is used ONLY for unloading/reloading the LLM via keep_alive=0.
# It is NOT used for image generation when IMAGE_BACKEND=zimage.
OLLAMA_IMAGE_URL = os.getenv("OLLAMA_IMAGE_URL", "http://localhost:11434/api/generate")
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


# --------------- ZImagePipeline backend (diffusers + CUDA) ---------------
# Used when IMAGE_BACKEND=zimage.
# Reads:
#   ZIMAGE_MODEL          — HuggingFace repo id (default: mrfakename/Z-Image-Turbo)
#   HUGGINGFACE_TOKEN     — HF token for gated repos
#   TRANSFORMERS_OFFLINE  — set to "1" to skip network after first download
# Ollama is NOT involved for image generation; it is only used to
# unload/reload the qwen3 LLM around inference to free VRAM.
_ZIMAGE_MODEL_ID = os.getenv("ZIMAGE_MODEL", "mrfakename/Z-Image-Turbo").split("#")[0].strip()
_zimage_pipe = None
_zimage_init_lock = threading.Lock()

# Flag set by generate_ai_image when GPU image generation ran.
# Checked by main.py to decide whether to unload Ollama after the request.
_image_was_generated = False


def _get_zimage_pipe():
    """Lazy-load ZImagePipeline once with bfloat16 + Flash Attention."""
    global _zimage_pipe
    if _zimage_pipe is None:
        with _zimage_init_lock:
            if _zimage_pipe is None:
                import torch
                from diffusers import ZImagePipeline
                import sys as _sys

                hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
                offline = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"

                print(
                    f"[ZIMG] Loading ZImagePipeline ({_ZIMAGE_MODEL_ID})"
                    f"{' [offline]' if offline else ''}...",
                    flush=True,
                )
                pipe = ZImagePipeline.from_pretrained(
                    _ZIMAGE_MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    token=hf_token,
                    local_files_only=offline,
                )
                # Leverage diffusers memory optimizations for low VRAM.
                # Use sequential CPU offload (moves individual layers to GPU
                # one at a time) for minimal peak VRAM — essential when sharing
                # the GPU with Ollama.
                try:
                    pipe.enable_sequential_cpu_offload()
                    _sys.stderr.write("[ZIMG] Sequential CPU offload enabled\n")
                except Exception:
                    # Fallback to model-level offload
                    try:
                        pipe.enable_model_cpu_offload()
                        _sys.stderr.write("[ZIMG] Model CPU offload enabled (fallback)\n")
                    except Exception:
                        _sys.stderr.write("[ZIMG] CPU offload unavailable\n")

                try:
                    pipe.enable_attention_slicing()
                    _sys.stderr.write("[ZIMG] Attention slicing enabled\n")
                except Exception:
                    _sys.stderr.write("[ZIMG] Attention slicing unavailable\n")

                # VAE tiling + slicing: decode latents in tiles to slash VRAM
                try:
                    pipe.enable_vae_tiling()
                    _sys.stderr.write("[ZIMG] VAE tiling enabled\n")
                except Exception:
                    _sys.stderr.write("[ZIMG] VAE tiling unavailable\n")
                try:
                    pipe.enable_vae_slicing()
                    _sys.stderr.write("[ZIMG] VAE slicing enabled\n")
                except Exception:
                    _sys.stderr.write("[ZIMG] VAE slicing unavailable\n")

                # Flash Attention - try pipeline-level, then transformer-level
                try:
                    pipe.enable_flash_attention()
                    _sys.stderr.write("[ZIMG] Flash Attention enabled\n")
                except AttributeError:
                    try:
                        pipe.transformer.enable_flash_attn()
                        _sys.stderr.write(
                            "[ZIMG] Flash Attention enabled (transformer)\n"
                        )
                    except Exception:
                        _sys.stderr.write(
                            "[ZIMG] Flash Attention unavailable - using default\n"
                        )
                _zimage_pipe = pipe
                print("[ZIMG] Pipeline ready.", flush=True)
    return _zimage_pipe


def preload_zimage():
    """Warm up the ZImagePipeline in a background thread."""
    threading.Thread(target=_get_zimage_pipe, daemon=True).start()


@tool("GenerateAIImage")
def generate_ai_image(prompt: str) -> str:
    """Generate a realistic image from a text prompt using Stable Diffusion AI or Ollama z-image.
    Use for animals, landscapes, people, objects, scenes. Add style keywords like photorealistic, 8k.
    Copy the returned image tag EXACTLY into your Final Answer."""
    import sys as _sys
    import requests

    global _image_was_generated
    backend = IMAGE_BACKEND.lower()
    _sys.stderr.write(
        f"[IMG] generate_ai_image called (backend={backend}): {prompt[:100]}\n"
    )
    _sys.stderr.flush()
    _image_was_generated = True
    if backend == "sd":
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
    elif backend == "zimage":
        # ZImagePipeline via diffusers + CUDA.
        # Controlled by .env: ZIMAGE_MODEL, HUGGINGFACE_TOKEN, TRANSFORMERS_OFFLINE.
        # Ollama is used only to unload/reload the LLM around inference to free VRAM.
        import requests as _requests

        # Query Ollama for ALL currently loaded models and unload each one.
        # Using /api/ps avoids model-name mismatches when the user switched
        # models via the UI (the ENV var may be stale).
        _ollama_base = OLLAMA_IMAGE_URL.rsplit("/", 2)[0]  # http://host:port
        _unloaded_models = []
        try:
            _ps_resp = _requests.get(f"{_ollama_base}/api/ps", timeout=10)
            _ps_resp.raise_for_status()
            for _m in _ps_resp.json().get("models", []):
                _mname = _m.get("name", "")
                if _mname:
                    _requests.post(
                        OLLAMA_IMAGE_URL,
                        json={"model": _mname, "keep_alive": 0, "prompt": ""},
                        timeout=30,
                    )
                    _unloaded_models.append(_mname)
            _sys.stderr.write(f"[ZIMG] Unloaded LLM(s): {_unloaded_models}\n")
        except Exception as e:
            _sys.stderr.write(f"[ZIMG] Failed to unload LLM via /api/ps: {e}\n")
            # Fallback: try the ENV model name
            _fallback_model = os.getenv("MODEL", "qwen3.5:9b").replace("ollama/", "")
            try:
                _requests.post(
                    OLLAMA_IMAGE_URL,
                    json={"model": _fallback_model, "keep_alive": 0, "prompt": ""},
                    timeout=30,
                )
                _unloaded_models.append(_fallback_model)
                _sys.stderr.write(f"[ZIMG] Unloaded LLM (fallback): {_fallback_model}\n")
            except Exception as e2:
                _sys.stderr.write(f"[ZIMG] Fallback unload also failed: {e2}\n")

        # Wait for Ollama to actually release VRAM (async unload)
        import time as _time
        import torch

        _sys.stderr.write("[ZIMG] Waiting for VRAM to free...\n")
        for _wait_i in range(20):
            torch.cuda.empty_cache()
            _free = torch.cuda.mem_get_info()[0] / (1024**3)
            _sys.stderr.write(f"[ZIMG]   free VRAM: {_free:.2f} GiB\n")
            if _free > 4.0:  # sequential CPU offload needs much less peak VRAM
                break
            _time.sleep(1)
        else:
            _sys.stderr.write("[ZIMG] Warning: VRAM may still be tight\n")

        img_tag = None
        gen_method = "unknown"
        last_exc = None
        pipe = None
        try:
            pipe = _get_zimage_pipe()

            tile_w = int(os.getenv("ZIMAGE_WIDTH", "512"))
            tile_h = int(os.getenv("ZIMAGE_HEIGHT", "512"))

            # Decide whether to use tiled hi-res generation
            hires = ZIMAGE_HIRES
            target_w = _ZIMAGE_HIRES_WIDTH if hires else tile_w
            target_h = _ZIMAGE_HIRES_HEIGHT if hires else tile_h
            overlap = _ZIMAGE_TILE_OVERLAP if hires else 0
            _sys.stderr.write(
                f"[ZIMG] Config: hires={hires}, target={target_w}x{target_h}, "
                f"tile={tile_w}x{tile_h}, overlap={overlap}, "
                f"strength={_ZIMAGE_HIRES_STRENGTH}\n"
            )
            _sys.stderr.flush()

            if hires and (target_w > tile_w or target_h > tile_h):
                # ---- Hi-res: base → progressive 2x upscale+refine passes ----
                #
                # Instead of jumping 512→2048 (4x) in one pass — which causes
                # hallucination at high strength or blur at low strength — we
                # do progressive 2x passes:
                #   512→1024 (refine tiles) → 2048 (refine tiles)
                # Each pass only doubles, so moderate strength (0.4-0.5)
                # produces sharp detail without hallucination.
                _sys.stderr.write(
                    f"[ZIMG] Hi-res: generating base image "
                    f"({tile_w}x{tile_h})...\n"
                )
                _sys.stderr.flush()

                # Step 1: generate coherent base image
                base_img = None
                _base_sizes = [(tile_w, tile_h)]
                if (tile_w, tile_h) != (384, 384):
                    _base_sizes.append((384, 384))
                for _bw, _bh in _base_sizes:
                    try:
                        with torch.inference_mode():
                            result = pipe(
                                prompt=prompt,
                                width=_bw,
                                height=_bh,
                                num_inference_steps=_ZIMAGE_STEPS,
                                guidance_scale=_ZIMAGE_GUIDANCE,
                                max_sequence_length=_ZIMAGE_MAX_SEQ,
                            )
                        base_img = result.images[0]
                        _sys.stderr.write(
                            f"[ZIMG] Base image generated ({_bw}x{_bh})\n"
                        )
                        torch.cuda.empty_cache()
                        break
                    except RuntimeError as oom_exc:
                        if "out of memory" in str(oom_exc).lower():
                            _sys.stderr.write(
                                f"[ZIMG] OOM at {_bw}x{_bh} for base, "
                                f"trying smaller...\n"
                            )
                            torch.cuda.empty_cache()
                            gc.collect()
                            last_exc = oom_exc
                            continue
                        raise

                if base_img is None:
                    raise RuntimeError(
                        f"Failed to generate base image: {last_exc}"
                    )

                # Step 2: build progressive 2x scale schedule
                # e.g. 512→2048 = [1024, 2048], 512→1024 = [1024]
                current_w, current_h = base_img.size
                scale_steps: list[tuple[int, int]] = []
                _sw, _sh = current_w, current_h
                while _sw * 2 <= target_w and _sh * 2 <= target_h:
                    _sw *= 2
                    _sh *= 2
                    scale_steps.append((_sw, _sh))
                # Final step to exact target if not a power-of-2 multiple
                if (not scale_steps) or scale_steps[-1] != (target_w, target_h):
                    scale_steps.append((target_w, target_h))

                _sys.stderr.write(
                    f"[ZIMG] Progressive scale plan: "
                    f"{current_w}x{current_h} → "
                    f"{' → '.join(f'{w}x{h}' for w, h in scale_steps)}\n"
                )
                _sys.stderr.flush()

                # Step 3: try tiled img2img refinement at each scale
                current_img = base_img
                total_tiles_refined = 0
                refinement_ok = True

                try:
                    from diffusers import ZImageImg2ImgPipeline

                    i2i_pipe = ZImageImg2ImgPipeline.from_pipe(pipe)
                    # Inherit sequential CPU offload from parent pipe.
                    # Model-level offload would be faster but OOMs on FLUX
                    # (~12 GB) with the 5070 Ti.
                    try:
                        i2i_pipe.enable_vae_tiling()
                    except Exception:
                        pass
                    try:
                        i2i_pipe.enable_vae_slicing()
                    except Exception:
                        pass

                    # Compute steps for refinement once
                    _min_effective = max(4, _ZIMAGE_STEPS)
                    _i2i_steps = max(
                        _ZIMAGE_STEPS,
                        math.ceil(_min_effective / _ZIMAGE_HIRES_STRENGTH),
                    )

                    # Split refinement prompts by tile content.
                    # Foreground tiles (high variance) get anatomy-aware
                    # prompt for sharp skin/muscle/joint detail.
                    # Background tiles (low variance) get a neutral
                    # texture-only prompt so the model doesn't
                    # hallucinate body parts in empty dark areas.
                    _fg_refine_prompt = (
                        "high quality, sharp details, fine textures, "
                        "photorealistic, 8K resolution, "
                        "anatomically correct, natural proportions"
                    )
                    _bg_refine_prompt = (
                        "high quality, sharp details, fine textures, "
                        "photorealistic, 8K resolution, "
                        "clean smooth background"
                    )
                    _negative_prompt = (
                        "deformed, bad anatomy, mechanical joints, "
                        "extra limbs, fused fingers, mutated hands, "
                        "disconnected limbs, disfigured, poorly drawn face, "
                        "long neck, blurry, lowres, watermark"
                    )
                    _sys.stderr.write(
                        f"[ZIMG] FG prompt: {_fg_refine_prompt}\n"
                        f"[ZIMG] BG prompt: {_bg_refine_prompt}\n"
                    )

                    for pass_idx, (step_w, step_h) in enumerate(scale_steps):
                        # Upscale current image to this step's size
                        upscaled = current_img.resize(
                            (step_w, step_h), Image.LANCZOS
                        )
                        _sys.stderr.write(
                            f"[ZIMG] Pass {pass_idx+1}/{len(scale_steps)}: "
                            f"upscaled to {step_w}x{step_h}\n"
                        )

                        # Tile and refine
                        positions = _compute_tile_grid(
                            step_w, step_h, tile_w, tile_h, overlap
                        )
                        stride_x = tile_w - overlap
                        stride_y = tile_h - overlap
                        cols = max(
                            1, math.ceil((step_w - overlap) / stride_x)
                        )
                        rows = max(
                            1, math.ceil((step_h - overlap) / stride_y)
                        )
                        _sys.stderr.write(
                            f"[ZIMG]   Refining {len(positions)} tiles "
                            f"({cols}x{rows}), strength={_ZIMAGE_HIRES_STRENGTH}, "
                            f"steps={_i2i_steps}...\n"
                        )
                        _sys.stderr.flush()

                        refined_tiles: list[Image.Image] = []
                        pass_ok = True
                        for i, (tx, ty) in enumerate(positions):
                            tile_crop = upscaled.crop(
                                (tx, ty, tx + tile_w, ty + tile_h)
                            )

                            # Variance gate: low-detail tiles (dark
                            # backgrounds) get much lower strength to
                            # prevent the model from hallucinating
                            # subjects in empty areas.
                            _tile_arr = np.array(tile_crop)
                            _tile_var = float(_tile_arr.var())
                            # Threshold: typical dark studio BG has
                            # variance < 200; detailed areas > 1000.
                            _BG_VAR_THRESHOLD = 300.0
                            _BG_STRENGTH_MULT = 0.3
                            if _tile_var < _BG_VAR_THRESHOLD:
                                _tile_strength = min(
                                    _ZIMAGE_HIRES_STRENGTH * _BG_STRENGTH_MULT,
                                    0.12,
                                )
                                _tile_label = "bg"
                                _tile_prompt = _bg_refine_prompt
                            else:
                                _tile_strength = _ZIMAGE_HIRES_STRENGTH
                                _tile_label = "fg"
                                _tile_prompt = _fg_refine_prompt

                            _sys.stderr.write(
                                f"[ZIMG]     tile {i+1}/"
                                f"{len(positions)} ({tx},{ty})"
                                f" var={_tile_var:.0f}"
                                f" [{_tile_label}]"
                                f" str={_tile_strength:.3f}...\n"
                            )
                            _sys.stderr.flush()
                            try:
                                with torch.inference_mode():
                                    _i2i_kwargs = dict(
                                        prompt=_tile_prompt,
                                        image=tile_crop,
                                        strength=_tile_strength,
                                        width=tile_w,
                                        height=tile_h,
                                        num_inference_steps=_i2i_steps,
                                        guidance_scale=_ZIMAGE_GUIDANCE,
                                        max_sequence_length=_ZIMAGE_MAX_SEQ,
                                    )
                                    # Add negative prompt if the pipeline
                                    # supports it (FLUX pipelines may not).
                                    if _ZIMAGE_GUIDANCE > 0:
                                        _i2i_kwargs["negative_prompt"] = (
                                            _negative_prompt
                                        )
                                    ref_result = i2i_pipe(**_i2i_kwargs)
                                refined_tiles.append(ref_result.images[0])
                                torch.cuda.empty_cache()
                            except RuntimeError as oom_exc:
                                if "out of memory" in str(oom_exc).lower():
                                    _sys.stderr.write(
                                        f"[ZIMG]     OOM on tile {i+1}, "
                                        f"using Lanczos for this pass\n"
                                    )
                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    pass_ok = False
                                    break
                                raise

                        if pass_ok and len(refined_tiles) == len(positions):
                            current_img = stitch_tiles(
                                refined_tiles, positions,
                                step_w, step_h, overlap,
                            )
                            total_tiles_refined += len(refined_tiles)
                            _sys.stderr.write(
                                f"[ZIMG]   Pass {pass_idx+1} complete "
                                f"({step_w}x{step_h})\n"
                            )
                        else:
                            # OOM on this pass — use Lanczos result, skip
                            # further refinement passes
                            current_img = upscaled
                            refinement_ok = False
                            _sys.stderr.write(
                                f"[ZIMG]   Pass {pass_idx+1} failed, "
                                f"using Lanczos-only from here\n"
                            )
                            # Still upscale remaining steps with Lanczos
                            for _, (rw, rh) in enumerate(
                                scale_steps[pass_idx+1:]
                            ):
                                current_img = current_img.resize(
                                    (rw, rh), Image.LANCZOS
                                )
                            break

                    img = current_img
                    if refinement_ok:
                        gen_method = (
                            f"hi-res progressive img2img: "
                            f"{base_img.size[0]}x{base_img.size[1]} → "
                            f"{' → '.join(f'{w}x{h}' for w, h in scale_steps)}, "
                            f"{total_tiles_refined} tiles total, "
                            f"strength={_ZIMAGE_HIRES_STRENGTH}, overlap={overlap}"
                        )
                    else:
                        gen_method = (
                            f"hi-res partial: base "
                            f"{base_img.size[0]}x{base_img.size[1]} → "
                            f"{target_w}x{target_h} "
                            f"(some passes Lanczos-only due to OOM)"
                        )
                    _sys.stderr.write(
                        f"[ZIMG] Tiled refinement complete "
                        f"({target_w}x{target_h})\n"
                    )

                    del i2i_pipe
                except ImportError:
                    # No img2img pipeline — just Lanczos all the way
                    _sys.stderr.write(
                        "[ZIMG] ZImageImg2ImgPipeline not available, "
                        "using Lanczos-only upscale\n"
                    )
                    img = base_img.resize(
                        (target_w, target_h), Image.LANCZOS
                    )
                    gen_method = (
                        f"hi-res Lanczos-only: base "
                        f"{base_img.size[0]}x{base_img.size[1]} → "
                        f"{target_w}x{target_h} "
                        f"(ZImageImg2ImgPipeline not available)"
                    )
                except Exception as ref_exc:
                    _sys.stderr.write(
                        f"[ZIMG] img2img refinement failed ({ref_exc}), "
                        f"using Lanczos-only upscale\n"
                    )
                    img = base_img.resize(
                        (target_w, target_h), Image.LANCZOS
                    )
                    gen_method = (
                        f"hi-res Lanczos-only: base "
                        f"{base_img.size[0]}x{base_img.size[1]} → "
                        f"{target_w}x{target_h} "
                        f"(refinement error: {ref_exc})"
                    )

                fname = f"{_uuid.uuid4().hex[:12]}.png"
                img.save(_GENERATED_DIR / fname)
                img_tag = f"![generated image](/static/generated/{fname})"
                _sys.stderr.write(
                    f"[ZIMG] Hi-res done ({target_w}x{target_h}): "
                    f"{img_tag}\n"
                )
                _sys.stderr.flush()
            else:
                # ---- Standard single-tile generation ----
                # Try configured resolution first; on OOM, retry at smaller sizes
                _sizes = [(tile_w, tile_h)]
                if (tile_w, tile_h) != (384, 384):
                    _sizes.append((384, 384))
                if (tile_w, tile_h) != (256, 256):
                    _sizes.append((256, 256))

                for _w, _h in _sizes:
                    try:
                        _sys.stderr.write(
                            f"[ZIMG] Running inference ({_w}x{_h}): "
                            f"{prompt[:80]}\n"
                        )
                        _sys.stderr.flush()

                        with torch.inference_mode():
                            result = pipe(
                                prompt=prompt,
                                width=_w,
                                height=_h,
                                num_inference_steps=_ZIMAGE_STEPS,
                                guidance_scale=_ZIMAGE_GUIDANCE,
                                max_sequence_length=_ZIMAGE_MAX_SEQ,
                            )

                        img = result.images[0]

                        fname = f"{_uuid.uuid4().hex[:12]}.png"
                        img.save(_GENERATED_DIR / fname)
                        img_tag = f"![generated image](/static/generated/{fname})"
                        gen_method = f"standard: {_w}x{_h}"
                        _sys.stderr.write(f"[ZIMG] Done: {img_tag}\n")
                        _sys.stderr.flush()
                        break  # success
                    except RuntimeError as oom_exc:
                        if "out of memory" in str(oom_exc).lower():
                            _sys.stderr.write(
                                f"[ZIMG] OOM at {_w}x{_h}, trying smaller...\n"
                            )
                            torch.cuda.empty_cache()
                            gc.collect()
                            last_exc = oom_exc
                            continue
                        raise  # non-OOM RuntimeError — don't retry

        except Exception as e:
            last_exc = e
            _sys.stderr.write(f"[ZIMG] ERROR: {e}\n")
            _sys.stderr.flush()

        finally:
            # Free ZImage pipeline from GPU BEFORE reloading LLM
            global _zimage_pipe
            try:
                if _zimage_pipe is not None:
                    _zimage_pipe.to("cpu")
                    _sys.stderr.write("[ZIMG] Moved pipeline to CPU\n")
            except Exception:
                pass
            _zimage_pipe = None
            if pipe is not None:
                del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            _free = torch.cuda.mem_get_info()[0] / (1024**3)
            _sys.stderr.write(f"[ZIMG] GPU freed — {_free:.2f} GiB available\n")

            # Reload the LLM(s) now that VRAM is free
            for _mname in (_unloaded_models or [os.getenv("MODEL", "qwen3.5:9b").replace("ollama/", "")]):
                try:
                    _requests.post(
                        OLLAMA_IMAGE_URL,
                        json={"model": _mname, "prompt": ""},
                        timeout=30,
                    )
                    _sys.stderr.write(f"[ZIMG] Reloaded LLM: {_mname}\n")
                except Exception as e:
                    _sys.stderr.write(f"[ZIMG] Failed to reload LLM {_mname}: {e}\n")

        if img_tag:
            return f"{img_tag}\n<!-- method: {gen_method} -->"
        return f"Error generating image via ZImagePipeline: {last_exc}"
    else:
        return f"Error: Unknown IMAGE_BACKEND '{backend}'. Use 'sd' or 'ollama'."


# The image tag IS the final answer — skip the post-tool LLM call.
# Without this, CrewAI sends the result back to the LLM which returns
# None/empty because Ollama was just unloaded+reloaded for VRAM.
generate_ai_image.result_as_answer = True
