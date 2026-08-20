"""AI image generation pipelines (Stable Diffusion, ZImage-Turbo).

Handles lazy pipeline loading, VRAM management, Ollama unload/reload,
OOM retry cascade, and high-resolution tiled generation with overlap
blending.  Extracted from crew.py during v1.1.0 refactoring.
"""

import cv2
import io
import math
import os
import gc
import json
import re
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

# Permanent model weights directory — inside the project, never cleared by OS cache tools.
# Falls back to the legacy cache path so existing downloads aren't re-fetched.
_MODELS_DIR = Path(__file__).parent / "data" / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
            tw,
            th,
            overlap,
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

# Image backend selection ("sd", "zimage", or "ernie")
IMAGE_BACKEND = os.getenv("IMAGE_BACKEND", "sd")

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
_REALESRGAN_ENABLED = os.getenv("IMAGE_REALESRGAN", "0") == "1"

# Default image params dict — mirrors the module vars above.
# Keys here are what the API / UI use; values are the mutable module state.
DEFAULT_IMAGE_PARAMS: dict = {
    "image_backend": os.getenv("IMAGE_BACKEND", "sd"),
    "hires_enabled": False,
    "hires_width": 2048,
    "hires_height": 2048,
    "tile_overlap": 192,
    "hires_strength": 0.35,
    "realesrgan_enabled": False,
    "tile_width": int(os.getenv("ZIMAGE_WIDTH", "512")),
    "tile_height": int(os.getenv("ZIMAGE_HEIGHT", "512")),
}


def get_image_params() -> dict:
    """Return the current runtime image generation parameters."""
    return {
        "image_backend": IMAGE_BACKEND,
        "hires_enabled": ZIMAGE_HIRES,
        "hires_width": _ZIMAGE_HIRES_WIDTH,
        "hires_height": _ZIMAGE_HIRES_HEIGHT,
        "tile_overlap": _ZIMAGE_TILE_OVERLAP,
        "hires_strength": _ZIMAGE_HIRES_STRENGTH,
        "realesrgan_enabled": _REALESRGAN_ENABLED,
        "tile_width": int(os.getenv("ZIMAGE_WIDTH", "512")),
        "tile_height": int(os.getenv("ZIMAGE_HEIGHT", "512")),
        "refine_rounds": IMAGE_REFINE_ROUNDS,
        "critic_model": IMAGE_CRITIC_MODEL,
    }


_VALID_BACKENDS = {"sd", "zimage", "ernie", "flux"}


def update_image_params(params: dict) -> dict:
    """Update runtime image parameters. Only known keys are accepted."""
    global IMAGE_BACKEND, ZIMAGE_HIRES, _ZIMAGE_HIRES_WIDTH, _ZIMAGE_HIRES_HEIGHT, _ZIMAGE_TILE_OVERLAP, _ZIMAGE_HIRES_STRENGTH, _REALESRGAN_ENABLED
    global IMAGE_REFINE_ROUNDS, IMAGE_CRITIC_MODEL
    if "image_backend" in params:
        val = str(params["image_backend"]).lower()
        if val in _VALID_BACKENDS:
            IMAGE_BACKEND = val
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
    if "realesrgan_enabled" in params:
        _REALESRGAN_ENABLED = bool(params["realesrgan_enabled"])
    if "refine_rounds" in params:
        IMAGE_REFINE_ROUNDS = max(0, min(5, int(params["refine_rounds"])))
    if "critic_model" in params:
        val = str(params["critic_model"]).strip()
        if val:
            IMAGE_CRITIC_MODEL = val
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


from contextlib import contextmanager as _contextmanager


@_contextmanager
def _gpu_inference_ctx(tag: str = "IMG", min_vram_gib: float = 4.0):
    """Unified context manager for GPU image inference.

    1. Unloads all Ollama models via /api/ps to free VRAM.
    2. Waits until >4 GiB VRAM is free (up to 20 s).
    3. Disables cuDNN + Flash/MemEfficient SDP to avoid
       CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (Resolve ships
       cuDNN 8 in /opt/resolve/libs which conflicts with system cuDNN 9).
    4. Yields the list of unloaded model names.
    5. On exit: restores SDP/cuDNN flags, frees CUDA cache,
       reloads the previously unloaded Ollama model(s).
    """
    import sys as _sys
    import time as _time
    import torch
    import requests as _req

    _ollama_base = OLLAMA_IMAGE_URL.rsplit("/", 2)[0]
    _unloaded: list[str] = []

    # -- 1. Unload all Ollama models --
    try:
        _ps = _req.get(f"{_ollama_base}/api/ps", timeout=10)
        _ps.raise_for_status()
        for _m in _ps.json().get("models", []):
            _mname = _m.get("name", "")
            if _mname:
                _req.post(OLLAMA_IMAGE_URL,
                          json={"model": _mname, "keep_alive": 0, "prompt": ""},
                          timeout=30)
                _unloaded.append(_mname)
        _sys.stderr.write(f"[{tag}] Unloaded LLM(s): {_unloaded}\n")
    except Exception as _e:
        _sys.stderr.write(f"[{tag}] LLM unload failed: {_e}\n")
        _fallback = os.getenv("MODEL", "qwen3.5:9b").replace("ollama/", "")
        try:
            _req.post(OLLAMA_IMAGE_URL,
                      json={"model": _fallback, "keep_alive": 0, "prompt": ""},
                      timeout=30)
            _unloaded.append(_fallback)
        except Exception:
            pass

    # -- 2. Wait for VRAM --
    _sys.stderr.write(f"[{tag}] Waiting for VRAM...\n")
    for _ in range(20):
        torch.cuda.empty_cache()
        _free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        _sys.stderr.write(f"[{tag}]   free VRAM: {_free:.2f} GiB\n")
        if _free > min_vram_gib:
            break
        _time.sleep(1)
    else:
        _sys.stderr.write(f"[{tag}] Warning: VRAM still tight\n")

    # -- 3. Disable cuDNN / Flash SDP --
    # System has cuDNN 9 (PyTorch) + cuDNN 8 (DaVinci Resolve in /opt/resolve/libs).
    # Flash/MemEfficient SDP triggers the mismatch; math SDP avoids cuDNN entirely.
    _cudnn_was = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    try:
        yield _unloaded
    finally:
        # -- 4. Restore cuDNN / SDP --
        torch.backends.cudnn.enabled = _cudnn_was
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        _free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        _sys.stderr.write(f"[{tag}] GPU freed — {_free:.2f} GiB available\n")

        # -- 5. Reload Ollama LLM(s) --
        _reload_list = _unloaded or [os.getenv("MODEL", "qwen3.5:9b").replace("ollama/", "")]
        for _mname in _reload_list:
            try:
                _req.post(OLLAMA_IMAGE_URL,
                          json={"model": _mname, "prompt": ""},
                          timeout=30)
                _sys.stderr.write(f"[{tag}] Reloaded LLM: {_mname}\n")
            except Exception as _re:
                _sys.stderr.write(f"[{tag}] Failed to reload LLM {_mname}: {_re}\n")


def _hf_hub_cache_root() -> Path:
    """Resolve Hugging Face hub cache root, honoring HF env vars."""
    explicit = os.getenv("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_model_cache_present(repo_id: str) -> bool:
    """True if a model is cached on disk AND the download finished.

    The snapshots directory is created as soon as a download *starts*, so
    checking only for its existence lets a half-fetched model look ready — and
    then a chat request silently resumes a multi-gigabyte download instead of
    failing fast. Unfinished blobs are staged as ``*.incomplete``, so their
    absence is what actually distinguishes "downloaded" from "downloading".
    """
    model_dir = _hf_hub_cache_root() / f"models--{repo_id.replace('/', '--')}"
    snapshots = model_dir / "snapshots"
    if not snapshots.exists() or not snapshots.is_dir():
        return False
    snaps = [d for d in snapshots.iterdir() if d.is_dir()]
    if not snaps:
        return False
    snap = max(snaps, key=lambda d: d.stat().st_mtime)

    # A dangling symlink means a blob was staged but never finished.
    for link in snap.rglob("*"):
        if link.is_symlink() and not link.exists():
            return False

    # For a diffusers pipeline, model_index.json names the component folders.
    # Each must actually carry weights — a half-fetched repo leaves the folder
    # present but empty, which a symlink check alone would not catch.
    index = snap / "model_index.json"
    if index.is_file():
        try:
            spec = json.loads(index.read_text())
        except Exception:
            return False
        for name, val in spec.items():
            if name.startswith("_") or not isinstance(val, (list, tuple)):
                continue
            comp = snap / name
            if not comp.is_dir():
                return False
            has_weights = any(comp.glob("*.safetensors")) or any(comp.glob("*.bin"))
            has_config = any(comp.glob("*.json")) or any(comp.glob("*.model"))
            if not (has_weights or has_config):
                return False
    return True


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
                offline_transformers = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
                offline_hf_hub = os.getenv("HF_HUB_OFFLINE", "0") == "1"
                offline = offline_transformers or offline_hf_hub
                has_cache = _hf_model_cache_present(_SD_MODEL_ID)
                if not has_cache:
                    if offline:
                        raise RuntimeError(
                            "SD model cache missing while offline. "
                            "Unset HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE to allow download."
                        )
                    print(
                        f"[SD] Cache missing for {_SD_MODEL_ID}; bootstrapping download…",
                        flush=True,
                    )
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
                            local_files_only=offline,
                            cache_dir=str(_MODELS_DIR / "sd"),
                        )
                finally:
                    _sys.stdout.close()
                    _sys.stdout = _real_stdout
                _sd_pipe.enable_model_cpu_offload()
                print("[SD] Pipeline ready.", flush=True)
    return _sd_pipe


def preload_sd():
    """Call from the server to warm up the SD pipeline in a background thread."""

    def _warmup():
        # Opt-in: temporarily disable offline flags during warm-up bootstrap.
        # Keep default OFF so TRANSFORMERS_OFFLINE=1 remains authoritative.
        force_online = os.getenv("IMAGE_WARMUP_FORCE_ONLINE", "0") == "1"
        old_t = os.environ.get("TRANSFORMERS_OFFLINE")
        old_h = os.environ.get("HF_HUB_OFFLINE")
        try:
            if force_online:
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                os.environ["HF_HUB_OFFLINE"] = "0"
            _get_sd_pipe()
        except Exception as e:
            _logger.warning("SD warm-up skipped: %s", e)
        finally:
            if old_t is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_t
            if old_h is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_h

    threading.Thread(target=_warmup, daemon=True).start()


# --------------- ZImagePipeline backend (diffusers + CUDA) ---------------
# Used when IMAGE_BACKEND=zimage.
# Reads:
#   ZIMAGE_MODEL          — HuggingFace repo id (default: mrfakename/Z-Image-Turbo)
#   HUGGINGFACE_TOKEN     — HF token for gated repos
#   TRANSFORMERS_OFFLINE  — set to "1" to skip network after first download
# Ollama is NOT involved for image generation; it is only used to
# unload/reload the qwen3 LLM around inference to free VRAM.
_ZIMAGE_MODEL_ID = (
    os.getenv("ZIMAGE_MODEL", "mrfakename/Z-Image-Turbo").split("#")[0].strip()
)
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
                offline_transformers = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
                offline_hf_hub = os.getenv("HF_HUB_OFFLINE", "0") == "1"
                offline = offline_transformers or offline_hf_hub
                has_cache = _hf_model_cache_present(_ZIMAGE_MODEL_ID)
                if not has_cache:
                    if offline:
                        raise RuntimeError(
                            "ZImage model cache missing while offline. "
                            "Unset HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE to allow download."
                        )
                    _sys.stderr.write(
                        f"[ZIMG] Cache missing for {_ZIMAGE_MODEL_ID}; bootstrapping download...\n"
                    )

                print(
                    f"[ZIMG] Loading ZImagePipeline ({_ZIMAGE_MODEL_ID})"
                    f"{' [offline]' if offline else ''}...",
                    flush=True,
                )
                try:
                    pipe = ZImagePipeline.from_pretrained(
                        _ZIMAGE_MODEL_ID,
                        torch_dtype=torch.bfloat16,
                        token=hf_token,
                        local_files_only=offline,
                        cache_dir=str(_MODELS_DIR / "zimage"),
                    )
                except Exception as e:
                    msg = str(e).lower()
                    cache_miss = (
                        "cannot find an appropriate cached snapshot folder" in msg
                        or "outgoing traffic has been disabled" in msg
                    )
                    cache_corrupt = "no file named" in msg and "snapshots" in msg

                    if offline and (cache_miss or cache_corrupt):
                        raise RuntimeError(
                            "ZImage cache is missing or incomplete while offline. "
                            "Temporarily unset HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE "
                            "(or set IMAGE_WARMUP_FORCE_ONLINE=1) to repair cache."
                        ) from e

                    # If online is allowed and cache is bad/incomplete, force-refresh.
                    if (not offline) and (cache_miss or cache_corrupt):
                        _sys.stderr.write(
                            "[ZIMG] Cache missing/incomplete. Forcing model re-download...\n"
                        )
                        pipe = ZImagePipeline.from_pretrained(
                            _ZIMAGE_MODEL_ID,
                            torch_dtype=torch.bfloat16,
                            token=hf_token,
                            local_files_only=False,
                            force_download=True,
                            cache_dir=str(_MODELS_DIR / "zimage"),
                        )
                    else:
                        raise
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
                        _sys.stderr.write(
                            "[ZIMG] Model CPU offload enabled (fallback)\n"
                        )
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

                # Flash Attention is intentionally NOT enabled here.
                # ZImage's transformer uses cuDNN-backed Flash Attention which
                # triggers CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on this system.
                # math SDP (no cuDNN) is the safe fallback.
                _zimage_pipe = pipe
                print("[ZIMG] Pipeline ready.", flush=True)
    return _zimage_pipe


def preload_zimage():
    """Warm up the ZImagePipeline in a background thread."""

    def _warmup():
        # Opt-in: temporarily disable offline flags during warm-up bootstrap.
        # Keep default OFF so TRANSFORMERS_OFFLINE=1 remains authoritative.
        force_online = os.getenv("IMAGE_WARMUP_FORCE_ONLINE", "0") == "1"
        old_t = os.environ.get("TRANSFORMERS_OFFLINE")
        old_h = os.environ.get("HF_HUB_OFFLINE")
        try:
            if force_online:
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                os.environ["HF_HUB_OFFLINE"] = "0"
            _get_zimage_pipe()
        except Exception as e:
            _logger.warning("ZImage warm-up skipped: %s", e)
        finally:
            if old_t is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_t
            if old_h is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_h

    threading.Thread(target=_warmup, daemon=True).start()


# --------------- ERNIE-Image backend (diffusers ErnieImagePipeline + GGUF abliterated) ---------------
# Used when IMAGE_BACKEND=ernie.
# Base pipeline: baidu/ERNIE-Image (transformer, VAE, scheduler)
# PE + text_encoder: ponpoke/ERNIE-Image-Abliterated-GGUF (GGUF quantized, uncensored)
# Reads:
#   ERNIE_GGUF_VARIANT — quantization variant: Q4_K_M (default), Q6_K, Q8_0
#   ERNIE_STEPS        — inference steps (default: 20)
_ERNIE_BASE_ID = "baidu/ERNIE-Image"
_ERNIE_GGUF_ID = "ponpoke/ERNIE-Image-Abliterated-GGUF"
_ERNIE_VARIANT = os.getenv("ERNIE_GGUF_VARIANT", "Q4_K_M")
_ERNIE_STEPS = int(os.getenv("ERNIE_STEPS", "20"))
_ernie_pipe = None
_ernie_init_lock = threading.Lock()


def _get_ernie_pipe():
    """Lazy-load ErnieImagePipeline with GGUF abliterated PE and text_encoder."""
    global _ernie_pipe
    if _ernie_pipe is None:
        with _ernie_init_lock:
            if _ernie_pipe is None:
                import torch
                from diffusers import ErnieImagePipeline
                from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
                import sys as _sys

                hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
                offline = (
                    os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
                    or os.getenv("HF_HUB_OFFLINE", "0") == "1"
                )
                ernie_dir = _MODELS_DIR / "ernie"
                ernie_dir.mkdir(parents=True, exist_ok=True)
                variant = _ERNIE_VARIANT

                # ── Step 1: load the base pipeline (transformer, VAE, scheduler) ──
                print(f"[ERNIE] Loading base pipeline from {_ERNIE_BASE_ID}...", flush=True)
                pipe = ErnieImagePipeline.from_pretrained(
                    _ERNIE_BASE_ID,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                    token=hf_token,
                    local_files_only=offline,
                    cache_dir=str(ernie_dir / "base"),
                )

                # ── Step 2: swap PE and text_encoder with GGUF abliterated versions ──
                pe_gguf = f"ernie-image-abliterated-pe-{variant}.gguf"
                te_gguf = f"ernie-image-abliterated-te-{variant}.gguf"
                print(
                    f"[ERNIE] Loading abliterated PE ({pe_gguf}) and"
                    f" text_encoder ({te_gguf})...",
                    flush=True,
                )
                try:
                    abliterated_pe = AutoModelForCausalLM.from_pretrained(
                        _ERNIE_GGUF_ID,
                        subfolder="abliterated_pe-GGUF",
                        gguf_file=pe_gguf,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="eager",
                        token=hf_token,
                        local_files_only=offline,
                        cache_dir=str(ernie_dir / "gguf"),
                    )
                    abliterated_te = AutoModel.from_pretrained(
                        _ERNIE_GGUF_ID,
                        subfolder="abliterated_text_encoder-GGUF",
                        gguf_file=te_gguf,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="eager",
                        token=hf_token,
                        local_files_only=offline,
                        cache_dir=str(ernie_dir / "gguf"),
                    )
                    pipe.pe = abliterated_pe
                    pipe.text_encoder = abliterated_te
                    _sys.stderr.write("[ERNIE] Abliterated GGUF components loaded.\n")
                except Exception as e:
                    _sys.stderr.write(
                        f"[ERNIE] WARNING: Could not load GGUF components ({e}),"
                        " falling back to base model components.\n"
                    )

                # Sequential offload streams one submodule at a time through GPU —
                # safer than model_cpu_offload for custom/GGUF architectures where
                # accelerate's bulk-offload hooks are not applied to all submodules.
                try:
                    pipe.enable_sequential_cpu_offload()
                    _sys.stderr.write("[ERNIE] Sequential CPU offload enabled\n")
                except Exception:
                    _sys.stderr.write("[ERNIE] Sequential CPU offload unavailable\n")
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass
                try:
                    pipe.enable_vae_tiling()
                except Exception:
                    pass
                _ernie_pipe = pipe
                print("[ERNIE] Pipeline ready.", flush=True)
    return _ernie_pipe


def preload_ernie():
    """Warm up the ErnieImagePipeline in a background thread."""

    def _warmup():
        force_online = os.getenv("IMAGE_WARMUP_FORCE_ONLINE", "0") == "1"
        old_t = os.environ.get("TRANSFORMERS_OFFLINE")
        old_h = os.environ.get("HF_HUB_OFFLINE")
        try:
            if force_online:
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                os.environ["HF_HUB_OFFLINE"] = "0"
            _get_ernie_pipe()
        except Exception as e:
            _logger.warning("ERNIE warm-up skipped: %s", e)
        finally:
            if old_t is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_t
            if old_h is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_h

    threading.Thread(target=_warmup, daemon=True).start()


# --------------- FLUX.1-schnell backend (diffusers FluxPipeline, bf16) ---------------
# Used when IMAGE_BACKEND=flux.
# Model: black-forest-labs/FLUX.1-schnell (Apache 2.0, no safety filter)
# Reads:
#   FLUX_MODEL  — HuggingFace repo id (default: black-forest-labs/FLUX.1-schnell)
#   FLUX_STEPS  — inference steps (default: 4, schnell is distilled)
_FLUX_MODEL_ID = os.getenv("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
_FLUX_STEPS = int(os.getenv("FLUX_STEPS", "4"))
_flux_pipe = None
_flux_init_lock = threading.Lock()


def _get_flux_pipe():
    """Lazy-load FluxPipeline with sequential CPU offload for 16GB VRAM."""
    global _flux_pipe
    if _flux_pipe is None:
        with _flux_init_lock:
            if _flux_pipe is None:
                import torch
                from diffusers import FluxPipeline
                import sys as _sys

                hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
                offline = (
                    os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
                    or os.getenv("HF_HUB_OFFLINE", "0") == "1"
                )
                print(
                    f"[FLUX] Loading FluxPipeline ({_FLUX_MODEL_ID})"
                    f"{' [offline]' if offline else ''}...",
                    flush=True,
                )
                _flux_local = _MODELS_DIR / "flux"
                _flux_valid = (
                    (_flux_local / "model_index.json").exists()
                    and (_flux_local / "tokenizer" / "vocab.json").exists()
                    and (_flux_local / "transformer").exists()
                    and not (_flux_local / ".gitattributes").exists()
                )
                _flux_src = str(_flux_local) if _flux_valid else _FLUX_MODEL_ID
                pipe = FluxPipeline.from_pretrained(
                    _flux_src,
                    torch_dtype=torch.bfloat16,
                    token=hf_token,
                    local_files_only=offline or _flux_valid,
                    cache_dir=str(_MODELS_DIR) if not _flux_valid else None,
                )
                # Sequential offload: one layer at a time on GPU → minimal VRAM use
                pipe.enable_sequential_cpu_offload()
                _flux_pipe = pipe
                print("[FLUX] Pipeline ready.", flush=True)
    return _flux_pipe


def preload_flux():
    """Warm up the FluxPipeline in a background thread."""
    def _warmup():
        force_online = os.getenv("IMAGE_WARMUP_FORCE_ONLINE", "0") == "1"
        old_t = os.environ.get("TRANSFORMERS_OFFLINE")
        old_h = os.environ.get("HF_HUB_OFFLINE")
        try:
            if force_online:
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                os.environ["HF_HUB_OFFLINE"] = "0"
            _get_flux_pipe()
        except Exception as e:
            _logger.warning("FLUX warm-up skipped: %s", e)
        finally:
            if old_t is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_t
            if old_h is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_h

    threading.Thread(target=_warmup, daemon=True).start()


def apply_high_pass_filter(image: Image.Image, strength: float = 0.3) -> Image.Image:
    """BH Simple high-pass filter for sharpness enhancement"""
    img_array = np.array(image)
    if img_array.ndim == 2:  # Grayscale
        img_array = np.expand_dims(img_array, axis=2)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    img_float = img_array.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img_float, (0, 0), 1.0)
    high_pass = img_float - blurred
    result = img_float + strength * high_pass
    result = np.clip(result, 0, 1)
    return Image.fromarray((result * 255).astype(np.uint8))


def apply_advanced_high_pass(
    image: Image.Image, radius: int = 3, strength: float = 0.4
) -> Image.Image:
    """BH Fixed advanced high-pass filter with proper type handling"""
    img_array = np.array(image)
    if img_array.ndim == 2:  # Grayscale
        img_array = np.expand_dims(img_array, axis=2)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Convert to float32 consistently
    img_float = img_array.astype(np.float32) / 255.0

    # Apply Laplacian with proper type handling
    laplacian = cv2.Laplacian(img_float, cv2.CV_32F, ksize=radius)
    high_pass = cv2.convertScaleAbs(laplacian).astype(np.float32) / 255.0

    # Blend with proper type matching
    result = cv2.addWeighted(img_float, 1.0, high_pass, strength, 0)
    result = np.clip(result, 0, 1)

    return Image.fromarray((result * 255).astype(np.uint8))


def _apply_realesrgan(img: Image.Image) -> Image.Image:
    """Apply Real-ESRGAN x2 super-resolution to a PIL image.

    Downloads RealESRGAN_x2plus.pth on first use via HuggingFace Hub.
    Returns image at 2x the input resolution, or the original image if
    spandrel is not installed, weights are unavailable, or inference fails.
    """
    import sys as _sys
    import torch

    try:
        import spandrel
    except ImportError:
        _sys.stderr.write("[ESRGAN] spandrel not installed — skipping\n")
        return img

    # Resolution order for weights: permanent project dir first, then legacy cache
    _legacy_dir = Path.home() / ".cache" / "researcher_agent" / "realesrgan"
    weights_path = None
    for candidate_name in ("RealESRGAN_x2.pth", "RealESRGAN_x2plus.pth"):
        for search_dir in (_MODELS_DIR, _legacy_dir):
            p = search_dir / candidate_name
            if p.exists():
                weights_path = p
                break
        if weights_path:
            break

    if weights_path is None:
        offline = (
            os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
            or os.getenv("HF_HUB_OFFLINE", "0") == "1"
        )
        if offline:
            _sys.stderr.write(
                "[ESRGAN] Weights not found and offline mode is set — skipping\n"
            )
            return img
        try:
            from huggingface_hub import hf_hub_download
            _sys.stderr.write("[ESRGAN] Downloading RealESRGAN_x2.pth to project models dir...\n")
            hf_hub_download(
                repo_id="ai-forever/Real-ESRGAN",
                filename="RealESRGAN_x2.pth",
                local_dir=str(_MODELS_DIR),
            )
            weights_path = _MODELS_DIR / "RealESRGAN_x2.pth"
            _sys.stderr.write(f"[ESRGAN] Weights saved to {weights_path}\n")
        except Exception as _e:
            _sys.stderr.write(f"[ESRGAN] Failed to download weights: {_e}\n")
            return img

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    try:
        model_descriptor = spandrel.ModelLoader().load_from_file(str(weights_path))
        model = model_descriptor.model.eval().to(device)
        if device == "cuda":
            model = model.half()

        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_t = img_t.half().to(device) if device == "cuda" else img_t.to(device)

        with torch.inference_mode():
            out_t = model(img_t)

        out_np = out_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        result = Image.fromarray((out_np * 255).astype(np.uint8))
        _sys.stderr.write(
            f"[ESRGAN] {img.size[0]}x{img.size[1]} → "
            f"{result.size[0]}x{result.size[1]}\n"
        )
        return result
    except Exception as _e:
        _sys.stderr.write(f"[ESRGAN] Inference failed: {_e}\n")
        return img
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _generate_image_once(prompt: str) -> str:
    """Render one image and return its markdown/HTML tag.

    Single-shot: no inspection of the result. ``generate_ai_image`` wraps this
    in a critique-and-retry loop.
    """
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
        import torch

        img_tag = None
        gen_method = "unknown"
        last_exc = None
        pipe = None
        with _gpu_inference_ctx("ZIMG"):
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
                        f"[ZIMG] Hi-res: generating base image " f"({tile_w}x{tile_h})...\n"
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
                        raise RuntimeError(f"Failed to generate base image: {last_exc}")

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
                    _esrgan_used = False

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
                            f"[ZIMG] REALESRGAN_ENABLED={_REALESRGAN_ENABLED}\n"
                        )

                        for pass_idx, (step_w, step_h) in enumerate(scale_steps):
                            # Upscale current image to this step's size
                            upscaled = current_img.resize((step_w, step_h), Image.LANCZOS)
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
                            cols = max(1, math.ceil((step_w - overlap) / stride_x))
                            rows = max(1, math.ceil((step_h - overlap) / stride_y))
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
                                    refined_tiles,
                                    positions,
                                    step_w,
                                    step_h,
                                    overlap,
                                )
                                total_tiles_refined += len(refined_tiles)
                                _sys.stderr.write(
                                    f"[ZIMG]   Pass {pass_idx+1} complete "
                                    f"({step_w}x{step_h})\n"
                                )
                                # Real-ESRGAN branch: after first pass use ESRGAN
                                # instead of continuing with more img2img passes
                                if _REALESRGAN_ENABLED and pass_idx == 0 and len(scale_steps) > 1:
                                    _sys.stderr.write(
                                        "[ZIMG] Real-ESRGAN: upscaling after first pass...\n"
                                    )
                                    _esrgan_out = _apply_realesrgan(current_img)
                                    if _esrgan_out.size != (target_w, target_h):
                                        current_img = _esrgan_out.resize(
                                            (target_w, target_h), Image.LANCZOS
                                        )
                                    else:
                                        current_img = _esrgan_out
                                    _esrgan_used = True
                                    break  # skip remaining img2img passes
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
                                for _, (rw, rh) in enumerate(scale_steps[pass_idx + 1 :]):
                                    current_img = current_img.resize(
                                        (rw, rh), Image.LANCZOS
                                    )
                                break

                        img = current_img
                        if refinement_ok:
                            if _esrgan_used:
                                gen_method = (
                                    f"hi-res img2img+Real-ESRGAN: "
                                    f"{base_img.size[0]}x{base_img.size[1]} → "
                                    f"{scale_steps[0][0]}x{scale_steps[0][1]} (img2img) → "
                                    f"{img.size[0]}x{img.size[1]} (ESRGAN x2), "
                                    f"{total_tiles_refined} tiles, "
                                    f"strength={_ZIMAGE_HIRES_STRENGTH}"
                                )
                            else:
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
                        img = base_img.resize((target_w, target_h), Image.LANCZOS)
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
                        img = base_img.resize((target_w, target_h), Image.LANCZOS)
                        gen_method = (
                            f"hi-res Lanczos-only: base "
                            f"{base_img.size[0]}x{base_img.size[1]} → "
                            f"{target_w}x{target_h} "
                            f"(refinement error: {ref_exc})"
                        )

                    # BH Apply post-processing sharpening
                    # High-pass only runs when Real-ESRGAN is not active
                    if not _REALESRGAN_ENABLED:
                        try:
                            img = apply_advanced_high_pass(img, radius=3, strength=0.3)
                        except ImportError:
                            # Fallback if OpenCV not available
                            img = apply_high_pass_filter(img, strength=0.3)

                    fname = f"{_uuid.uuid4().hex[:12]}.png"
                    img.save(_GENERATED_DIR / fname)
                    img_tag = f"![generated image](/static/generated/{fname})"
                    _sys.stderr.write(
                        f"[ZIMG] Hi-res done ({target_w}x{target_h}): " f"{img_tag}\n"
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
                                f"[ZIMG] Running inference ({_w}x{_h}): " f"{prompt[:80]}\n"
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
                # Move pipeline off GPU before context manager reloads LLM
                global _zimage_pipe
                try:
                    if _zimage_pipe is not None:
                        _zimage_pipe.to("cpu")
                except Exception:
                    pass
                _zimage_pipe = None
                if pipe is not None:
                    del pipe

        if img_tag:
            return f"{img_tag}\n<!-- method: {gen_method} -->"
        return f"Error generating image via ZImagePipeline: {last_exc}"

    elif backend == "ernie":
        import torch
        result_img = None
        with _gpu_inference_ctx("ERNIE", min_vram_gib=4.0):
            try:
                pipe = _get_ernie_pipe()
            except Exception as e:
                return f"Error loading ERNIE pipeline: {e}"
            _sys.stderr.write("[ERNIE] Starting inference...\n"); _sys.stderr.flush()
            try:
                import warnings
                # Suppress the harmless device-mismatch warning from the GGUF pe
                # model: sequential_cpu_offload sets _execution_device=cuda but pe
                # (GGUF) stays on cpu. The warning fires in .generate() but is safe.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*input_ids.*being on a device type different.*")
                    with torch.inference_mode():
                        result_img = pipe(
                            prompt,
                            num_inference_steps=_ERNIE_STEPS,
                            generator=torch.Generator().manual_seed(_next_seed()),
                        ).images[0]
            except Exception as e:
                return f"Error during ERNIE generation: {e}"

        if result_img is None:
            return "Error during ERNIE generation: no image produced"
        fname = f"{_uuid.uuid4().hex[:12]}.png"
        result_img.save(str(_GENERATED_DIR / fname))
        _sys.stderr.write(f"[ERNIE] Saved: {fname}\n")
        return f'<img src="/static/generated/{fname}" alt="{prompt[:60]}">'

    elif backend == "flux":
        import torch
        result = None
        with _gpu_inference_ctx("FLUX"):
            try:
                pipe = _get_flux_pipe()
            except Exception as e:
                return f"Error loading FLUX pipeline: {e}"
            _sys.stderr.write(f"[FLUX] Starting inference ({_FLUX_STEPS} steps)...\n")
            _sys.stderr.flush()
            try:
                with _vram_lock:
                    result = pipe(
                        prompt,
                        num_inference_steps=_FLUX_STEPS,
                        guidance_scale=0.0,  # schnell is guidance-distilled
                        generator=torch.Generator().manual_seed(0),
                    ).images[0]
            except Exception as e:
                return f"Error during FLUX generation: {e}"

        if result is None:
            return "Error during FLUX generation: no image produced"
        fname = f"{_uuid.uuid4().hex[:12]}.png"
        result.save(str(_GENERATED_DIR / fname))
        _sys.stderr.write(f"[FLUX] Saved: {fname}\n")
        return f'<img src="/static/generated/{fname}" alt="{prompt[:60]}">'

    else:
        return f"Error: Unknown IMAGE_BACKEND '{backend}'. Use 'sd', 'zimage', 'ernie', or 'flux'."


# ---------------------------------------------------------------------------
# Critique-and-retry loop
#
# A diffusion model renders a prompt once and never looks at what it produced.
# The agentic image systems that outperform it (Muse Image being the current
# example) do the obvious thing instead: inspect the result, name what is wrong,
# rewrite the prompt, and render again. That loop — not better weights — is most
# of the quality difference, and it is reproducible locally with a vision model.
#
# Cost is real: every extra round is another generation plus an Ollama
# unload/reload cycle. IMAGE_REFINE_ROUNDS=0 disables it entirely.
# ---------------------------------------------------------------------------

IMAGE_REFINE_ROUNDS = int(os.getenv("IMAGE_REFINE_ROUNDS", "2"))
IMAGE_CRITIC_MODEL = os.getenv("IMAGE_CRITIC_MODEL", "qwen3-vl:8b")

_IMG_PATH_RE = re.compile(r'/static/generated/([A-Za-z0-9_.-]+\.(?:png|jpg|jpeg|webp))')


def check_critic_available() -> tuple[bool, str]:
    """Report whether the configured critic model can actually be used.

    The refine loop is fail-safe by design: if the critic errors, the image is
    kept and generation continues. That makes a misconfigured critic invisible —
    you get no refinement and no failure. Calling this at startup turns a silent
    no-op into a log line.
    """
    if IMAGE_REFINE_ROUNDS <= 0:
        return True, "image refinement disabled (IMAGE_REFINE_ROUNDS=0)"
    try:
        import requests as _req

        base = OLLAMA_IMAGE_URL.rsplit("/", 2)[0]
        resp = _req.get(f"{base}/api/tags", timeout=10)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
    except Exception as e:
        return False, f"could not reach Ollama to verify critic model: {e}"

    if IMAGE_CRITIC_MODEL in names or f"{IMAGE_CRITIC_MODEL}:latest" in names:
        return True, f"image critic ready: {IMAGE_CRITIC_MODEL}"
    return False, (
        f"IMAGE_CRITIC_MODEL '{IMAGE_CRITIC_MODEL}' is not pulled — image "
        f"refinement will silently do nothing. Run: ollama pull {IMAGE_CRITIC_MODEL}"
    )


def _next_seed() -> int:
    """Seed for a fresh sample.

    FLUX was pinned to seed 0, which made every render of the same prompt
    identical — harmless alone, but it also made the refine loop's re-roll a
    no-op. Honour an explicitly pinned seed, otherwise vary.
    """
    pinned = int(os.getenv("IMAGE_SEED", "0"))
    if pinned > 0:
        return pinned
    return _uuid.uuid4().int % (2 ** 31 - 1)


def _tag_to_path(tag: str) -> Path | None:
    """Extract the on-disk path from a returned image tag, markdown or HTML."""
    m = _IMG_PATH_RE.search(tag or "")
    if not m:
        return None
    p = _GENERATED_DIR / m.group(1)
    return p if p.is_file() else None


_CRITIC_PROMPT = (
    "You are inspecting an AI-generated image against the request it was made from.\n"
    "Judge ONLY whether the image depicts what was asked. Ignore watermarks and "
    "minor style preferences.\n\n"
    "REQUEST: {prompt}\n\n"
    "If the request names a NUMBER of things, count them in the image one by one "
    "and write the count on the COUNT line. Vision models routinely miscount when "
    "they answer straight away; counting explicitly first is what makes the "
    "verdict trustworthy.\n\n"
    "Reply in EXACTLY this format, four lines, nothing else:\n"
    "COUNT: how many of the requested item you actually see, or 'n/a'\n"
    "VERDICT: GOOD or RETRY\n"
    "PROBLEMS: one sentence naming what is wrong, or 'none'\n"
    "IMPROVED_PROMPT: a rewritten image prompt that fixes the problems, "
    "keeping every requirement of the original request. If the verdict is GOOD, "
    "repeat the original request unchanged.\n\n"
    "Say RETRY only for real failures: wrong subject, wrong count, missing or "
    "extra major elements, garbled text, malformed anatomy."
)


def _critique_image(image_path: Path, prompt: str) -> tuple[bool, str, str]:
    """Ask a local vision model whether the image matches the prompt.

    Returns ``(is_good, problems, improved_prompt)``. Any failure returns
    ``(True, ...)`` — a broken critic must never block image generation.
    """
    import sys as _sys
    import base64
    import requests as _req

    try:
        img = Image.open(image_path)
        img.thumbnail((1024, 1024), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = _req.post(
            OLLAMA_IMAGE_URL,
            json={
                "model": IMAGE_CRITIC_MODEL,
                "prompt": _CRITIC_PROMPT.format(prompt=prompt),
                "images": [b64],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            },
            timeout=180,
        )
        resp.raise_for_status()
        text = (resp.json().get("response") or "").strip()
    except Exception as e:
        _sys.stderr.write(f"[REFINE] Critic unavailable ({e}); keeping image\n")
        return True, "", prompt

    if not text:
        return True, "", prompt

    verdict, problems, improved = "GOOD", "", prompt
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("PROBLEMS:"):
            problems = line.split(":", 1)[1].strip()
        elif line.upper().startswith("IMPROVED_PROMPT:"):
            improved = line.split(":", 1)[1].strip()

    is_good = not verdict.startswith("RETRY")
    if not improved or len(improved) < 10:
        improved = prompt
    return is_good, problems, improved


@tool("GenerateAIImage")
def generate_ai_image(prompt: str) -> str:
    """Generate a realistic image from a text prompt using Stable Diffusion AI or Ollama z-image.
    Use for animals, landscapes, people, objects, scenes. Add style keywords like photorealistic, 8k.
    Copy the returned image tag EXACTLY into your Final Answer."""
    import sys as _sys

    tag = _generate_image_once(prompt)
    if IMAGE_REFINE_ROUNDS <= 0 or tag.startswith("Error"):
        return tag

    current_prompt = prompt
    for round_no in range(1, IMAGE_REFINE_ROUNDS + 1):
        path = _tag_to_path(tag)
        if path is None:
            return tag  # nothing on disk to inspect

        is_good, problems, improved = _critique_image(path, prompt)
        if is_good:
            _sys.stderr.write(f"[REFINE] Accepted after {round_no - 1} retry(s)\n")
            return tag

        _sys.stderr.write(
            f"[REFINE] Round {round_no}/{IMAGE_REFINE_ROUNDS} — {problems[:120]}\n"
        )
        if improved.strip() == current_prompt.strip():
            # The critic found a real fault but had nothing to add to the prompt —
            # which is the NORMAL case when the prompt was already precise and the
            # render simply got it wrong (miscounted objects, dropped an element).
            # Bailing here skipped the retry in exactly the situation this loop
            # exists for, so re-roll instead: these pipelines are stochastic, and
            # a different sample often fixes a count error outright.
            _sys.stderr.write("[REFINE] No better prompt offered; re-rolling\n")
        current_prompt = improved
        retry_tag = _generate_image_once(current_prompt)
        if retry_tag.startswith("Error"):
            return tag  # keep the last good render
        tag = retry_tag

    _sys.stderr.write(f"[REFINE] Exhausted {IMAGE_REFINE_ROUNDS} round(s)\n")
    return tag


# The image tag IS the final answer — skip the post-tool LLM call.
# Without this, CrewAI sends the result back to the LLM which returns
# None/empty because Ollama was just unloaded+reloaded for VRAM.
generate_ai_image.result_as_answer = True
