"""AI image generation pipelines (Stable Diffusion, ZImage-Turbo).

Handles lazy pipeline loading, VRAM management, Ollama unload/reload,
and OOM retry cascade.  Extracted from crew.py during v1.1.0 refactoring.
"""

import os
import gc
import threading
import uuid as _uuid
import logging
from pathlib import Path

from crewai.tools import tool

_logger = logging.getLogger(__name__)

_GENERATED_DIR = Path(__file__).parent / "static" / "generated"
_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

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
_ZIMAGE_MODEL_ID = os.getenv("ZIMAGE_MODEL", "mrfakename/Z-Image-Turbo")
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
        last_exc = None
        pipe = None
        try:
            pipe = _get_zimage_pipe()

            width = int(os.getenv("ZIMAGE_WIDTH", "512"))
            height = int(os.getenv("ZIMAGE_HEIGHT", "512"))

            # Try configured resolution first; on OOM, retry at smaller sizes
            _sizes = [(width, height)]
            if (width, height) != (384, 384):
                _sizes.append((384, 384))
            if (width, height) != (256, 256):
                _sizes.append((256, 256))

            for _w, _h in _sizes:
                try:
                    _sys.stderr.write(
                        f"[ZIMG] Running inference ({_w}x{_h}): {prompt[:80]}\n"
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
            return img_tag
        return f"Error generating image via ZImagePipeline: {last_exc}"
    else:
        return f"Error: Unknown IMAGE_BACKEND '{backend}'. Use 'sd' or 'ollama'."


# The image tag IS the final answer — skip the post-tool LLM call.
# Without this, CrewAI sends the result back to the LLM which returns
# None/empty because Ollama was just unloaded+reloaded for VRAM.
generate_ai_image.result_as_answer = True
