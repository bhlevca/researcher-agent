"""Download FLUX.1-schnell into data/models (HF hub cache format).

Run with:
    python tests/download_flux.py

The model will be saved to:
    src/researcher/data/models/models--black-forest-labs--FLUX.1-schnell/
and will be found automatically by the server on next startup.
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "src" / "researcher" / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FLUX_MODEL_ID = os.getenv("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

print(f"Downloading {FLUX_MODEL_ID} into {MODELS_DIR} ...")
print("This is ~33 GB — go test the tutor :)\n")

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    FLUX_MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    cache_dir=str(MODELS_DIR),
)

print(f"\nDone. Model cached in {MODELS_DIR}")
print("You can now start the server — it will load from local cache.")
