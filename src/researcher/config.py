"""Shared constants, regex patterns, Pydantic models, and utility classes.

Used across main.py and route modules.
"""

import os
import re
import queue
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
STATIC_DIR = Path(__file__).parent / "static"
DB_PATH = Path(__file__).parent / "data" / "sessions.db"
GENERATED_DIR = STATIC_DIR / "generated"

# Ensure directories exist
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Strip ANSI escape codes and box-drawing decorations from captured output
_ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")
_box_re = re.compile(r"[╭╮╰╯│─┌┐└┘├┤┬┴┼]+")
_think_re = re.compile(r"<think>[\s\S]*?</think>\s*")
_think_open_re = re.compile(r"</?think>")

# Session ID validation
_SESSION_ID_RE = re.compile(r"^[a-f0-9]{8}$")


def _clean_line(text: str) -> str:
    """Clean a single line of stdout output."""
    text = _ansi_re.sub("", text)
    text = _box_re.sub("", text)
    text = _think_open_re.sub("", text)
    return text.strip()


class _QueueWriter:
    """Captures writes to stdout and puts cleaned lines into a queue."""

    encoding = "utf-8"

    def __init__(self, q: queue.Queue):
        self._q = q
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            cleaned = _clean_line(line)
            if cleaned:
                self._q.put(cleaned)

    def flush(self):
        if self._buf:
            cleaned = _clean_line(self._buf)
            if cleaned:
                self._q.put(cleaned)
            self._buf = ""

    def isatty(self):
        return False

    def fileno(self):
        raise OSError("not a real file")


# --- Pydantic request models ---


class ChatRequest(BaseModel):
    message: str
    history: list = []
    file_ids: list[str] = []


class ModelRequest(BaseModel):
    model: str


class SessionSaveRequest(BaseModel):
    name: str
    messages: list


class AskRequest(BaseModel):
    topic: str
    file_ids: list[str] = []


class ContinueRequest(BaseModel):
    original_query: str
    partial_response: str
    file_ids: list[str] = []


def _unload_ollama_model():
    """Send keep_alive=0 to Ollama to free VRAM after crew completes."""
    import requests as _req

    try:
        model = os.getenv("MODEL", "qwen3.5:9b").replace("ollama/", "")
        _req.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "keep_alive": 0, "prompt": ""},
            timeout=10,
        )
        logger.info("[VRAM] Unloaded Ollama model %s after crew completion", model)
    except Exception as e:
        logger.warning("[VRAM] Failed to unload Ollama model: %s", e)


def _maybe_unload_ollama():
    """Unload Ollama only if image generation used GPU this request (non-blocking)."""
    from researcher.image import _image_was_generated
    import researcher.image as _image_mod

    if _image_was_generated:
        _image_mod._image_was_generated = False  # reset for next request
        import threading

        threading.Thread(target=_unload_ollama_model, daemon=True).start()
        logger.info("[VRAM] Scheduling Ollama unload (image gen used GPU)")
    else:
        logger.debug("[VRAM] Skipping Ollama unload (no image gen this request)")
