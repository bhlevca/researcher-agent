"""Whisper speech-to-text endpoint."""

import asyncio
import tempfile
import threading
import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile, File, Request

from researcher.auth import get_current_user

logger = logging.getLogger(__name__)

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                import whisper

                logger.info("Loading Whisper base model (CPU)…")
                _whisper_model = whisper.load_model("base", device="cpu")
                logger.info("Whisper model ready.")
    return _whisper_model


def register_speech_routes(app):
    """Attach /transcribe endpoint to the FastAPI app."""

    @app.post("/transcribe")
    async def transcribe(
        request: Request, file: UploadFile = File(...), language: str = "en"
    ):
        """Transcribe audio using Whisper (CPU). Accepts any ffmpeg-supported audio format."""
        await get_current_user(request)
        lang = language.strip().lower()[:2] if language else "en"
        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            loop = asyncio.get_running_loop()

            def _transcribe():
                model = _get_whisper()
                return model.transcribe(tmp_path, language=lang)

            result = await loop.run_in_executor(None, _transcribe)
            return {"text": result.get("text", "").strip()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
