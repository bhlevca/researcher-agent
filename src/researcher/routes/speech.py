"""Whisper speech-to-text endpoint."""

import asyncio
import re
import tempfile
import threading
import logging
import unicodedata
from pathlib import Path

from fastapi import HTTPException, UploadFile, File, Request

from researcher.auth import get_current_user

logger = logging.getLogger(__name__)

_whisper_model = None
_whisper_lock = threading.Lock()

# Unicode script ranges for major language families
_SCRIPT_RANGES = {
    "latin":    re.compile(r'[\u0000-\u024F\u1E00-\u1EFF]'),
    "cyrillic": re.compile(r'[\u0400-\u04FF]'),
    "cjk":      re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u30FF\u31F0-\u31FF\uFF00-\uFFEF]'),
    "arabic":   re.compile(r'[\u0600-\u06FF\u0750-\u077F]'),
    "devanagari": re.compile(r'[\u0900-\u097F]'),
    "hangul":   re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF]'),
}

# Which scripts are expected for each language code
_LANG_SCRIPTS = {
    "en": {"latin"}, "fr": {"latin"}, "de": {"latin"}, "es": {"latin"},
    "it": {"latin"}, "pt": {"latin"}, "nl": {"latin"}, "sv": {"latin"},
    "pl": {"latin"}, "ro": {"latin"}, "tr": {"latin"}, "el": {"latin"},
    "ru": {"cyrillic"},
    "zh": {"cjk"}, "ja": {"cjk", "latin"}, "ko": {"hangul", "cjk", "latin"},
    "ar": {"arabic"},
    "hi": {"devanagari", "latin"},
}


def _dominant_script(text: str) -> str | None:
    """Return the dominant non-latin script in text, or 'latin' if mostly latin."""
    counts = {name: len(pat.findall(text)) for name, pat in _SCRIPT_RANGES.items()}
    total = sum(counts.values())
    if total == 0:
        return None
    # Return script with most characters
    return max(counts, key=counts.get)


def _has_script_mismatch(text: str, lang: str) -> bool:
    """Return True if text contains characters from unexpected scripts for the language."""
    expected = _LANG_SCRIPTS.get(lang)
    if not expected:
        return False
    dominant = _dominant_script(text)
    if dominant is None:
        return False
    return dominant not in expected


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
        """Transcribe audio using Whisper (CPU). Accepts any ffmpeg-supported audio format.

        Pass language=auto to let Whisper auto-detect the spoken language.
        """
        await get_current_user(request)
        raw_lang = (language or "").strip().lower()
        # "auto" means let Whisper detect; otherwise take first 2 chars
        lang = None if raw_lang in ("auto", "") else raw_lang[:2]
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
                opts = dict(
                    condition_on_previous_text=False,  # prevent hallucination loops
                    no_speech_threshold=0.6,           # skip silent segments
                    logprob_threshold=-1.0,            # Whisper default
                    compression_ratio_threshold=2.4,   # filter repetitive hallucinations
                )
                if lang:
                    opts["language"] = lang
                # else: omit language → Whisper auto-detects

                result = model.transcribe(tmp_path, **opts)
                detected_lang = result.get("language", lang or "en")

                # Filter segments
                segments = result.get("segments", [])
                filtered = []
                for s in segments:
                    # Skip segments with high no-speech probability
                    if s.get("no_speech_prob", 0) >= 0.6:
                        continue
                    # Skip segments with very low confidence
                    if s.get("avg_logprob", 0) < -1.0:
                        continue
                    # Skip segments with script mismatch (e.g. CJK in French)
                    if _has_script_mismatch(s["text"], detected_lang):
                        logger.debug("Filtered script-mismatch segment: %s", s["text"][:60])
                        continue
                    filtered.append(s["text"])

                return " ".join(filtered).strip() if filtered else ""

            text = await loop.run_in_executor(None, _transcribe)
            return {"text": text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
