"""Whisper speech-to-text and pronunciation scoring endpoints.

Transcription runs on faster-whisper (CTranslate2).  Compared to openai-whisper
it is roughly 4x faster at the same model size and, importantly, pulls in no
numba/triton, so a NumPy major bump cannot take speech recognition offline.
"""

import os
import asyncio
import re
import tempfile
import threading
import logging
import unicodedata
from pathlib import Path

from fastapi import HTTPException, UploadFile, File, Form, Request

from researcher.auth import get_current_user


def _wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: 0.0 = perfect, 1.0+ = completely wrong.

    Uses dynamic programming edit distance on word sequences.
    Normalises both strings: lowercase, strip punctuation, collapse spaces.
    """
    def _norm(s: str) -> list[str]:
        # NFD decompose so accented chars split into base + combining mark,
        # then strip combining marks → accent-insensitive comparison
        s = unicodedata.normalize("NFD", s.lower())
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
        return s.split()

    ref = _norm(reference)
    hyp = _norm(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    # Wagner-Fischer DP
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[m] / len(ref)


def _pronunciation_score(reference: str, hypothesis: str) -> float:
    """Convert WER to a 0.0–1.0 pronunciation score."""
    if not hypothesis.strip():
        return 0.0
    wer = _wer(reference, hypothesis)
    return max(0.0, round(1.0 - wer, 3))

logger = logging.getLogger(__name__)

_whisper_model = None
_whisper_lock = threading.Lock()

# Silero VAD pre-filter; disable with WHISPER_VAD=0 if it clips quiet speech.
_WHISPER_VAD = os.getenv("WHISPER_VAD", "1") == "1"

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


def get_whisper_params() -> dict:
    """Current speech-to-text settings, for the Params panel."""
    device = os.getenv("WHISPER_DEVICE", "cpu")
    return {
        "whisper_model": os.getenv("WHISPER_MODEL", "small"),
        "whisper_device": device,
        "whisper_compute_type": os.getenv(
            "WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16"
        ),
        "whisper_vad": _WHISPER_VAD,
    }


def update_whisper_params(params: dict) -> dict:
    """Update speech-to-text settings at runtime.

    Model, device and compute type are baked into the loaded WhisperModel, so
    changing any of them drops the cached instance; the next transcription
    reloads it. VAD is applied per call and needs no reload.
    """
    global _whisper_model, _WHISPER_VAD

    reload_needed = False
    if "whisper_model" in params:
        val = str(params["whisper_model"]).strip()
        if val in {"tiny", "base", "small", "medium", "large-v2", "large-v3"}:
            os.environ["WHISPER_MODEL"] = val
            reload_needed = True
    if "whisper_device" in params:
        val = str(params["whisper_device"]).strip().lower()
        if val in {"cpu", "cuda"}:
            os.environ["WHISPER_DEVICE"] = val
            # int8 is meaningless on cuda and float16 is wrong on cpu, so keep
            # the pair consistent unless the caller overrides it below.
            os.environ["WHISPER_COMPUTE_TYPE"] = "int8" if val == "cpu" else "float16"
            reload_needed = True
    if "whisper_compute_type" in params:
        val = str(params["whisper_compute_type"]).strip().lower()
        if val in {"int8", "int8_float16", "float16", "float32"}:
            os.environ["WHISPER_COMPUTE_TYPE"] = val
            reload_needed = True
    if "whisper_vad" in params:
        _WHISPER_VAD = bool(params["whisper_vad"])

    if reload_needed:
        with _whisper_lock:
            _whisper_model = None
        logger.info("[STT] Settings changed — model will reload on next request")
    return get_whisper_params()


class _FasterWhisperAdapter:
    """Presents faster-whisper's ``(segments, info)`` API as an openai-whisper dict.

    Both endpoints below filter Whisper's *segments* (by no-speech probability,
    confidence, and Unicode script) rather than trusting the joined text, so the
    adapter normalises to that same shape.  Keeping the dict contract here means
    the filtering logic — and its tests — stay independent of the STT backend.
    """

    # openai-whisper spelled this ``logprob_threshold``; faster-whisper uses
    # ``log_prob_threshold``.  Translated so callers keep one vocabulary.
    _RENAMED_OPTS = {"logprob_threshold": "log_prob_threshold"}

    def __init__(self, model):
        self._model = model

    def transcribe(self, path: str, **opts) -> dict:
        for old, new in self._RENAMED_OPTS.items():
            if old in opts:
                opts[new] = opts.pop(old)

        # Silero VAD drops silent regions before decoding.  This is a stronger
        # defence against the classic "Thank you for watching." hallucination on
        # near-silent mic clips than post-hoc no_speech_prob filtering alone.
        opts.setdefault("vad_filter", _WHISPER_VAD)

        segments, info = self._model.transcribe(path, **opts)
        # segments is a lazy generator — decoding happens as it is consumed.
        seg_dicts = [
            {
                "text": s.text,
                "no_speech_prob": s.no_speech_prob,
                "avg_logprob": s.avg_logprob,
            }
            for s in segments
        ]
        return {
            "text": "".join(s["text"] for s in seg_dicts).strip(),
            "language": info.language,
            "segments": seg_dicts,
        }


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel

                # "small" has dramatically better multilingual accuracy than "base"
                # (~244M params vs 74M). int8 on CPU keeps it off the GPU, which is
                # already contested by Ollama and the image pipelines.
                model_size = os.getenv("WHISPER_MODEL", "small")
                device = os.getenv("WHISPER_DEVICE", "cpu")
                compute_type = os.getenv(
                    "WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16"
                )
                logger.info(
                    "Loading faster-whisper %s (device=%s, compute_type=%s)…",
                    model_size, device, compute_type,
                )
                _whisper_model = _FasterWhisperAdapter(
                    WhisperModel(model_size, device=device, compute_type=compute_type)
                )
                logger.info("Whisper model ready.")
    return _whisper_model


def register_speech_routes(app):
    """Attach /transcribe endpoint to the FastAPI app."""

    @app.post("/transcribe")
    async def transcribe(
        request: Request, file: UploadFile = File(...), language: str = "en"
    ):
        """Transcribe audio with faster-whisper. Accepts any PyAV-decodable format.

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

    @app.post("/tutor/pronunciation/score")
    async def pronunciation_score(
        request: Request,
        file: UploadFile = File(...),
        expected: str = Form(""),
        language: str = Form("auto"),
    ):
        """Transcribe audio and score pronunciation against an expected sentence.

        Returns transcript, WER-based score (0–1), and a feedback string.
        """
        await get_current_user(request)
        if not expected.strip():
            raise HTTPException(status_code=400, detail="expected text is required")

        raw_lang = (language or "").strip().lower()
        lang = None if raw_lang in ("auto", "") else raw_lang[:2]
        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            loop = asyncio.get_running_loop()

            def _transcribe_for_score():
                model = _get_whisper()
                opts = dict(
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                )
                if lang:
                    opts["language"] = lang
                result = model.transcribe(tmp_path, **opts)
                segments = result.get("segments", [])
                parts = [
                    s["text"] for s in segments
                    if s.get("no_speech_prob", 0) < 0.6
                    and s.get("avg_logprob", 0) >= -1.0
                ]
                return " ".join(parts).strip()

            transcript = await loop.run_in_executor(None, _transcribe_for_score)
            score = _pronunciation_score(expected, transcript)

            if score >= 0.9:
                feedback = "Excellent pronunciation!"
            elif score >= 0.7:
                feedback = "Good — a few small differences from the target."
            elif score >= 0.4:
                feedback = "Getting there — try again, focusing on each word."
            else:
                feedback = "Needs practice — compare your recording to the target."

            return {
                "transcript": transcript,
                "expected": expected,
                "score": score,
                "feedback": feedback,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
