"""TTS (text-to-speech) with multilingual voice switching.

Provides:
- ``_detect_lang``        – detect language of a text segment
- ``_split_multilingual`` – split text into (lang, segment) pairs
- ``_pick_voice_for_lang``– choose the best edge-tts voice for a language
- ``register_tts_routes`` – attach /tts/* endpoints to a FastAPI app
"""

import io
import re
import logging

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice cache
# ---------------------------------------------------------------------------

_tts_voices_cache: list[dict] | None = None


async def _ensure_tts_voices() -> list[dict]:
    """Return cached voice list, fetching from edge-tts if needed."""
    global _tts_voices_cache
    if _tts_voices_cache is None:
        import edge_tts
        voices = await edge_tts.list_voices()
        _tts_voices_cache = [
            {"name": v["ShortName"], "gender": v.get("Gender", ""), "locale": v.get("Locale", "")}
            for v in voices
        ]
    return _tts_voices_cache


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

def _detect_lang(text: str) -> str:
    """Detect the language of a text segment, defaulting to 'en'."""
    from langdetect import detect, LangDetectException
    _FALSE_POS = {'no', 'so', 'sw', 'tl', 'cy', 'af', 'da'}
    try:
        lang = detect(text)
        if lang in _FALSE_POS and len(text.split()) < 6:
            return "en"
        return lang
    except LangDetectException:
        return "en"


# Regex to find quoted or emphasised foreign phrases in LLM output.
# Matches: "phrase", 'phrase', «phrase», *phrase*, **phrase**, _phrase_
# All patterns are constrained to a single line (no newline crossing)
# and the single-quote pattern requires a word boundary to avoid contractions.
_FOREIGN_PHRASE_RE = re.compile(
    r'["\u201c\u201d«]([^"\u201c\u201d»\n]{2,})["\u201c\u201d»]'
    r"|(?<![a-zA-Z])'([^'\n]{2,})'(?![a-zA-Z])"
    r'|\*{1,2}([^*\n]{2,})\*{1,2}'
    r'|_([^_\n]{2,})_'
)


def _split_multilingual(text: str, base_lang: str = "en") -> list[tuple[str, str]]:
    """Split *text* into ``(lang, segment)`` pairs so that each segment can be
    spoken with the correct TTS voice.

    Strategy (designed to **minimise** fragmentation):

    1.  Detect the dominant language of the **whole text**.  If it differs from
        *base_lang* and no sentence detects as base_lang, return the entire
        text as one foreign block.
    2.  Otherwise, build segments around quoted/emphasised foreign phrases
        (kept as foreign) and classify the **gaps** between them by detecting
        each gap as a whole.  Consecutive same-language pieces are merged.
    """
    from langdetect import detect, LangDetectException

    # ── tunables ──────────────────────────────────────────────────
    _FALSE_LANGS = {'no', 'so', 'sw', 'tl', 'cy', 'af', 'da'}
    _TRUSTED_LANGS = {
        'fr', 'de', 'es', 'it', 'pt', 'ro', 'ru', 'zh-cn', 'zh-tw',
        'ja', 'ko', 'ar', 'nl', 'pl', 'uk', 'el', 'tr', 'sv', 'cs', 'hu',
    }

    def _safe_detect(s: str) -> str:
        try:
            lang = detect(s)
            if lang in _FALSE_LANGS and len(s.split()) < 8:
                return base_lang
            return lang
        except LangDetectException:
            return base_lang

    def _classify(s: str) -> str:
        """Return the language for a text chunk, biased toward base_lang for
        short or ambiguous text."""
        stripped = s.strip()
        if not stripped:
            return base_lang
        wc = len(stripped.split())
        lang = _safe_detect(stripped)
        if lang == base_lang or lang in _FALSE_LANGS:
            return base_lang
        if lang in _TRUSTED_LANGS and wc >= 3:
            return lang
        if wc >= 4:
            return lang
        return base_lang

    # ── Step 1: whole-text detection ──────────────────────────────
    overall = _safe_detect(text)
    if overall != base_lang and overall in _TRUSTED_LANGS:
        # Quick check: split into sentences and see if any is clearly base_lang
        _quick_bounds = list(re.finditer(r'(?<=[.!?])\s+', text))
        _q_starts = [0] + [m.end() for m in _quick_bounds]
        _q_ends = [m.start() for m in _quick_bounds] + [len(text)]
        _q_sents = [text[s:e].strip() for s, e in zip(_q_starts, _q_ends)
                    if text[s:e].strip()]
        has_base = any(
            _safe_detect(s) == base_lang and len(s.split()) >= 3
            for s in _q_sents
        )
        if not has_base:
            return [(overall, text)]

    # ── Step 2: build pieces around quoted/emphasised foreign spans ──
    foreign_spans: list[tuple[int, int, str]] = []  # (start, end, lang)
    for m in _FOREIGN_PHRASE_RE.finditer(text):
        phrase = next(g for g in m.groups() if g is not None)
        lang = _safe_detect(phrase)
        if lang != base_lang and lang not in _FALSE_LANGS:
            foreign_spans.append((m.start(), m.end(), lang))

    # Sort and de-overlap
    foreign_spans.sort(key=lambda x: x[0])
    deduped: list[tuple[int, int, str]] = []
    for span in foreign_spans:
        if deduped and span[0] < deduped[-1][1]:
            continue
        deduped.append(span)
    foreign_spans = deduped

    def _classify_gap(gap_text: str) -> list[tuple[str, str]]:
        """Split a gap (text between foreign spans) into sentences and
        classify each one."""
        sb = list(re.finditer(r'(?<=[.!?])\s+|(?<=[:;])\s+', gap_text))
        ss = [0] + [m.end() for m in sb]
        se = [m.start() for m in sb] + [len(gap_text)]
        pieces = []
        for s, e in zip(ss, se):
            chunk = gap_text[s:e].strip()
            if chunk:
                pieces.append((_classify(chunk), chunk))
        return pieces or [(base_lang, gap_text)]

    # Build raw pieces: gap, foreign, gap, foreign, …, gap
    # Each piece: (lang, text, confirmed) — confirmed=True for quoted spans.
    raw: list[tuple[str, str, bool]] = []
    pos = 0
    for fs, fe, fl in foreign_spans:
        if fs > pos:
            gap = text[pos:fs].strip()
            if gap:
                raw.extend((l, t, False) for l, t in _classify_gap(gap))
        raw.append((fl, text[fs:fe].strip(), True))
        pos = fe
    if pos < len(text):
        tail = text[pos:].strip()
        if tail:
            raw.extend((l, t, False) for l, t in _classify_gap(tail))

    # If no foreign spans were found, classify sentence-by-sentence
    if not foreign_spans:
        raw = [(l, t, False) for l, t in _classify_gap(text)]

    if not raw:
        return [(base_lang, text)]

    # ── Post-process: fix isolated misdetections ──────────────────
    # Only fix *unconfirmed* (gap-classified) pieces that are sandwiched
    # between two pieces of the same language.  Never override confirmed
    # foreign quotes.
    for i in range(1, len(raw) - 1):
        lang_i, text_i, confirmed_i = raw[i]
        if confirmed_i:
            continue  # never override a quoted foreign phrase
        prev_lang = raw[i - 1][0]
        next_lang = raw[i + 1][0]
        if prev_lang == next_lang and lang_i != prev_lang:
            # Don't absorb base_lang gaps between confirmed foreign quotes;
            # these are usually legitimate English explanations in LLM output.
            if lang_i == base_lang and (raw[i - 1][2] or raw[i + 1][2]):
                continue
            raw[i] = (prev_lang, text_i, False)

    # ── Merge consecutive same-language pieces ────────────────────
    merged: list[tuple[str, str]] = [(raw[0][0], raw[0][1])]
    for lang, seg, _ in raw[1:]:
        if lang == merged[-1][0]:
            merged[-1] = (lang, merged[-1][1] + " " + seg)
        else:
            merged.append((lang, seg))

    # If everything ended up as base_lang, return the whole text as one piece
    if all(lang == base_lang for lang, _ in merged):
        return [(base_lang, text)]

    return merged


def _pick_voice_for_lang(lang: str, base_voice: str, voices: list[dict]) -> str:
    """Pick the best edge-tts voice for a detected language, preserving the gender
    of the user's selected base voice."""
    base_gender = "Female"
    for v in voices:
        if v["name"] == base_voice:
            base_gender = v["gender"]
            break

    lang_prefix = lang.lower()[:2]

    # If the base voice already matches the language, keep it
    for v in voices:
        if v["name"] == base_voice and v["locale"].lower().startswith(lang_prefix):
            return base_voice

    candidates = [v for v in voices if v["locale"].lower().startswith(lang_prefix)]
    if not candidates:
        return base_voice  # no voice for this language; fall back

    # Prefer same gender, then Neural voices (higher quality)
    gender_match = [v for v in candidates if v["gender"] == base_gender]
    pool = gender_match or candidates
    neural = [v for v in pool if "Neural" in v["name"] and "Multilingual" not in v["name"]]
    return (neural or pool)[0]["name"]


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-AriaNeural"


def register_tts_routes(app):
    """Attach /tts/voices and /tts/speak endpoints to the FastAPI *app*."""

    @app.get("/tts/voices")
    async def tts_voices():
        """Return available edge-tts voices (cached after first call)."""
        return await _ensure_tts_voices()

    @app.post("/tts/speak")
    async def tts_speak(req: TTSRequest):
        """Generate speech audio from text using edge-tts.

        Automatically detects language per sentence/phrase and switches to a
        matching voice so foreign words are pronounced correctly.
        """
        import edge_tts

        clean = req.text.strip()
        if not clean:
            raise HTTPException(status_code=400, detail="Empty text")

        voices = await _ensure_tts_voices()

        # Determine the base language from the user's selected voice
        base_lang = "en"
        for v in voices:
            if v["name"] == req.voice:
                base_lang = v["locale"][:2].lower()
                break

        segments = _split_multilingual(clean, base_lang)
        logger.warning("[TTS] req.voice=%s base_lang=%s segments=%d text=%.80s",
                       req.voice, base_lang, len(segments), clean)

        buf = io.BytesIO()
        for lang, text in segments:
            voice = _pick_voice_for_lang(lang, req.voice, voices)
            logger.warning("[TTS] segment lang=%s voice=%s text=%.60s", lang, voice, text)
            communicate = edge_tts.Communicate(text, voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])

        if buf.tell() == 0:
            raise HTTPException(status_code=500, detail="TTS produced no audio")

        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg")
