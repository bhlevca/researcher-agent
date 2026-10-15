"""Direct access to a music-specialist model.

ChatMusician is LLaMA2 continually pre-trained on ABC notation. Measured against
the general chat model on the same task it wrote 7 distinct rhythms against 2,
and used 11 of 12 pitch classes against 7 — it has actually heard how jazz uses
blue notes and approach tones, rather than being told to in a prompt.

It is called directly here rather than through CrewAI, for the same reason the
OCR specialist is: it is not an instruction-following chat model. It has no
tools, needs no planning loop, and wrapping it in a ReAct agent would add
minutes of overhead and a format it does not speak.

Two quirks, both discovered by running it:

* The Ollama template is a bare ``{{ .Prompt }}`` with no chat format, so the
  ``Human:``/``Assistant:`` framing has to be supplied explicitly. Without it the
  model behaves as a raw completion engine and returns headerless fragments in
  the wrong key.
* It repeats. An eight-bar request comes back as the same eight bars two or
  three times, so the output must be truncated to the length asked for.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

logger = logging.getLogger(__name__)

# Independent of MODEL, exactly as OCR_MODEL is. Empty disables the specialist
# and falls back to the general agent path.
COMPOSER_MODEL = os.getenv("COMPOSER_MODEL", "").strip()
COMPOSER_TIMEOUT = int(os.getenv("COMPOSER_TIMEOUT", "300"))
_OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def is_available() -> tuple[bool, str]:
    """Report whether the configured specialist is usable.

    Like the image critic, a missing specialist degrades silently to "no
    improvement", so this exists to make that visible at startup.
    """
    if not COMPOSER_MODEL:
        return False, "COMPOSER_MODEL unset — composing with the general chat model"
    try:
        with urllib.request.urlopen(f"{_OLLAMA}/api/tags", timeout=10) as r:
            names = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
    except Exception as e:
        return False, f"could not reach Ollama to verify COMPOSER_MODEL: {e}"
    if COMPOSER_MODEL in names or f"{COMPOSER_MODEL}:latest" in names:
        return True, f"composer specialist ready: {COMPOSER_MODEL}"
    return False, (
        f"COMPOSER_MODEL '{COMPOSER_MODEL}' is not pulled — composing will fall "
        f"back to the general model. Run: ollama pull {COMPOSER_MODEL}"
    )


def _abc_key(key_signature: str) -> str:
    """"Bb major" -> "Bb"; "A minor" -> "Am"."""
    parts = (key_signature or "C major").strip().split()
    tonic = parts[0] if parts else "C"
    mode = parts[1].lower() if len(parts) > 1 else "major"
    return tonic + {"minor": "m", "dorian": "Dor", "mixolydian": "Mix"}.get(mode, "")


def build_prompt(description: str, genre: str, key_signature: str,
                 time_signature: str, tempo: int, bars: int,
                 style_guide: str = "", motif: str = "") -> str:
    """Compose the instruction, in the format this model actually expects."""
    lines = [
        f"Develop a complete {bars}-bar {genre} melody in ABC notation.",
        f"Key: {key_signature}. Time signature: {time_signature}. Tempo: {tempo} BPM.",
    ]
    if description:
        lines.append(f"The music should be: {description}.")
    if motif:
        lines.append(f"Begin from this motif and develop it: {motif}")
    lines.append(
        "Include the full ABC header (X, T, M, L, K). Use chord symbols in "
        'double quotes above the melody. Write exactly '
        f"{bars} bars and then stop."
    )
    # Style guidance is kept short: this model already knows the idiom, and a
    # long lecture crowds out its own vocabulary.
    if style_guide:
        first = [ln.strip() for ln in style_guide.splitlines() if ln.strip()][:4]
        if first:
            lines.append("Style notes: " + " ".join(first))
    instruction = "\n".join(lines)
    return f"Human: {instruction}\n\nAssistant: "


_BAR_SPLIT = re.compile(r"\|")


def truncate_to_bars(abc: str, bars: int) -> str:
    """Cut the tune to *bars* bars.

    The model restates the whole phrase two or three times; keeping all of it
    would hand the user a piece several times longer than they asked for.
    """
    if not abc or bars <= 0:
        return abc
    header, body = [], []
    for line in abc.splitlines():
        (header if re.match(r"^[A-Za-z]:", line.strip()) else body).append(line)

    music = "\n".join(body)
    # Count barlines, keeping repeat marks attached to the bar they close.
    pieces = [p for p in _BAR_SPLIT.split(music)]
    kept, count = [], 0
    for piece in pieces:
        if count >= bars:
            break
        kept.append(piece)
        if piece.strip():
            count += 1
    out = "|".join(kept).rstrip()
    if not out.endswith("|"):
        out += " |"
    return "\n".join(header + [out])


def compose(description: str, genre: str, key_signature: str,
            time_signature: str, tempo: int, bars: int = 16,
            style_guide: str = "", motif: str = "",
            temperature: float = 0.6) -> str | None:
    """Ask the specialist for a tune. Returns ABC, or None if unavailable."""
    if not COMPOSER_MODEL:
        return None
    prompt = build_prompt(description, genre, key_signature, time_signature,
                          tempo, bars, style_guide, motif)
    body = {
        "model": COMPOSER_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            # Generous: the model writes past the requested length and is then
            # truncated, which is cheaper than a truncated tune with no cadence.
            "num_predict": max(600, bars * 60),
        },
    }
    # Ollama returns an empty response when it is mid-swap between models —
    # this box holds one model at a time, so composing right after a chat turn
    # can land exactly there. One retry costs a few seconds and turns a failed
    # composition into a successful one.
    text = ""
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(
                f"{_OLLAMA}/api/generate",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=COMPOSER_TIMEOUT) as r:
                text = (json.loads(r.read()).get("response") or "").strip()
        except Exception as e:
            logger.warning("[Composer] specialist call failed (try %d): %s", attempt, e)
            text = ""
        if text:
            break
        if attempt == 1:
            logger.info("[Composer] empty response, retrying once (model swap?)")

    if not text:
        logger.warning("[Composer] specialist returned nothing after 2 tries")
        return None

    # Strip any code fence, then require a real ABC header.
    m = re.search(r"```(?:abc)?\s*(.*?)```", text, re.DOTALL)
    abc = (m.group(1) if m else text).strip()
    if not re.search(r"^K:", abc, re.M):
        # It answered without a key line; supply the header it was asked for.
        head = (f"X:1\nT:{(description or genre).title()[:40]}\n"
                f"M:{time_signature}\nL:1/8\nQ:1/4={tempo}\n"
                f"K:{_abc_key(key_signature)}\n")
        abc = head + abc
    if not re.search(r"^X:", abc, re.M):
        abc = "X:1\n" + abc
    return truncate_to_bars(abc, bars)


# ---------------------------------------------------------------------------
# Cleaning what the model actually emits
# ---------------------------------------------------------------------------

_INLINE_FIELD = re.compile(r"\[[MKLQVmklqv]:[^\]]*\]")
_BROKEN_ONTO_REST = re.compile(r"[<>]\s*(z)", re.IGNORECASE)
_MULTI_VOICE_LINE = re.compile(r"(\[V:\d+\])(?=.*\[V:\d+\])")


def sanitize_abc(abc: str) -> tuple[str, list[str]]:
    """Repair the notation errors this model reliably makes.

    music21 parses leniently and accepted all of these, so validation passed
    while the browser's stricter renderer refused the tune. The fixes are for
    real faults seen in output, not hypothetical ones:

    * ``[M:3/4]`` dropped mid-bar — a metre change halfway through a line,
      which leaves the bar the wrong length either side of it
    * ``C4 >z`` — a broken-rhythm marker pointing at a rest, which has no
      meaning: there is nothing to lengthen or shorten
    * unbalanced slur parentheses, e.g. ``(D (E F) G)``
    * two ``[V:n]`` voice switches on one line

    Returns ``(cleaned, notes)`` where *notes* describes what was changed.
    """
    if not abc:
        return abc, []

    notes: list[str] = []
    out_lines: list[str] = []
    for line in abc.splitlines():
        stripped = line.strip()
        # Header fields (X:, T:, M:, K: at line start) are left alone.
        if re.match(r"^[A-Za-z]:", stripped):
            out_lines.append(line)
            continue

        original = line

        # Inline metre/key changes mid-line: drop them and keep the header's.
        if _INLINE_FIELD.search(line):
            # Keep a leading voice marker, drop everything else inline.
            leading = ""
            m = re.match(r"^\s*(\[V:\d+\])", line)
            if m:
                leading = m.group(1) + " "
                line = line[m.end():]
            line = _INLINE_FIELD.sub("", line)
            line = leading + line
            notes.append("removed inline metre/key change")

        # Broken rhythm pointing at a rest has no meaning.
        if _BROKEN_ONTO_REST.search(line):
            line = _BROKEN_ONTO_REST.sub(r"\1", line)
            notes.append("removed broken-rhythm marker on a rest")

        # Unbalanced slurs confuse strict parsers.
        opens, closes = line.count("("), line.count(")")
        if opens > closes:
            line += ")" * (opens - closes)
            notes.append("closed unbalanced slur")
        elif closes > opens:
            line = line.replace(")", "", closes - opens)
            notes.append("removed unmatched slur close")

        # Collapse stray leading voice markers: one per line is enough, and a
        # second mid-line silently reassigns the rest of the bar.
        if len(re.findall(r"\[V:\d+\]", line)) > 1:
            first = re.search(r"\[V:\d+\]", line).group(0)
            line = first + " " + re.sub(r"\[V:\d+\]", "", line[len(first):])
            notes.append("merged multiple voice switches on one line")

        line = re.sub(r"\s{2,}", " ", line).rstrip()
        if line.strip() or not original.strip():
            out_lines.append(line)

    return "\n".join(out_lines), sorted(set(notes))


def extract_motif(abc: str, max_notes: int = 6) -> list[tuple[str, float]]:
    """Take the opening idea from a tune, for use as long-form material.

    The specialist's real value is idiomatic vocabulary. Rather than ask it for
    eighty bars it cannot hold together, take its opening gesture and let the
    realisation engine develop it across a full form.
    """
    from music21 import converter, note as m21note

    try:
        score = converter.parse(abc, format="abc")
    except Exception:
        return []
    out: list[tuple[str, float]] = []
    for n in score.recurse().getElementsByClass(m21note.Note):
        out.append((n.nameWithOctave, float(n.quarterLength) or 1.0))
        if len(out) >= max_notes:
            break
    return out


def compose_long(description: str, genre: str, key_signature: str,
                 time_signature: str, tempo: int, minutes: float = 2.5,
                 style_guide: str = "",
                 instruments: list[str] | None = None):
    """Compose a piece of roughly *minutes*, combining both halves.

    The specialist writes a short seed in its own idiom; the realisation engine
    develops that seed across a recurring form with genre-appropriate extended
    harmony. Returns ``(score, info)`` — a music21 Score and a dict describing
    what happened, or ``(None, info)`` on failure.
    """
    from researcher.composer import realize

    info: dict = {"seed_bars": 8, "used_specialist": False}

    seed = compose(description, genre, key_signature, time_signature, tempo,
                   bars=8, style_guide=style_guide)
    motif: list[tuple[str, float]] = []
    if seed:
        seed, repairs = sanitize_abc(seed)
        info["repairs"] = repairs
        motif = extract_motif(seed)
        info["used_specialist"] = bool(motif)
        info["seed_abc"] = seed

    plan = realize.build_long_plan(genre, key_signature, time_signature, tempo,
                                   minutes, motif=motif or None)
    score = realize.assemble(plan, instruments=instruments)
    info["instruments"] = instruments or ["Piano"]
    info["bars"] = plan.total_bars()
    info["seconds"] = realize.estimate_duration(plan)
    info["sections"] = len(plan.form)
    return score, info
