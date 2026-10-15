"""Turning model failures into something a person can act on.

CrewAI reports an exhausted output budget as "Invalid response from LLM call -
None or empty", which says nothing about the cause and sent us chasing Ollama,
the GPU and the network before the real answer turned up: a reasoning model had
spent its entire num_predict budget thinking and never reached the answer.

These helpers name the likely cause and say what to change.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# What the underlying error text looks like, and what it actually means.
_PATTERNS: list[tuple[tuple[str, ...], str]] = [
    (("invalid response from llm call", "none or empty"),
     "The model returned nothing. This almost always means it ran out of "
     "output tokens — a reasoning model can spend the whole num_predict "
     "budget thinking and never reach the answer. Raise num_predict in "
     "Params, or pick a model that does not reason for structured output."),
    (("context length", "context window", "too many tokens", "maximum context"),
     "The prompt exceeded the model's context window. Lower num_ctx, shorten "
     "the topic, or ask for fewer questions."),
    (("action input is not a valid", "not a valid key, value dictionary"),
     "The model produced a malformed tool call and could not recover. This "
     "task does not need tools; a smaller or better instruction-following "
     "model usually fixes it."),
    (("connection refused", "connectionerror", "max retries"),
     "Ollama is not reachable on localhost:11434. Check that it is running."),
    (("out of memory", "cuda", "vram"),
     "The GPU ran out of memory. The model is probably too large for the card "
     "once the KV cache is allocated — try a smaller model or a lower num_ctx."),
    (("timed out", "timeout"),
     "The model did not answer in time. A model that does not fit in VRAM runs "
     "partly on CPU and can take minutes per call; check `ollama ps` for a "
     "CPU/GPU split."),
]


def explain(exc: BaseException | str) -> str:
    """A diagnosis for *exc*, or a plain restatement if none fits."""
    text = str(exc).lower()
    for needles, message in _PATTERNS:
        if any(n in text for n in needles):
            return message
    return str(exc)[:300]


def describe_for_user(exc: BaseException, what: str = "generation") -> str:
    """The message an endpoint should return, logged with the detail."""
    diagnosis = explain(exc)
    logger.warning("[tutor] %s failed: %s | %s", what, type(exc).__name__,
                   str(exc)[:300])
    return f"{what.capitalize()} failed. {diagnosis}"
