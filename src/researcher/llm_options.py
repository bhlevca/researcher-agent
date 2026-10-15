"""Ollama request options, in one place.

The three crews each built this dict themselves, and each got the same two
things wrong.

**Thinking.** The switch they used — ``chat_template_kwargs:
{"enable_thinking": ...}`` — is silently ignored by Ollama. Measured against
qwen3.5:9b with ``num_predict: 24`` and a prompt of "Say exactly: READY":

    thinking left on (default)            ->  ''        (empty)
    chat_template_kwargs enable_thinking  ->  ''        (ignored)
    think: False                          ->  'READY'

So thinking was on for every thinking-capable model regardless of intent. The
model spends its whole ``num_predict`` budget in the thinking channel, Ollama
returns that separately as ``thinking``, and ``response`` comes back empty.
What that looks like from the outside is an agent that writes a plan and stops.
gemma4:26b did exactly that — 2031 characters of thinking, zero of response,
``done_reason: length`` — and with thinking off the same prompt produced a
correct ``Action:`` line.

**Layer placement.** ``num_gpu: 99`` forced every layer onto the card. On a
15.5 GiB GPU, gemma4:26b then asks for a 17 GB buffer and either fails with
``cudaMalloc failed: out of memory`` or lands in a lopsided split — which is
what ``ollama ps`` was reporting as ``21%/79% CPU/GPU`` with only 1.6 GB
resident. Letting Ollama fit the model is both faster and the only thing that
works for models larger than the card.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Models that reason in a separate channel. Ollama turns that on by default,
# and it has to be turned off explicitly for any task whose output we need to
# read — which is all of them except conversational tutoring.
_THINKING_MODELS = (
    "qwen3",
    "deepseek-r1",
    "gemma4",
    "gpt-oss",
    "phi4-reasoning",
    "magistral",
)


def is_thinking_model(model: str) -> bool:
    """True when *model* reasons in a channel Ollama reports separately."""
    return any(tag in (model or "").lower() for tag in _THINKING_MODELS)


def ollama_options(params: dict | None = None, **overrides) -> dict:
    """The ``options`` block for an Ollama request.

    ``num_gpu`` is deliberately absent: Ollama decides how many layers fit.
    Pass ``num_gpu`` explicitly only to override that, and only for a model
    known to fit.
    """
    params = params or {}
    opts = {
        "num_ctx": int(params.get("num_ctx", 32768)),
        "num_predict": int(params.get("num_predict", 4096)),
        "top_k": int(params.get("top_k", 40)),
        "top_p": float(params.get("top_p", 0.9)),
        "min_p": float(params.get("min_p", 0.0)),
        "repeat_penalty": float(params.get("repeat_penalty", 1.1)),
    }
    seed = int(params.get("seed", 0))
    if seed > 0:
        opts["seed"] = seed
    opts.update(overrides)
    return opts


def extra_body(model: str, params: dict | None = None, *,
               thinking: bool = False, **option_overrides) -> dict:
    """The full ``extra_body`` for a CrewAI ``LLM`` talking to Ollama.

    *thinking* defaults to False because every crew here needs tokens it can
    read: the ReAct loop needs an ``Action:`` line, the composer needs ABC
    notation, the sheet writers need Markdown. Only conversational tutoring
    wants the thinking channel, and it asks for it.
    """
    body: dict = {"options": ollama_options(params, **option_overrides)}
    if is_thinking_model(model):
        # Ollama's own switch. chat_template_kwargs does not work.
        body["think"] = bool(thinking)
        if thinking:
            logger.debug("[llm] thinking enabled for %s", model)
    return body
