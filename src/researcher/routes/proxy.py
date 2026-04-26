"""OpenAI-compatible /v1 proxy.

Bridges external tools (Open WebUI, Claude Code, Continue.dev, OpenCode,
and anything that speaks the OpenAI API) to the local Ollama backend.

Configure your external tool with:
    OPENAI_BASE_URL = http://localhost:8000/v1
    OPENAI_API_KEY  = <value of PROXY_API_KEY env var, or any string if unset>

Endpoints:
    GET  /v1/models                — list locally available Ollama models
    GET  /v1/models/{model_id}     — single model record
    POST /v1/chat/completions      — streaming + non-streaming chat
    POST /v1/completions           — legacy text completions pass-through
    POST /v1/embeddings            — embeddings pass-through

Security:
    Set PROXY_API_KEY in the environment to require Bearer token auth on all
    /v1/* routes.  If the var is empty/unset, all requests are allowed
    (matching the default Ollama behaviour — trusted local network only).

Reasoning enhancement:
    For known thinking models (deepseek-r1, qwq, qwen3) the proxy injects a
    brief reasoning system message when none is already present, nudging the
    model to show its chain of thought before answering.
"""

import os
import json
import time
import logging
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from researcher.config import OLLAMA_BASE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# If set, every /v1/* request must carry  Authorization: Bearer <key>.
_PROXY_API_KEY: str = os.getenv("PROXY_API_KEY", "")

# Ollama OpenAI-compatible base
_OLLAMA_V1 = f"{OLLAMA_BASE}/v1"

# Families that benefit from explicit reasoning instructions
_THINKING_FAMILIES = ("deepseek-r1", "qwq", "qwen3", "qwq-32b")

_REASONING_SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. "
    "Think step-by-step inside <think>…</think> tags before writing your final answer. "
    "Keep your final answer concise and well-structured."
)

# ---------------------------------------------------------------------------
# Persona registry — mirrors `ollama launch <persona> --model <base>`
#
# Each entry defines the SYSTEM block that gets baked into the Ollama
# Modelfile when the user "launches" that persona on top of a local model.
# The resulting model is registered in Ollama as "<persona>-<base>" and can
# be used like any other model from that point on (no runtime injection).
# ---------------------------------------------------------------------------
PERSONAS: dict[str, dict] = {
    "claude": {
        "label": "Claude",
        "description": "Anthropic Claude style — thoughtful, nuanced, honest about uncertainty",
        "system": (
            "You are Claude, an AI assistant made by Anthropic. "
            "Your core traits: intellectually curious, warm but never sycophantic, "
            "direct and confident while remaining genuinely open to other views. "
            "Think carefully before answering. Acknowledge uncertainty honestly using "
            "phrases like 'I think', 'I'm not certain', or 'you may want to verify this'. "
            "Structure long answers with markdown headers or bullet points for clarity. "
            "Never open a response with flattery or filler like 'Great question!'. "
            "When reasoning through a hard problem, think step-by-step inside "
            "<think>…</think> tags before writing your final answer."
        ),
    },
    "gpt4": {
        "label": "GPT-4",
        "description": "OpenAI GPT-4 style — thorough, structured, formal",
        "system": (
            "You are GPT-4, a large language model trained by OpenAI. "
            "Always be thorough, structured, and precise. "
            "Use numbered steps for procedures and bullet points for lists. "
            "Bold key terms and important concepts. "
            "Show your reasoning step-by-step when solving problems. "
            "Be concise in casual exchanges but comprehensive when depth is needed. "
            "Cite caveats and limitations where relevant."
        ),
    },
    "gemini": {
        "label": "Gemini",
        "description": "Google Gemini style — factual, analytical, multi-perspective",
        "system": (
            "You are Gemini, an AI assistant made by Google DeepMind. "
            "Prioritise factual accuracy above all else; explicitly flag anything you "
            "are uncertain about. For multi-part questions, address each part in order "
            "with clear labels. Present multiple perspectives when a topic is nuanced. "
            "Be analytical and balanced. Use concise plain language; avoid padding."
        ),
    },
    "coder": {
        "label": "Coder",
        "description": "Expert software engineer — production-quality code, minimal prose",
        "system": (
            "You are an expert software engineer with deep knowledge across languages "
            "and frameworks. Always provide working, production-quality code. "
            "Use the most idiomatic approach for the language in question. "
            "Minimal prose before the code block; after the block note edge cases, "
            "performance considerations, and any follow-up steps the caller should take."
        ),
    },
}

router = APIRouter(prefix="/v1", tags=["OpenAI Proxy"])
anthropic_router = APIRouter(prefix="/anthropic", tags=["Anthropic Proxy"])


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> None:
    """Validate Bearer token if PROXY_API_KEY is configured."""
    if not _PROXY_API_KEY:
        return  # no key configured → open access (Ollama default)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth[len("Bearer "):]
    if token != _PROXY_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_thinking_model(model: str) -> bool:
    name = model.lower()
    return any(f in name for f in _THINKING_FAMILIES)


def _maybe_inject_reasoning(body: dict) -> dict:
    """For thinking models with no existing system message, prepend reasoning hint."""
    model = body.get("model", "")
    if not _is_thinking_model(model):
        return body
    messages = body.get("messages", [])
    has_system = any(m.get("role") == "system" for m in messages)
    if has_system:
        return body
    # Insert reasoning system message at the front
    body = dict(body)
    body["messages"] = [{"role": "system", "content": _REASONING_SYSTEM_PROMPT}] + messages
    return body


def _ollama_model_to_openai(m: dict) -> dict:
    """Convert an Ollama model record to the OpenAI /v1/models item format."""
    name = m.get("name", "")
    modified = m.get("modified_at", "")
    # OpenAI format uses integer epoch; Ollama gives ISO 8601
    try:
        from datetime import datetime
        created = int(datetime.fromisoformat(modified.replace("Z", "+00:00")).timestamp())
    except Exception:
        created = int(time.time())
    return {
        "id": name,
        "object": "model",
        "created": created,
        "owned_by": "ollama",
        "permission": [],
        "root": name,
        "parent": None,
    }


async def _stream_ollama(
    method: str,
    path: str,
    body: dict,
    headers: dict,
) -> AsyncIterator[bytes]:
    """Yield SSE chunks from Ollama, forwarding them verbatim."""
    url = f"{_OLLAMA_V1}{path}"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(method, url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                err = await resp.aread()
                yield b"data: " + json.dumps({"error": err.decode()}).encode() + b"\n\n"
                return
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

@router.get("/models")
async def list_models(request: Request) -> JSONResponse:
    """Return Ollama models in OpenAI /v1/models format."""
    _check_auth(request)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {exc}")

    models = [_ollama_model_to_openai(m) for m in resp.json().get("models", [])]
    return JSONResponse({"object": "list", "data": models})


# ---------------------------------------------------------------------------
# GET /v1/models/{model_id}
# ---------------------------------------------------------------------------

@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> JSONResponse:
    """Return a single model record."""
    _check_auth(request)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {exc}")

    all_models = resp.json().get("models", [])
    for m in all_models:
        if m.get("name") == model_id:
            return JSONResponse(_ollama_model_to_openai(m))

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions to Ollama with optional reasoning enhancement."""
    _check_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    body = _maybe_inject_reasoning(body)

    # Forward original Accept / Content-Type headers only (strip auth so
    # Ollama's own unauthenticated API is not broken by our Bearer token)
    fwd_headers = {"Content-Type": "application/json"}
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            _stream_ollama("POST", "/chat/completions", body, fwd_headers),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming — forward and return full response
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{_OLLAMA_V1}/chat/completions",
                json=body,
                headers=fwd_headers,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}")

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# POST /v1/completions  (legacy)
# ---------------------------------------------------------------------------

@router.post("/completions")
async def completions(request: Request):
    """Proxy legacy text completions to Ollama."""
    _check_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    fwd_headers = {"Content-Type": "application/json"}
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            _stream_ollama("POST", "/completions", body, fwd_headers),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{_OLLAMA_V1}/completions",
                json=body,
                headers=fwd_headers,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}")

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# POST /v1/embeddings
# ---------------------------------------------------------------------------

@router.post("/embeddings")
async def embeddings(request: Request):
    """Proxy embedding requests to Ollama."""
    _check_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    fwd_headers = {"Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{_OLLAMA_V1}/embeddings",
                json=body,
                headers=fwd_headers,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}")

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# GET /v1/personas  — list available persona definitions
# ---------------------------------------------------------------------------

@router.get("/personas")
async def list_personas(request: Request) -> JSONResponse:
    """Return the available persona presets."""
    _check_auth(request)
    return JSONResponse({
        "personas": [
            {"id": k, "label": v["label"], "description": v["description"]}
            for k, v in PERSONAS.items()
        ]
    })


# ---------------------------------------------------------------------------
# POST /v1/launch  — create a named Ollama model with persona baked in
#
# Body: { "persona": "claude", "model": "llama3.2:latest" }
# Equivalent to:
#   ollama launch claude --model llama3.2:latest
#
# Calls Ollama /api/create with a Modelfile:
#   FROM llama3.2:latest
#   SYSTEM "..."
#
# Returns: { "name": "claude-llama3.2", "persona": "claude", "base": "llama3.2:latest" }
# ---------------------------------------------------------------------------

@router.post("/launch")
async def launch_persona(request: Request) -> JSONResponse:
    """
    Bake a persona system prompt into a local Ollama model, creating a new
    named model in Ollama's registry.  Equivalent to `ollama launch <persona>
    --model <base>`.
    """
    _check_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    persona_id = body.get("persona", "").strip()
    base_model  = body.get("model", "").strip()

    if not persona_id:
        raise HTTPException(status_code=400, detail="'persona' is required")
    if not base_model:
        raise HTTPException(status_code=400, detail="'model' (base model) is required")
    if persona_id not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona '{persona_id}'. Available: {list(PERSONAS.keys())}"
        )

    persona = PERSONAS[persona_id]

    # Reject launching a persona on top of an already-launched persona model
    persona_prefixes = tuple(PERSONAS.keys())
    base_short = base_model.split("/")[-1]  # strip registry prefix
    if any(base_short.startswith(p + "-") for p in persona_prefixes):
        raise HTTPException(
            status_code=400,
            detail=f"'{base_model}' is already a launched persona model. Select the original base model."
        )

    # Build a clean model name: "<persona>-<base>" stripping :latest suffix
    name_part = base_short.replace(":", "-")
    if name_part.endswith("-latest"):
        name_part = name_part[:-7]  # strip -latest
    launched_name = f"{persona_id}-{name_part}"

    logger.info("Launching persona '%s' on '%s' → '%s'", persona_id, base_model, launched_name)

    # Ollama v0.14+ uses `from` + `system` fields directly (no `modelfile` string).
    # Older versions used `{"name": ..., "modelfile": "FROM ...\nSYSTEM ..."}` — dropped.
    create_body = {
        "model": launched_name,
        "from": base_model,
        "system": persona["system"],
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/create",
                json=create_body,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}")

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama create failed ({resp.status_code}): {resp.text}",
        )

    return JSONResponse({
        "name": launched_name,
        "persona": persona_id,
        "persona_label": persona["label"],
        "base": base_model,
        "status": "created",
    })


# ---------------------------------------------------------------------------
# GET /v1/launched  — list models that look like launched personas
# ---------------------------------------------------------------------------

@router.get("/launched")
async def list_launched(request: Request) -> JSONResponse:
    """Return Ollama models whose name matches a known persona prefix."""
    _check_auth(request)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {exc}")

    persona_prefixes = tuple(PERSONAS.keys())
    launched = [
        _ollama_model_to_openai(m)
        for m in resp.json().get("models", [])
        if m.get("name", "").startswith(persona_prefixes)
    ]
    return JSONResponse({"object": "list", "data": launched})


# ===========================================================================
# ANTHROPIC-COMPATIBLE PROXY  (/anthropic/v1/messages)
#
# Implements enough of the Anthropic Messages API so that Claude Code (and
# any other tool that speaks the Anthropic SDK) can talk to a local Ollama
# model without any real API key.
#
# Point Claude Code at this server:
#   export ANTHROPIC_BASE_URL=http://localhost:8000/anthropic
#   export ANTHROPIC_API_KEY=local        # any non-empty string
#   export ANTHROPIC_AUTH_TOKEN=local     # any non-empty string
#
# Equivalent to what `ollama launch claude --model <base>` does automatically.
# ===========================================================================

def _anthropic_to_oai_messages(messages: list, system: str | None) -> list:
    """Translate Anthropic messages array + system string → OpenAI messages."""
    oai: list[dict] = []
    if system:
        oai.append({"role": "system", "content": system})
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # Anthropic content can be a list of typed blocks
        if isinstance(content, list):
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        oai.append({"role": role, "content": content})
    return oai


def _oai_to_anthropic_response(oai: dict, model: str, msg_id: str) -> dict:
    """Translate a non-streaming OpenAI response → Anthropic Messages format."""
    choice = oai.get("choices", [{}])[0]
    text = choice.get("message", {}).get("content", "")
    finish = choice.get("finish_reason", "stop")
    stop_reason = "end_turn" if finish in ("stop", "length", None) else finish
    usage = oai.get("usage", {})
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def _stream_as_anthropic(
    oai_body: dict,
    msg_id: str,
    model: str,
) -> AsyncIterator[bytes]:
    """Stream Ollama SSE and re-emit in Anthropic streaming event format."""

    def _sse(event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield _sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    yield b"event: ping\ndata: {\"type\":\"ping\"}\n\n"

    out_tokens = 0
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{_OLLAMA_V1}/chat/completions",
            json={**oai_body, "stream": True},
            headers={"Content-Type": "application/json"},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        out_tokens += 1
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": delta},
                        })
                except Exception:
                    continue

    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": out_tokens},
    })
    yield b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"


# ---------------------------------------------------------------------------
# GET /anthropic/v1/models  (Claude Code hits this on startup)
# ---------------------------------------------------------------------------

@anthropic_router.get("/v1/models")
async def anthropic_list_models(request: Request) -> JSONResponse:
    """Return Ollama models in Anthropic model-list format."""
    if _PROXY_API_KEY:
        key = (
            request.headers.get("x-api-key", "")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )
        if key != _PROXY_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {exc}")

    models = [
        {
            "type": "model",
            "id": m.get("name", ""),
            "display_name": m.get("name", ""),
            "created_at": m.get("modified_at", ""),
        }
        for m in resp.json().get("models", [])
    ]
    return JSONResponse({"data": models})


# ---------------------------------------------------------------------------
# POST /anthropic/v1/messages
# ---------------------------------------------------------------------------

@anthropic_router.post("/v1/messages")
async def anthropic_messages(request: Request):
    """
    Anthropic Messages API compatible endpoint backed by local Ollama.
    Point Claude Code here via env vars — no real Anthropic API key needed.
    """
    if _PROXY_API_KEY:
        key = (
            request.headers.get("x-api-key", "")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )
        if key != _PROXY_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    import uuid
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    model      = body.get("model", "")
    system     = body.get("system", None)
    if isinstance(system, list):
        system = "\n".join(
            b.get("text", "") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )
    messages    = body.get("messages", [])
    max_tokens  = body.get("max_tokens", 4096)
    stream      = body.get("stream", False)
    temperature = body.get("temperature", 0.7)

    oai_body = {
        "model": model,
        "messages": _anthropic_to_oai_messages(messages, system),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if stream:
        return StreamingResponse(
            _stream_as_anthropic(oai_body, msg_id, model),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{_OLLAMA_V1}/chat/completions",
                json=oai_body,
                headers={"Content-Type": "application/json"},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}")

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return JSONResponse(_oai_to_anthropic_response(resp.json(), model, msg_id))


# ---------------------------------------------------------------------------
# Registration helper (called from main.py)
# ---------------------------------------------------------------------------

def register_proxy_routes(app) -> None:
    """Include the /v1 proxy router and /anthropic proxy router in the FastAPI app."""
    app.include_router(router)
    app.include_router(anthropic_router)
    logger.info(
        "OpenAI-compatible /v1 proxy registered → %s/v1  (auth=%s)",
        OLLAMA_BASE,
        "enabled" if _PROXY_API_KEY else "disabled (open)",
    )
    logger.info(
        "Anthropic-compatible /anthropic proxy registered → %s/anthropic/v1/messages",
        "http://localhost:8000",
    )
