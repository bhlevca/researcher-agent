"""Which model a request runs on: the person's choice, not the server's default.

``MODEL`` in ``.env`` is the fallback for a fresh install. It is not a setting,
and it must never beat a choice someone made in the interface — but that is
exactly what used to happen. The model picker wrote the choice to
``users.model`` in the database, and also to ``app.state.current_model``; the
crews were then built from ``current_model``. That global is reset from the
environment on every restart, so after a restart the picker showed the saved
model while the crews ran the one from ``.env``. When the ``.env`` model had
since been deleted from Ollama, the result was a 404 naming a model nobody had
selected for months.

So the database is the only authority here. A crew is rebuilt when the person's
saved model differs from the one their cached crew was built with, which also
fixes the second half of the bug: ``current_model`` was process-wide, so in a
household the last person to touch the picker silently changed which model
everybody else's lazily-created crews would use.

The per-crew LLM parameter overrides live in their own columns and are
reapplied on every rebuild, because a rebuilt crew starts from the class
defaults and would otherwise quietly discard them.
"""

from __future__ import annotations

import os
import json
import logging

logger = logging.getLogger(__name__)

# kind -> (app.state attribute holding the per-user cache, params column)
_KINDS = {
    "research": ("research_crews", "llm_params"),
    "tutor": ("tutor_crews", "tutor_llm_params"),
    "composer": ("composer_crews", "composer_llm_params"),
}


def _crew_class(kind: str):
    """Imported lazily: these pull in CrewAI, which is slow to import."""
    if kind == "research":
        from researcher.crew import ResearchCrew
        return ResearchCrew
    if kind == "tutor":
        from researcher.tutor.crew import TutorCrew
        return TutorCrew
    if kind == "composer":
        from researcher.composer.crew import ComposerCrew
        return ComposerCrew
    raise ValueError(f"unknown crew kind: {kind}")


def default_model(app) -> str:
    """The fallback for someone who has never chosen a model."""
    return (getattr(app.state, "default_model", None)
            or os.getenv("MODEL", "ollama/qwen3.5:9b"))


async def resolve_model(request, uid: str) -> tuple[str, str | None]:
    """Return (model, llm_params_json) for *uid* from the database.

    Falls back to the configured default only when the person has no saved
    choice at all.
    """
    db = getattr(request.app.state, "db", None)
    if db is None:                      # tests and health checks
        return default_model(request.app), None
    cursor = await db.execute(
        "SELECT model, llm_params FROM users WHERE id = ?", (uid,))
    row = await cursor.fetchone()
    chosen = (row[0] if row and row[0] else "") or default_model(request.app)
    return chosen, (row[1] if row and row[1] else None)


async def get_crew(request, uid: str, kind: str):
    """The per-user crew for *kind*, built with the model that person chose.

    Rebuilt when their saved model has changed since the cached crew was made;
    returned as-is otherwise, because constructing a crew probes the model's
    capabilities and is not free.
    """
    store_name, params_column = _KINDS[kind]
    store = getattr(request.app.state, store_name, None)
    if store is None:
        store = {}
        setattr(request.app.state, store_name, store)

    db = getattr(request.app.state, "db", None)
    if db is None:
        chosen, params_json = default_model(request.app), None
    else:
        cursor = await db.execute(
            f"SELECT model, {params_column} FROM users WHERE id = ?", (uid,))
        row = await cursor.fetchone()
        chosen = (row[0] if row and row[0] else "") or default_model(request.app)
        params_json = row[1] if row and row[1] else None

    crew = store.get(uid)
    if crew is not None and getattr(crew, "model_name", None) == chosen:
        return crew

    if crew is not None:
        logger.info("[model] rebuilding %s crew for %s: %s -> %s", kind, uid,
                    getattr(crew, "model_name", "?"), chosen)
    crew = _crew_class(kind)(model=chosen)
    if params_json:
        try:
            crew.update_llm_params(json.loads(params_json))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("[model] ignoring unreadable %s params for %s: %s",
                           kind, uid, e)
    store[uid] = crew
    return crew


async def verify_default_model(app) -> str:
    """Check the configured default exists in Ollama; substitute if it does not.

    ``MODEL`` in ``.env`` names a model that may since have been deleted —
    which is how this whole class of bug announced itself: a 404 for
    ``huihui_ai/mistral-small-abliterated``, a model removed months earlier and
    still named in ``.env``. Anyone who has chosen a model in the interface is
    unaffected, because their choice is read from the database. Someone who
    never chose would otherwise get the 404 and no clue why.

    Substituting beats failing: the server starts, the log says plainly what
    happened, and the person can pick what they want in the interface. Ollama
    being unreachable is not treated as an answer either way.
    """
    import httpx

    from researcher.config import OLLAMA_BASE

    configured = default_model(app)
    want = configured.replace("ollama/", "")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            installed = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        logger.warning("[model] could not check MODEL=%s against Ollama: %s",
                       configured, e)
        return configured

    if want in installed:
        return configured

    # Prefer a model with no suffix oddities; otherwise just take the first.
    substitute = next((m for m in installed if "embed" not in m), None)
    logger.error(
        "[model] MODEL=%s in .env is not installed in Ollama. Installed: %s",
        configured, ", ".join(installed) or "(none)")
    if not substitute:
        logger.error("[model] no usable model installed; requests will fail "
                     "until one is pulled or chosen in the interface")
        return configured
    chosen = f"ollama/{substitute}"
    logger.error("[model] falling back to %s — fix MODEL in .env, or just "
                 "pick a model in the interface (that choice is saved "
                 "per-person and always wins)", chosen)
    app.state.default_model = chosen
    return chosen
