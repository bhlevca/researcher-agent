"""Model switching, memory depth, and LLM parameter endpoints."""

import json
import logging

from fastapi import HTTPException, Request
import httpx

from researcher.config import OLLAMA_BASE, ModelRequest
from researcher.crew import ResearchCrew
import researcher.crew as _crew_mod
from researcher.auth import get_current_user, get_optional_user

logger = logging.getLogger(__name__)


def register_model_routes(app):
    """Attach /models, /model, /memory-depth, /llm-params endpoints."""

    @app.get("/info")
    async def info(request: Request):
        user = await get_optional_user(request)
        if user:
            db = request.app.state.db
            cursor = await db.execute("SELECT model FROM users WHERE id = ?", (user["id"],))
            row = await cursor.fetchone()
            model = (row[0] if row and row[0] else None) or request.app.state.current_model
        else:
            model = request.app.state.current_model
        return {"model": model, "user": user}

    @app.get("/models")
    async def list_models(request: Request):
        """Query Ollama for available local models."""
        await get_current_user(request)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{OLLAMA_BASE}/api/tags")
                resp.raise_for_status()
                data = resp.json()
            models = [
                {
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / 1e9, 1),
                    "family": m.get("details", {}).get("family", ""),
                    "params": m.get("details", {}).get("parameter_size", ""),
                }
                for m in data.get("models", [])
            ]
            return {"models": models, "current": request.app.state.current_model}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {e}")

    @app.post("/model")
    async def switch_model(req: ModelRequest, request: Request):
        """Switch the active LLM model. Stores per-user preference."""
        user = await get_current_user(request)
        new_model = req.model if req.model.startswith("ollama/") else f"ollama/{req.model}"
        try:
            new_crew = ResearchCrew(model=new_model)
            # Restore user's saved LLM params on model switch
            db = request.app.state.db
            cursor = await db.execute("SELECT llm_params FROM users WHERE id = ?", (user["id"],))
            row = await cursor.fetchone()
            saved = row[0] if row and row[0] else None
            if saved:
                try:
                    new_crew.update_llm_params(json.loads(saved))
                except (json.JSONDecodeError, TypeError):
                    pass
            request.app.state.research_crew = new_crew
            request.app.state.current_model = new_model

            await db.execute(
                "UPDATE users SET model = ? WHERE id = ?",
                (new_model, user["id"]),
            )
            await db.commit()

            return {"model": new_model}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to switch model: {e}")

    # --- Memory depth toggle ---

    @app.get("/memory-depth")
    async def get_memory_depth():
        """Return the current memory recall depth."""
        return {"depth": _crew_mod.memory_depth}

    @app.post("/memory-depth")
    async def set_memory_depth(request: Request):
        """Switch memory recall depth between 'shallow' and 'deep'."""
        body = await request.json()
        depth = body.get("depth", "shallow")
        if depth not in ("shallow", "deep"):
            raise HTTPException(status_code=400, detail="depth must be 'shallow' or 'deep'")
        _crew_mod.memory_depth = depth
        logger.info("Memory depth set to: %s", depth)
        return {"depth": depth}

    # --- LLM parameters ---

    @app.get("/llm-params")
    async def get_llm_params(request: Request):
        """Return current LLM generation parameters (loads from DB if available)."""
        user = await get_current_user(request)
        research_crew = request.app.state.research_crew
        db = request.app.state.db
        cursor = await db.execute("SELECT llm_params FROM users WHERE id = ?", (user["id"],))
        row = await cursor.fetchone()
        saved = row[0] if row and row[0] else None
        if saved:
            try:
                saved_params = json.loads(saved)
                research_crew.update_llm_params(saved_params)
            except (json.JSONDecodeError, TypeError):
                pass
        return {"params": research_crew.get_llm_params()}

    @app.post("/llm-params")
    async def set_llm_params(request: Request):
        """Update LLM generation parameters. Accepts a partial dict. Persists to DB."""
        user = await get_current_user(request)
        body = await request.json()
        params = body.get("params", {})
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="params must be a dict")
        research_crew = request.app.state.research_crew
        updated = research_crew.update_llm_params(params)
        db = request.app.state.db
        await db.execute(
            "UPDATE users SET llm_params = ? WHERE id = ?",
            (json.dumps(updated), user["id"]),
        )
        await db.commit()
        logger.info("LLM params updated and saved: %s", params)
        return {"params": updated}

    @app.post("/llm-params/reset")
    async def reset_llm_params(request: Request):
        """Reset LLM parameters to defaults. Clears DB storage."""
        user = await get_current_user(request)
        research_crew = request.app.state.research_crew
        updated = research_crew.update_llm_params(dict(ResearchCrew.DEFAULT_LLM_PARAMS))
        db = request.app.state.db
        await db.execute(
            "UPDATE users SET llm_params = '' WHERE id = ?",
            (user["id"],),
        )
        await db.commit()
        logger.info("LLM params reset to defaults")
        return {"params": updated}

    # --- Image generation parameters ---

    from researcher.image import (
        get_image_params,
        update_image_params,
        DEFAULT_IMAGE_PARAMS,
        load_image_params_from_json,
    )

    @app.get("/image-params")
    async def get_img_params(request: Request):
        """Return current image generation parameters (loads from DB if available)."""
        user = await get_current_user(request)
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT image_params FROM users WHERE id = ?", (user["id"],)
        )
        row = await cursor.fetchone()
        saved = row[0] if row and row[0] else None
        load_image_params_from_json(saved)
        return {"params": get_image_params()}

    @app.post("/image-params")
    async def set_img_params(request: Request):
        """Update image generation parameters. Persists to DB."""
        user = await get_current_user(request)
        body = await request.json()
        params = body.get("params", {})
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="params must be a dict")
        updated = update_image_params(params)
        db = request.app.state.db
        await db.execute(
            "UPDATE users SET image_params = ? WHERE id = ?",
            (json.dumps(updated), user["id"]),
        )
        await db.commit()
        logger.info("Image params updated and saved: %s", params)
        return {"params": updated}

    @app.post("/image-params/reset")
    async def reset_img_params(request: Request):
        """Reset image parameters to defaults. Clears DB storage."""
        user = await get_current_user(request)
        updated = update_image_params(dict(DEFAULT_IMAGE_PARAMS))
        db = request.app.state.db
        await db.execute(
            "UPDATE users SET image_params = '' WHERE id = ?",
            (user["id"],),
        )
        await db.commit()
        logger.info("Image params reset to defaults")
        return {"params": updated}
