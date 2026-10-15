"""Model switching, memory depth, and LLM parameter endpoints."""

import json
import logging

from fastapi import HTTPException, Request
import httpx

from researcher.gpu_broker import GPU
from researcher.user_crews import get_crew, resolve_model, default_model
from researcher.config import OLLAMA_BASE, ModelRequest
from researcher.crew import ResearchCrew
from researcher.tutor.crew import TutorCrew
from researcher.composer.crew import ComposerCrew
import researcher.crew as _crew_mod
import researcher.patches as _patches_mod
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
        user = await get_current_user(request)
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
            db = request.app.state.db
            cursor = await db.execute(
                "SELECT model FROM users WHERE id = ?", (user["id"],)
            )
            row = await cursor.fetchone()
            current = (row[0] if row and row[0] else None) or request.app.state.current_model
            return {"models": models, "current": current}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {e}")

    @app.post("/model/unload")
    async def unload_model(request: Request):
        """Unload the current model from GPU (free VRAM). Sends keep_alive=0 to Ollama."""
        user = await get_current_user(request)
        # Unload what this person is running, not whatever the global happens
        # to hold.
        current, _ = await resolve_model(request, user["id"])
        model_name = (current or "").replace("ollama/", "")
        if not model_name:
            raise HTTPException(status_code=400, detail="No model loaded")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={"model": model_name, "keep_alive": 0, "prompt": ""},
                )
                resp.raise_for_status()
            logger.info("Unloaded model %s from GPU", model_name)
            # The broker's idea of what is resident is now wrong.
            GPU.forget_residency()
            return {"status": "unloaded", "model": model_name}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to unload model: {e}")

    @app.post("/model")
    async def switch_model(req: ModelRequest, request: Request):
        """Record this person's model choice.

        The choice is saved to the database and the person's cached crews are
        dropped, so the next request rebuilds them from the saved value. This
        used to construct all three crews here and also assign
        app.state.current_model, which had two consequences: the work of
        building three crews happened on a settings click even for the two
        tabs the person was not using, and the global meant one person's
        choice became the default every other user's lazily-created crews were
        built with.
        """
        user = await get_current_user(request)
        uid = user["id"]
        new_model = (req.model if req.model.startswith("ollama/")
                     else f"ollama/{req.model}")
        try:
            db = request.app.state.db
            await db.execute("UPDATE users SET model = ? WHERE id = ?",
                             (new_model, uid))
            await db.commit()
            for store in ("research_crews", "tutor_crews", "composer_crews"):
                getattr(request.app.state, store, {}).pop(uid, None)
            # Swapping models means the card's contents are no longer what the
            # broker last recorded.
            GPU.forget_residency()
            return {"model": new_model}
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Failed to switch model: {e}")

    # --- Memory depth toggle ---

    @app.get("/memory-depth")
    async def get_memory_depth():
        """Return the current memory recall depth."""
        return {"depth": _patches_mod.memory_depth}

    @app.post("/memory-depth")
    async def set_memory_depth(request: Request):
        """Switch memory recall depth between 'shallow' and 'deep'."""
        body = await request.json()
        depth = body.get("depth", "shallow")
        if depth not in ("shallow", "deep"):
            raise HTTPException(status_code=400, detail="depth must be 'shallow' or 'deep'")
        _patches_mod.memory_depth = depth
        logger.info("Memory depth set to: %s", depth)
        return {"depth": depth}

    # --- LLM parameters ---

    # ── Subsystem parameters ──
    #
    # OCR, speech, spaced repetition, video and crew wiring are all process-wide
    # rather than per-user, so they live behind one endpoint instead of five.
    # Grouped so the Params panel can render a labelled section per subsystem.

    @app.get("/subsystem-params")
    async def get_subsystem_params(request: Request):
        """Return every runtime-tunable subsystem setting, grouped."""
        _user = await get_current_user(request)
        from researcher.ingestion import get_ocr_params
        from researcher.routes.speech import get_whisper_params
        from researcher.tutor.storage import get_srs_params
        from researcher.video import get_video_params
        import researcher.patches as _patches

        return {
            "params": {
                "ocr": get_ocr_params(),
                "whisper": get_whisper_params(),
                "srs": get_srs_params(),
                "video": get_video_params(),
                "crew": {
                    "executor": _patches.get_executor_class().__name__
                    if _patches.get_executor_class() else "AgentExecutor (experimental)",
                    "crew_timeout_default": request.app.state.crew_timeout,
                    # Both, because the whole point of this panel is telling
                    # them apart: "your_model" is what your requests run on,
                    # "default_model" is only the fallback from .env.
                    "your_model": (await resolve_model(request, _user["id"]))[0],
                    "default_model": default_model(request.app),
                },
            }
        }

    @app.post("/subsystem-params")
    async def set_subsystem_params(request: Request):
        """Update subsystem settings. Accepts any subset of the groups."""
        await get_current_user(request)
        body = await request.json()
        params = body.get("params", {})
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="params must be a dict")

        from researcher.ingestion import update_ocr_params, get_ocr_params
        from researcher.routes.speech import update_whisper_params, get_whisper_params
        from researcher.tutor.storage import update_srs_params, get_srs_params
        from researcher.video import update_video_params, get_video_params

        out = {}
        out["ocr"] = (update_ocr_params(params["ocr"]) if "ocr" in params
                      else get_ocr_params())
        out["whisper"] = (update_whisper_params(params["whisper"]) if "whisper" in params
                          else get_whisper_params())
        out["srs"] = (update_srs_params(params["srs"]) if "srs" in params
                      else get_srs_params())
        out["video"] = (update_video_params(params["video"]) if "video" in params
                        else get_video_params())
        # Crew settings are read-only here: the executor is chosen at import time
        # (patches bind to it) and the model has its own switcher.
        logger.info("Subsystem params updated: %s", list(params))
        return {"params": out}

    @app.get("/llm-params")
    async def get_llm_params(request: Request):
        """Return current LLM generation parameters (loads from DB if available)."""
        user = await get_current_user(request)
        uid = user["id"]
        research_crew = await get_crew(request, uid, "research")
        db = request.app.state.db
        cursor = await db.execute("SELECT llm_params FROM users WHERE id = ?", (uid,))
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
        uid = user["id"]
        body = await request.json()
        params = body.get("params", {})
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="params must be a dict")
        research_crew = await get_crew(request, uid, "research")
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
        uid = user["id"]
        research_crew = await get_crew(request, uid, "research")
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
        import json as _json
        _saved_dict = _json.loads(saved) if saved else {}
        logger.info("[GET /image-params] user=%s DB realesrgan_enabled=%s",
                    user["username"], _saved_dict.get("realesrgan_enabled", "MISSING"))
        load_image_params_from_json(saved)
        result = get_image_params()
        logger.info("[GET /image-params] returning realesrgan_enabled=%s", result.get("realesrgan_enabled"))
        return {"params": result}

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
        logger.info("[POST /image-params] user=%s received realesrgan=%s → saved realesrgan=%s",
                    user["username"], params.get("realesrgan_enabled", "MISSING"), updated.get("realesrgan_enabled"))
        return {"params": updated}

    @app.post("/image-params/reset")
    async def reset_img_params(request: Request):
        """Reset image parameters to defaults. Clears DB storage."""
        user = await get_current_user(request)
        updated = update_image_params(dict(DEFAULT_IMAGE_PARAMS))
        db = request.app.state.db
        await db.execute(
            "UPDATE users SET image_params = ? WHERE id = ?",
            (json.dumps(updated), user["id"]),
        )
        await db.commit()
        logger.info("Image params reset to defaults")
        return {"params": updated}

    # --- Tutor LLM parameters ---

    @app.get("/tutor/llm-params")
    async def get_tutor_llm_params(request: Request):
        """Return current Tutor LLM parameters (loads from DB if available)."""
        user = await get_current_user(request)
        uid = user["id"]
        tutor_crew = await get_crew(request, uid, "tutor")
        db = request.app.state.db
        cursor = await db.execute("SELECT tutor_llm_params FROM users WHERE id = ?", (uid,))
        row = await cursor.fetchone()
        saved = row[0] if row and row[0] else None
        if saved:
            try:
                tutor_crew.update_llm_params(json.loads(saved))
            except (json.JSONDecodeError, TypeError):
                pass
        return {"params": tutor_crew.get_llm_params()}

    @app.post("/tutor/llm-params")
    async def set_tutor_llm_params(request: Request):
        """Update Tutor LLM parameters. Persists to DB."""
        user = await get_current_user(request)
        uid = user["id"]
        body = await request.json()
        params = body.get("params", {})
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="params must be a dict")
        tutor_crew = await get_crew(request, uid, "tutor")
        updated = tutor_crew.update_llm_params(params)
        db = request.app.state.db
        await db.execute(
            "UPDATE users SET tutor_llm_params = ? WHERE id = ?",
            (json.dumps(updated), user["id"]),
        )
        await db.commit()
        logger.info("Tutor LLM params updated and saved: %s", params)
        return {"params": updated}

    @app.post("/tutor/llm-params/reset")
    async def reset_tutor_llm_params(request: Request):
        """Reset Tutor LLM parameters to defaults."""
        user = await get_current_user(request)
        uid = user["id"]
        tutor_crew = await get_crew(request, uid, "tutor")
        updated = tutor_crew.update_llm_params(dict(TutorCrew.DEFAULT_LLM_PARAMS))
        db = request.app.state.db
        await db.execute(
            "UPDATE users SET tutor_llm_params = '' WHERE id = ?",
            (user["id"],),
        )
        await db.commit()
        logger.info("Tutor LLM params reset to defaults")
        return {"params": updated}

    # --- Composer LLM parameters ---

    @app.get("/composer/llm-params")
    async def get_composer_llm_params(request: Request):
        """Return current Composer LLM parameters (loads from DB if available)."""
        user = await get_current_user(request)
        uid = user["id"]
        composer_crew = await get_crew(request, uid, "composer")
        db = request.app.state.db
        cursor = await db.execute("SELECT composer_llm_params FROM users WHERE id = ?", (uid,))
        row = await cursor.fetchone()
        saved = row[0] if row and row[0] else None
        if saved:
            try:
                composer_crew.update_llm_params(json.loads(saved))
            except (json.JSONDecodeError, TypeError):
                pass
        return {"params": composer_crew.get_llm_params()}

    @app.post("/composer/llm-params")
    async def set_composer_llm_params(request: Request):
        """Update Composer LLM parameters. Persists to DB."""
        user = await get_current_user(request)
        uid = user["id"]
        body = await request.json()
        params = body.get("params", {})
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="params must be a dict")
        composer_crew = await get_crew(request, uid, "composer")
        updated = composer_crew.update_llm_params(params)
        db = request.app.state.db
        await db.execute(
            "UPDATE users SET composer_llm_params = ? WHERE id = ?",
            (json.dumps(updated), user["id"]),
        )
        await db.commit()
        logger.info("Composer LLM params updated and saved: %s", params)
        return {"params": updated}

    @app.post("/composer/llm-params/reset")
    async def reset_composer_llm_params(request: Request):
        """Reset Composer LLM parameters to defaults."""
        user = await get_current_user(request)
        uid = user["id"]
        composer_crew = await get_crew(request, uid, "composer")
        updated = composer_crew.update_llm_params(dict(ComposerCrew.DEFAULT_LLM_PARAMS))
        db = request.app.state.db
        await db.execute(
            "UPDATE users SET composer_llm_params = '' WHERE id = ?",
            (user["id"],),
        )
        await db.commit()
        logger.info("Composer LLM params reset to defaults")
        return {"params": updated}
