import sys
import os

# Never write .pyc bytecode files — prevents stale cache bugs after edits
sys.dont_write_bytecode = True

# Disable CrewAI interactive trace prompt that blocks the server
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import aiosqlite

from researcher.image import generate_ai_image, preload_sd, preload_zimage, preload_ernie, preload_flux
from researcher.config import STATIC_DIR, DB_PATH
from researcher.tutor.storage import init_tutor_tables
from researcher.composer.storage import init_composer_tables
from researcher.mathtutor.storage import init_math_tables

from researcher.auth import (
    register_auth_routes,
    init_users_table,
    migrate_sessions_table,
)
from researcher.ingestion import (
    register_ingestion_routes,
    init_files_table,
)
from researcher.routes.chat import register_chat_routes
from researcher.routes.ask import register_ask_routes
from researcher.routes.sessions import register_session_routes
from researcher.routes.models import register_model_routes
from researcher.routes.speech import register_speech_routes
from researcher.routes.tutor import register_tutor_routes
from researcher.routes.composer import register_composer_routes
from researcher.routes.mathtutor import register_mathtutor_routes
from researcher.routes.proxy import register_proxy_routes
from researcher.tts import register_tts_routes

load_dotenv()
logger = logging.getLogger(__name__)


# --- Lifespan: initialise app.state, SQLite, SD preload ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -- Crew / model init --
    model = os.getenv("MODEL", "ollama/qwen3.5:9b")
    app.state.current_model = model

    # Per-user crew instances, semaphores, and cancel events.
    # Keyed by user_id so users are isolated from each other.
    # The underlying Ollama inference is still serial (one GPU), but state
    # corruption between users is prevented.
    app.state.research_crews = {}   # user_id -> ResearchCrew
    app.state.tutor_crews = {}      # user_id -> TutorCrew
    app.state.composer_crews = {}   # user_id -> ComposerCrew
    app.state.crew_semaphores = {}  # user_id -> asyncio.Semaphore(1)
    app.state.cancel_events = {}    # user_id -> threading.Event

    # Crew timeout in seconds (configurable; default 3 minutes)
    app.state.crew_timeout = int(os.getenv("CREW_TIMEOUT_SEC", "180"))

    # -- SQLite session database --
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            model       TEXT DEFAULT '',
            messages    TEXT NOT NULL DEFAULT '[]'
        )
    """
    )
    await db.commit()

    # Migrate any legacy JSON session files
    legacy_dir = Path(__file__).parent / "data" / "sessions"
    if legacy_dir.exists():
        migrated = 0
        for f in legacy_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                await db.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(id, name, created_at, updated_at, model, messages) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        data["id"],
                        data["name"],
                        data["created_at"],
                        data.get("updated_at", data["created_at"]),
                        data.get("model", ""),
                        json.dumps(data.get("messages", [])),
                    ),
                )
                migrated += 1
            except (json.JSONDecodeError, KeyError):
                continue
        if migrated:
            await db.commit()
            for f in legacy_dir.glob("*.json"):
                f.unlink()
            try:
                legacy_dir.rmdir()
            except OSError:
                pass
            logger.info("Migrated %d JSON session(s) to SQLite", migrated)

    app.state.db = db

    # Auth tables
    await init_users_table(db)
    await migrate_sessions_table(db)

    # Files table
    await init_files_table(db)

    # Tutor tables
    await init_tutor_tables(db)

    # Composer tables
    await init_composer_tables(db)
    await init_math_tables(db)

    # Optional image backend warm-up/bootstrap.
    # This can prefill HF cache on startup when missing.
    # Disable with IMAGE_WARMUP=0.
    # Prefer the backend stored in the DB (per-user setting) over the env var,
    # so that switching via the UI is respected on next restart.
    # A misconfigured image critic degrades silently to "no refinement", so
    # surface it once at startup instead of leaving it to be noticed never.
    try:
        from researcher.image import check_critic_available

        ok, msg = check_critic_available()
        # Written to stderr, not logger.info: uvicorn leaves the app logger at
        # WARNING, so an INFO confirmation is invisible — which would defeat a
        # check whose only job is to be seen. Matches the [ZIMG]/[OCR] convention.
        sys.stderr.write(f"[IMAGE] {'OK' if ok else 'WARNING'}: {msg}\n")
        sys.stderr.flush()
    except Exception as e:
        logger.warning("[IMAGE] Critic check failed: %s", e)

    # Video warm-up is opt-in (VIDEO_WARMUP=1): it reserves ~16 GiB of system RAM
    # to save the one-time pipeline load on the first request.
    try:
        from researcher.video import preload_video

        preload_video()
    except Exception as e:
        logger.warning("[VIDEO] Preload failed: %s", e)

    image_warmup = os.getenv("IMAGE_WARMUP", "1") == "1"
    if image_warmup:
        image_backend = os.getenv("IMAGE_BACKEND", "sd").lower()
        try:
            cursor = await db.execute(
                "SELECT image_params FROM users LIMIT 1"
            )
            row = await cursor.fetchone()
            if row and row[0]:
                _saved = json.loads(row[0])
                _db_backend = str(_saved.get("image_backend", "")).lower()
                if _db_backend in {"sd", "zimage", "ernie", "flux"}:
                    image_backend = _db_backend
                    logger.info("Image backend from DB: %s", image_backend)
        except Exception as e:
            logger.warning("Could not read image_backend from DB: %s", e)
        if image_backend == "zimage":
            logger.info("Warming up ZImage backend in background")
            preload_zimage()
        elif image_backend == "sd":
            logger.info("Warming up Stable Diffusion backend in background")
            preload_sd()
        elif image_backend == "ernie":
            logger.info("Warming up ERNIE backend in background")
            preload_ernie()
        elif image_backend == "flux":
            logger.info("Warming up FLUX backend in background")
            preload_flux()

    yield

    # -- Shutdown --
    await app.state.db.close()


app = FastAPI(lifespan=lifespan)

# --- Register all route modules ---
register_auth_routes(app)
register_ingestion_routes(app)
register_chat_routes(app)
register_ask_routes(app)
register_session_routes(app)
register_model_routes(app)
register_speech_routes(app)
register_tutor_routes(app)
register_composer_routes(app)
register_mathtutor_routes(app)
register_proxy_routes(app)
register_tts_routes(app)


# --- UI pages ---


_NO_STORE = {"Cache-Control": "no-store"}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers=_NO_STORE)


@app.get("/tutor")
async def tutor_page():
    return FileResponse(STATIC_DIR / "tutor.html", headers=_NO_STORE)


@app.get("/composer")
async def composer_page():
    return FileResponse(STATIC_DIR / "composer.html", headers=_NO_STORE)


@app.get("/math")
async def math_page():
    return FileResponse(STATIC_DIR / "math.html", headers=_NO_STORE)


@app.get("/chat-direct")
async def chat_direct_page():
    return FileResponse(STATIC_DIR / "chat-direct.html", headers=_NO_STORE)


# --- Static files & server entry point ---

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn

    # 120-second timeout for 5070 Ti to process long searches
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)
