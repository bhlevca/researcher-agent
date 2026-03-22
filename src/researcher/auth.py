"""Authentication and multi-user support.

Provides JWT-based authentication with bcrypt password hashing.
Users register/login and receive a token that is sent as a Bearer header.
Unauthenticated requests get a ``guest`` user context so the app still
works without login — but guests cannot save/load sessions.

Public API
----------
- ``register_auth_routes(app)`` — attach ``/auth/*`` endpoints
- ``get_current_user(request)`` — FastAPI dependency returning user dict
- ``get_optional_user(request)`` — same but returns ``None`` for guests
- ``init_users_table(db)`` — create the ``users`` table if missing
- ``migrate_sessions_table(db)`` — add ``user_id`` column if missing
"""

import os
import re
import time
import logging
import collections
from datetime import datetime, timezone, timedelta

import bcrypt
import jwt
from fastapi import HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Secret for signing JWTs — override via env var in production
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production-please!!")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))

# Invite code — only people who know this can register.
# Set via env var; empty string disables the gate (open registration).
INVITE_CODE = os.getenv("INVITE_CODE", "")

_USERNAME_RE = re.compile(r'^[a-zA-Z0-9_]{3,30}$')

# ---------------------------------------------------------------------------
# Rate limiting (per-IP, in-memory)
# ---------------------------------------------------------------------------

_AUTH_RATE_WINDOW = 60          # seconds
_AUTH_RATE_MAX_ATTEMPTS = 5     # max attempts per window
_rate_log: dict[str, collections.deque] = {}  # ip -> deque of timestamps


def _check_rate_limit(ip: str):
    """Raise 429 if this IP has exceeded the auth attempt limit."""
    now = time.monotonic()
    dq = _rate_log.setdefault(ip, collections.deque())
    # Purge entries older than the window
    while dq and dq[0] < now - _AUTH_RATE_WINDOW:
        dq.popleft()
    if len(dq) >= _AUTH_RATE_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please wait a minute and try again.",
        )
    dq.append(now)

# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _create_token(user_id: str, username: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_user(request: Request) -> dict:
    """Require a valid JWT. Returns ``{"id": ..., "username": ...}``."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = _decode_token(auth[7:])
    return {"id": payload["sub"], "username": payload["username"]}


async def get_optional_user(request: Request) -> dict | None:
    """Return user dict if a valid JWT is present, else ``None`` (guest)."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        payload = _decode_token(auth[7:])
        return {"id": payload["sub"], "username": payload["username"]}
    except HTTPException:
        return None


# ---------------------------------------------------------------------------
# DB helpers — called once during lifespan startup
# ---------------------------------------------------------------------------

async def init_users_table(db):
    """Create the ``users`` table if it doesn't exist."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            username    TEXT UNIQUE NOT NULL,
            pw_hash     TEXT NOT NULL,
            model       TEXT DEFAULT '',
            created_at  TEXT NOT NULL
        )
    """)
    await db.commit()


async def migrate_sessions_table(db):
    """Add ``user_id`` column to sessions if missing (backward-compat)."""
    cursor = await db.execute("PRAGMA table_info(sessions)")
    cols = {row[1] for row in await cursor.fetchall()}
    if "user_id" not in cols:
        await db.execute(
            "ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT '' "
        )
        await db.commit()
        logger.info("Migrated sessions table: added user_id column")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str
    password: str
    invite_code: str = ""
    website: str = ""       # honeypot — must be empty


class LoginRequest(BaseModel):
    username: str
    password: str
    website: str = ""       # honeypot — must be empty


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_auth_routes(app):
    """Attach ``/auth/register``, ``/auth/login``, ``/auth/me`` to *app*."""
    import uuid

    @app.get("/auth/config")
    async def auth_config():
        """Tell the frontend whether an invite code is required."""
        return {"invite_required": bool(INVITE_CODE)}

    @app.post("/auth/register")
    async def register(req: RegisterRequest, request: Request):
        ip = request.client.host if request.client else "unknown"
        _check_rate_limit(ip)

        # Honeypot: bots fill the invisible 'website' field
        if req.website:
            # Silently reject — looks like success to bots
            logger.warning("Honeypot triggered from %s", ip)
            raise HTTPException(status_code=400, detail="Registration failed")

        # Invite code gate
        if INVITE_CODE and req.invite_code.strip() != INVITE_CODE:
            raise HTTPException(status_code=403, detail="Invalid invite code")

        username = req.username.strip()
        if not _USERNAME_RE.match(username):
            raise HTTPException(
                status_code=400,
                detail="Username must be 3-30 characters: letters, digits, underscore",
            )
        if len(req.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

        db = request.app.state.db
        cursor = await db.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        )
        if await cursor.fetchone():
            raise HTTPException(status_code=409, detail="Username already taken")

        user_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        pw_hash = _hash_password(req.password)

        await db.execute(
            "INSERT INTO users (id, username, pw_hash, model, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, pw_hash, "", now),
        )
        await db.commit()
        logger.info("New user registered: %s from %s", username, ip)

        token = _create_token(user_id, username)
        return {"token": token, "user": {"id": user_id, "username": username}}

    @app.post("/auth/login")
    async def login(req: LoginRequest, request: Request):
        ip = request.client.host if request.client else "unknown"
        _check_rate_limit(ip)

        # Honeypot
        if req.website:
            logger.warning("Honeypot triggered from %s", ip)
            raise HTTPException(status_code=401, detail="Invalid username or password")

        db = request.app.state.db
        cursor = await db.execute(
            "SELECT id, username, pw_hash FROM users WHERE username = ?",
            (req.username.strip(),),
        )
        row = await cursor.fetchone()
        if not row or not _verify_password(req.password, row[2]):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        token = _create_token(row[0], row[1])
        return {"token": token, "user": {"id": row[0], "username": row[1]}}

    @app.get("/auth/me")
    async def me(request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT model FROM users WHERE id = ?", (user["id"],)
        )
        row = await cursor.fetchone()
        return {
            "id": user["id"],
            "username": user["username"],
            "model": row[0] if row else "",
        }

