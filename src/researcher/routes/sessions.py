"""Session management endpoints (SQLite-backed)."""

import json
import uuid
import logging
from datetime import datetime, timezone

from fastapi import HTTPException, Request

from researcher.config import SessionSaveRequest, _SESSION_ID_RE
from researcher.auth import get_current_user

logger = logging.getLogger(__name__)


def _validate_sid(session_id: str):
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")


def register_session_routes(app):
    """Attach /sessions CRUD endpoints to the FastAPI app."""

    @app.get("/sessions")
    async def list_sessions(request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT id, name, created_at, updated_at, model, messages "
            "FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user["id"],),
        )
        rows = await cursor.fetchall()
        sessions = []
        for row in rows:
            try:
                msgs = json.loads(row["messages"])
            except (json.JSONDecodeError, TypeError):
                msgs = []
            sessions.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "message_count": len(msgs),
                    "model": row["model"] or "",
                }
            )
        return {"sessions": sessions}

    @app.post("/sessions")
    async def save_session(req: SessionSaveRequest, request: Request):
        user = await get_current_user(request)
        db = request.app.state.db
        sid = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT INTO sessions (id, name, created_at, updated_at, model, messages, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                sid,
                req.name,
                now,
                now,
                request.app.state.current_model,
                json.dumps(req.messages),
                user["id"],
            ),
        )
        await db.commit()
        return {"id": sid, "name": req.name}

    @app.get("/sessions/{session_id}")
    async def load_session(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        owner = row["user_id"] if "user_id" in row.keys() else ""
        if owner and (not user or user["id"] != owner):
            raise HTTPException(status_code=403, detail="Not your session")
        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "model": row["model"],
            "messages": json.loads(row["messages"]),
        }

    @app.put("/sessions/{session_id}")
    async def update_session(session_id: str, req: SessionSaveRequest, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT id, user_id FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        owner = row[1] or ""
        if owner and (not user or user["id"] != owner):
            raise HTTPException(status_code=403, detail="Not your session")
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE sessions SET name=?, messages=?, updated_at=?, model=? WHERE id=?",
            (
                req.name,
                json.dumps(req.messages),
                now,
                request.app.state.current_model,
                session_id,
            ),
        )
        await db.commit()
        return {"id": session_id, "name": req.name}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        _validate_sid(session_id)
        user = await get_current_user(request)
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT id, user_id FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        owner = row[1] or ""
        if owner and (not user or user["id"] != owner):
            raise HTTPException(status_code=403, detail="Not your session")
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()
        return {"deleted": session_id}
