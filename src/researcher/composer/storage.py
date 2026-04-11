"""Database operations for the Music Composer.

Manages composer sessions and saved compositions in SQLite.
"""

import json
import uuid
import logging
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------


async def init_composer_tables(db: aiosqlite.Connection):
    """Create all composer-related tables if they don't exist."""

    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS composer_sessions (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL DEFAULT '',
            name            TEXT NOT NULL,
            genre           TEXT NOT NULL DEFAULT 'classical',
            key_signature   TEXT NOT NULL DEFAULT 'C major',
            time_signature  TEXT NOT NULL DEFAULT '4/4',
            tempo           INTEGER NOT NULL DEFAULT 120,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            messages        TEXT NOT NULL DEFAULT '[]'
        )
        """
    )

    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS compositions (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL DEFAULT '',
            session_id      TEXT NOT NULL,
            title           TEXT NOT NULL,
            description     TEXT NOT NULL DEFAULT '',
            instruments     TEXT NOT NULL DEFAULT '[]',
            genre           TEXT NOT NULL DEFAULT '',
            key_signature   TEXT NOT NULL DEFAULT '',
            time_signature  TEXT NOT NULL DEFAULT '',
            tempo           INTEGER NOT NULL DEFAULT 120,
            musicxml        TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES composer_sessions(id)
        )
        """
    )

    await db.commit()
    logger.info("Composer tables initialised")


# ---------------------------------------------------------------------------
# Composer sessions
# ---------------------------------------------------------------------------


async def create_composer_session(
    db: aiosqlite.Connection,
    user_id: str,
    name: str,
    genre: str,
    key_signature: str,
    time_signature: str,
    tempo: int,
) -> dict:
    sid = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO composer_sessions "
        "(id, user_id, name, genre, key_signature, time_signature, tempo, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, user_id, name, genre, key_signature, time_signature, tempo, now, now),
    )
    await db.commit()
    return {
        "id": sid, "name": name, "genre": genre,
        "key_signature": key_signature, "time_signature": time_signature,
        "tempo": tempo, "created_at": now, "updated_at": now,
        "messages": [],
    }


async def list_composer_sessions(db: aiosqlite.Connection, user_id: str) -> list[dict]:
    cur = await db.execute(
        "SELECT id, name, genre, key_signature, time_signature, tempo, "
        "created_at, updated_at FROM composer_sessions "
        "WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    )
    rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_composer_session(
    db: aiosqlite.Connection, session_id: str, user_id: str
) -> dict | None:
    cur = await db.execute(
        "SELECT * FROM composer_sessions WHERE id = ? AND user_id = ?",
        (session_id, user_id),
    )
    row = await cur.fetchone()
    if not row:
        return None
    d = dict(row)
    d["messages"] = json.loads(d["messages"])
    return d


async def update_composer_session_messages(
    db: aiosqlite.Connection, session_id: str, messages: list
):
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "UPDATE composer_sessions SET messages = ?, updated_at = ? WHERE id = ?",
        (json.dumps(messages), now, session_id),
    )
    await db.commit()


async def delete_composer_session(
    db: aiosqlite.Connection, session_id: str, user_id: str
) -> bool:
    cur = await db.execute(
        "DELETE FROM composer_sessions WHERE id = ? AND user_id = ?",
        (session_id, user_id),
    )
    await db.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Compositions
# ---------------------------------------------------------------------------


async def save_composition(
    db: aiosqlite.Connection,
    user_id: str,
    session_id: str,
    title: str,
    description: str,
    instruments: list[str],
    genre: str,
    key_signature: str,
    time_signature: str,
    tempo: int,
    musicxml: str,
) -> dict:
    cid = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO compositions "
        "(id, user_id, session_id, title, description, instruments, "
        "genre, key_signature, time_signature, tempo, musicxml, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            cid, user_id, session_id, title, description,
            json.dumps(instruments), genre, key_signature, time_signature,
            tempo, musicxml, now,
        ),
    )
    await db.commit()
    return {
        "id": cid, "title": title, "description": description,
        "instruments": instruments, "genre": genre,
        "key_signature": key_signature, "time_signature": time_signature,
        "tempo": tempo, "created_at": now,
    }


async def list_compositions(
    db: aiosqlite.Connection, user_id: str, session_id: str
) -> list[dict]:
    cur = await db.execute(
        "SELECT id, title, description, instruments, genre, key_signature, "
        "time_signature, tempo, created_at FROM compositions "
        "WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
        (user_id, session_id),
    )
    rows = await cur.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["instruments"] = json.loads(d["instruments"])
        result.append(d)
    return result


async def get_composition(
    db: aiosqlite.Connection, composition_id: str, user_id: str
) -> dict | None:
    cur = await db.execute(
        "SELECT * FROM compositions WHERE id = ? AND user_id = ?",
        (composition_id, user_id),
    )
    row = await cur.fetchone()
    if not row:
        return None
    d = dict(row)
    d["instruments"] = json.loads(d["instruments"])
    return d


async def delete_composition(
    db: aiosqlite.Connection, composition_id: str, user_id: str
) -> bool:
    cur = await db.execute(
        "DELETE FROM compositions WHERE id = ? AND user_id = ?",
        (composition_id, user_id),
    )
    await db.commit()
    return cur.rowcount > 0
