"""Database operations for the Math Tutor.

A worksheet is stored as its *recipe* — topics, grade, count, seed — and not as
the finished problems. The generators are deterministic, so the recipe
regenerates the identical sheet whenever it is needed, which means a worksheet
printed in March can have its answer key reprinted in June and the two still
match. It also keeps the table small: a 40-question sheet is a hundred bytes.

The classroom table records what was handed out and what came back, so a parent
can see which topics a child is actually getting wrong rather than guessing.
"""

import json
import uuid
import logging
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)


async def init_math_tables(db: aiosqlite.Connection):
    """Create the math tutor tables if they don't exist."""
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS math_worksheets (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL DEFAULT '',
            title       TEXT NOT NULL,
            grade       TEXT NOT NULL,
            topics      TEXT NOT NULL DEFAULT '[]',
            count       INTEGER NOT NULL DEFAULT 10,
            seed        INTEGER NOT NULL,
            student     TEXT NOT NULL DEFAULT '',
            notes       TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL
        )
        """
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_math_ws_user "
        "ON math_worksheets(user_id, created_at DESC)"
    )

    # One row per marked question, so "which topics is she weak on" is a query
    # rather than a memory.
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS math_results (
            id            TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL DEFAULT '',
            worksheet_id  TEXT NOT NULL,
            student       TEXT NOT NULL DEFAULT '',
            topic         TEXT NOT NULL DEFAULT '',
            position      INTEGER NOT NULL DEFAULT 0,
            correct       INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL,
            FOREIGN KEY (worksheet_id) REFERENCES math_worksheets(id)
        )
        """
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_math_res_ws "
        "ON math_results(worksheet_id)"
    )
    await db.commit()


async def save_worksheet(db: aiosqlite.Connection, user_id: str, *,
                         title: str, grade: str, topics: list[str],
                         count: int, seed: int, student: str = "",
                         notes: str = "") -> dict:
    wid = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO math_worksheets "
        "(id, user_id, title, grade, topics, count, seed, student, notes, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (wid, user_id, title, grade, json.dumps(topics), int(count), int(seed),
         student, notes, now),
    )
    await db.commit()
    return {"id": wid, "title": title, "grade": grade, "topics": topics,
            "count": count, "seed": seed, "student": student, "notes": notes,
            "created_at": now}


def _row(r) -> dict:
    return {"id": r[0], "title": r[1], "grade": r[2],
            "topics": json.loads(r[3] or "[]"), "count": r[4], "seed": r[5],
            "student": r[6], "notes": r[7], "created_at": r[8]}


_COLS = "id, title, grade, topics, count, seed, student, notes, created_at"


async def list_worksheets(db: aiosqlite.Connection, user_id: str,
                          limit: int = 100) -> list[dict]:
    cur = await db.execute(
        f"SELECT {_COLS} FROM math_worksheets WHERE user_id = ? "
        "ORDER BY created_at DESC LIMIT ?", (user_id, int(limit)))
    return [_row(r) for r in await cur.fetchall()]


async def get_worksheet(db: aiosqlite.Connection, user_id: str,
                        wid: str) -> dict | None:
    cur = await db.execute(
        f"SELECT {_COLS} FROM math_worksheets WHERE id = ? AND user_id = ?",
        (wid, user_id))
    row = await cur.fetchone()
    return _row(row) if row else None


async def delete_worksheet(db: aiosqlite.Connection, user_id: str,
                           wid: str) -> bool:
    cur = await db.execute(
        "DELETE FROM math_worksheets WHERE id = ? AND user_id = ?",
        (wid, user_id))
    await db.execute("DELETE FROM math_results WHERE worksheet_id = ? "
                     "AND user_id = ?", (wid, user_id))
    await db.commit()
    return cur.rowcount > 0


async def record_results(db: aiosqlite.Connection, user_id: str,
                         worksheet_id: str, student: str,
                         marks: list[dict]) -> int:
    """Store a marked sheet. *marks* is [{position, topic, correct}, ...]."""
    now = datetime.now(timezone.utc).isoformat()
    rows = [(uuid.uuid4().hex[:12], user_id, worksheet_id, student,
             str(m.get("topic", "")), int(m.get("position", 0)),
             1 if m.get("correct") else 0, now) for m in marks]
    await db.executemany(
        "INSERT INTO math_results "
        "(id, user_id, worksheet_id, student, topic, position, correct, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    await db.commit()
    return len(rows)


async def topic_performance(db: aiosqlite.Connection, user_id: str,
                            student: str = "") -> list[dict]:
    """Accuracy per topic, worst first — which is the useful ordering."""
    sql = ("SELECT topic, COUNT(*), SUM(correct) FROM math_results "
           "WHERE user_id = ?")
    args: list = [user_id]
    if student:
        sql += " AND student = ?"
        args.append(student)
    sql += " GROUP BY topic HAVING COUNT(*) > 0"
    cur = await db.execute(sql, args)
    out = [{"topic": t, "attempted": n, "correct": c or 0,
            "accuracy": round((c or 0) / n, 3)}
           for t, n, c in await cur.fetchall()]
    return sorted(out, key=lambda d: d["accuracy"])
