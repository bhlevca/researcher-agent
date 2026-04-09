"""Database operations for the Language Tutor.

Manages tutor sessions (separate from regular chat sessions),
vocabulary entries, lesson plans, and quiz results in SQLite.
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


async def init_tutor_tables(db: aiosqlite.Connection):
    """Create all tutor-related tables if they don't exist."""

    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS tutor_sessions (
            id            TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL DEFAULT '',
            name          TEXT NOT NULL,
            target_lang   TEXT NOT NULL,
            native_lang   TEXT NOT NULL DEFAULT 'English',
            level         TEXT NOT NULL DEFAULT 'A1',
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            messages      TEXT NOT NULL DEFAULT '[]'
        )
        """
    )

    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS vocabulary (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL DEFAULT '',
            session_id      TEXT NOT NULL,
            target_lang     TEXT NOT NULL DEFAULT '',
            word            TEXT NOT NULL,
            translation     TEXT NOT NULL,
            context         TEXT NOT NULL DEFAULT '',
            phonetic        TEXT NOT NULL DEFAULT '',
            part_of_speech  TEXT NOT NULL DEFAULT '',
            mastery_level   INTEGER NOT NULL DEFAULT 0,
            times_reviewed  INTEGER NOT NULL DEFAULT 0,
            times_correct   INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES tutor_sessions(id)
        )
        """
    )

    # Migration: add target_lang column if missing (existing databases)
    try:
        await db.execute("ALTER TABLE vocabulary ADD COLUMN target_lang TEXT NOT NULL DEFAULT ''")
        # Backfill from session's target_lang
        await db.execute(
            "UPDATE vocabulary SET target_lang = "
            "(SELECT target_lang FROM tutor_sessions WHERE tutor_sessions.id = vocabulary.session_id) "
            "WHERE target_lang = ''"
        )
        await db.commit()
    except Exception:
        pass  # column already exists

    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS lesson_plans (
            id            TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL DEFAULT '',
            session_id    TEXT NOT NULL,
            title         TEXT NOT NULL,
            topic         TEXT NOT NULL,
            lesson_type   TEXT NOT NULL DEFAULT 'mixed',
            target_lang   TEXT NOT NULL,
            level         TEXT NOT NULL DEFAULT 'A1',
            content       TEXT NOT NULL DEFAULT '{}',
            created_at    TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES tutor_sessions(id)
        )
        """
    )

    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_results (
            id            TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL DEFAULT '',
            session_id    TEXT NOT NULL,
            lesson_id     TEXT DEFAULT NULL,
            quiz_type     TEXT NOT NULL DEFAULT 'mixed',
            questions     TEXT NOT NULL DEFAULT '[]',
            answers       TEXT NOT NULL DEFAULT '[]',
            score         INTEGER NOT NULL DEFAULT 0,
            total         INTEGER NOT NULL DEFAULT 0,
            feedback      TEXT NOT NULL DEFAULT '',
            created_at    TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES tutor_sessions(id)
        )
        """
    )

    await db.commit()
    logger.info("Tutor tables initialised")


# ---------------------------------------------------------------------------
# Tutor sessions
# ---------------------------------------------------------------------------


async def create_tutor_session(
    db: aiosqlite.Connection,
    user_id: str,
    name: str,
    target_lang: str,
    native_lang: str = "English",
    level: str = "A1",
) -> dict:
    sid = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO tutor_sessions "
        "(id, user_id, name, target_lang, native_lang, level, created_at, updated_at, messages) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, user_id, name, target_lang, native_lang, level, now, now, "[]"),
    )
    await db.commit()
    return {
        "id": sid,
        "name": name,
        "target_lang": target_lang,
        "native_lang": native_lang,
        "level": level,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


async def list_tutor_sessions(db: aiosqlite.Connection, user_id: str) -> list[dict]:
    cursor = await db.execute(
        "SELECT id, name, target_lang, native_lang, level, created_at, updated_at, messages "
        "FROM tutor_sessions WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    )
    rows = await cursor.fetchall()
    sessions = []
    for row in rows:
        try:
            msgs = json.loads(row[7])
        except (json.JSONDecodeError, TypeError):
            msgs = []
        sessions.append(
            {
                "id": row[0],
                "name": row[1],
                "target_lang": row[2],
                "native_lang": row[3],
                "level": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "message_count": len(msgs),
            }
        )
    return sessions


async def get_tutor_session(
    db: aiosqlite.Connection, session_id: str, user_id: str
) -> dict | None:
    cursor = await db.execute(
        "SELECT id, user_id, name, target_lang, native_lang, level, "
        "created_at, updated_at, messages FROM tutor_sessions WHERE id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    owner = row[1] or ""
    if owner and owner != user_id:
        return None  # not your session
    try:
        msgs = json.loads(row[8])
    except (json.JSONDecodeError, TypeError):
        msgs = []
    return {
        "id": row[0],
        "user_id": row[1],
        "name": row[2],
        "target_lang": row[3],
        "native_lang": row[4],
        "level": row[5],
        "created_at": row[6],
        "updated_at": row[7],
        "messages": msgs,
    }


async def update_tutor_session_messages(
    db: aiosqlite.Connection, session_id: str, messages: list
):
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "UPDATE tutor_sessions SET messages = ?, updated_at = ? WHERE id = ?",
        (json.dumps(messages), now, session_id),
    )
    await db.commit()


async def delete_tutor_session(db: aiosqlite.Connection, session_id: str, user_id: str) -> bool:
    cursor = await db.execute(
        "SELECT id, user_id FROM tutor_sessions WHERE id = ?", (session_id,)
    )
    row = await cursor.fetchone()
    if not row:
        return False
    owner = row[1] or ""
    if owner and owner != user_id:
        return False
    await db.execute("DELETE FROM vocabulary WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM lesson_plans WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM quiz_results WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM tutor_sessions WHERE id = ?", (session_id,))
    await db.commit()
    return True


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


async def add_vocabulary(
    db: aiosqlite.Connection,
    user_id: str,
    session_id: str,
    word: str,
    translation: str,
    context: str = "",
    phonetic: str = "",
    part_of_speech: str = "",
    target_lang: str = "",
) -> dict:
    vid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO vocabulary "
        "(id, user_id, session_id, target_lang, word, translation, context, phonetic, "
        "part_of_speech, mastery_level, times_reviewed, times_correct, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?)",
        (vid, user_id, session_id, target_lang, word, translation, context, phonetic, part_of_speech, now),
    )
    await db.commit()
    return {
        "id": vid,
        "word": word,
        "translation": translation,
        "context": context,
        "phonetic": phonetic,
        "part_of_speech": part_of_speech,
        "mastery_level": 0,
        "created_at": now,
    }


async def add_vocabulary_batch(
    db: aiosqlite.Connection,
    user_id: str,
    session_id: str,
    entries: list[dict],
    target_lang: str = "",
) -> list[dict]:
    """Add multiple vocabulary entries at once (e.g. from a lesson)."""
    results = []
    now = datetime.now(timezone.utc).isoformat()
    for e in entries:
        vid = uuid.uuid4().hex[:12]
        await db.execute(
            "INSERT INTO vocabulary "
            "(id, user_id, session_id, target_lang, word, translation, context, phonetic, "
            "part_of_speech, mastery_level, times_reviewed, times_correct, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?)",
            (
                vid,
                user_id,
                session_id,
                target_lang,
                e.get("word", ""),
                e.get("translation", ""),
                e.get("context", ""),
                e.get("phonetic", ""),
                e.get("part_of_speech", ""),
                now,
            ),
        )
        results.append({"id": vid, "word": e.get("word", ""), "translation": e.get("translation", "")})
    await db.commit()
    return results


async def list_vocabulary(
    db: aiosqlite.Connection, user_id: str, session_id: str
) -> list[dict]:
    cursor = await db.execute(
        "SELECT id, word, translation, context, phonetic, part_of_speech, "
        "mastery_level, times_reviewed, times_correct, created_at "
        "FROM vocabulary WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
        (user_id, session_id),
    )
    rows = await cursor.fetchall()
    return [
        {
            "id": r[0],
            "word": r[1],
            "translation": r[2],
            "context": r[3],
            "phonetic": r[4],
            "part_of_speech": r[5],
            "mastery_level": r[6],
            "times_reviewed": r[7],
            "times_correct": r[8],
            "created_at": r[9],
        }
        for r in rows
    ]


async def list_vocabulary_by_lang(
    db: aiosqlite.Connection, user_id: str, target_lang: str
) -> list[dict]:
    """List all vocabulary for a user in a given target language (cross-session)."""
    cursor = await db.execute(
        "SELECT id, word, translation, context, phonetic, part_of_speech, "
        "mastery_level, times_reviewed, times_correct, created_at "
        "FROM vocabulary WHERE user_id = ? AND target_lang = ? ORDER BY created_at DESC",
        (user_id, target_lang),
    )
    rows = await cursor.fetchall()
    return [
        {
            "id": r[0],
            "word": r[1],
            "translation": r[2],
            "context": r[3],
            "phonetic": r[4],
            "part_of_speech": r[5],
            "mastery_level": r[6],
            "times_reviewed": r[7],
            "times_correct": r[8],
            "created_at": r[9],
        }
        for r in rows
    ]


async def update_vocabulary_mastery(
    db: aiosqlite.Connection, vocab_id: str, correct: bool
):
    """Update mastery after a quiz answer."""
    await db.execute(
        "UPDATE vocabulary SET times_reviewed = times_reviewed + 1"
        + (", times_correct = times_correct + 1" if correct else "")
        + ", mastery_level = CASE "
        "WHEN ? THEN MIN(mastery_level + 1, 5) "
        "ELSE MAX(mastery_level - 1, 0) END "
        "WHERE id = ?",
        (correct, vocab_id),
    )
    await db.commit()


async def delete_vocabulary(db: aiosqlite.Connection, vocab_id: str, user_id: str) -> bool:
    cursor = await db.execute(
        "SELECT id FROM vocabulary WHERE id = ? AND user_id = ?", (vocab_id, user_id)
    )
    if not await cursor.fetchone():
        return False
    await db.execute("DELETE FROM vocabulary WHERE id = ?", (vocab_id,))
    await db.commit()
    return True


# ---------------------------------------------------------------------------
# Lesson plans
# ---------------------------------------------------------------------------


async def save_lesson_plan(
    db: aiosqlite.Connection,
    user_id: str,
    session_id: str,
    title: str,
    topic: str,
    lesson_type: str,
    target_lang: str,
    level: str,
    content: dict,
) -> dict:
    lid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO lesson_plans "
        "(id, user_id, session_id, title, topic, lesson_type, target_lang, level, content, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (lid, user_id, session_id, title, topic, lesson_type, target_lang, level, json.dumps(content), now),
    )
    await db.commit()
    return {"id": lid, "title": title, "topic": topic, "created_at": now}


async def list_lesson_plans(
    db: aiosqlite.Connection, user_id: str, session_id: str
) -> list[dict]:
    cursor = await db.execute(
        "SELECT id, title, topic, lesson_type, target_lang, level, created_at "
        "FROM lesson_plans WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
        (user_id, session_id),
    )
    rows = await cursor.fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "topic": r[2],
            "lesson_type": r[3],
            "target_lang": r[4],
            "level": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


async def get_lesson_plan(
    db: aiosqlite.Connection, lesson_id: str, user_id: str
) -> dict | None:
    cursor = await db.execute(
        "SELECT id, session_id, title, topic, lesson_type, target_lang, level, content, created_at "
        "FROM lesson_plans WHERE id = ? AND user_id = ?",
        (lesson_id, user_id),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    try:
        content = json.loads(row[7])
    except (json.JSONDecodeError, TypeError):
        content = {}
    return {
        "id": row[0],
        "session_id": row[1],
        "title": row[2],
        "topic": row[3],
        "lesson_type": row[4],
        "target_lang": row[5],
        "level": row[6],
        "content": content,
        "created_at": row[8],
    }


# ---------------------------------------------------------------------------
# Quiz results
# ---------------------------------------------------------------------------


async def save_quiz(
    db: aiosqlite.Connection,
    user_id: str,
    session_id: str,
    quiz_type: str,
    questions: list[dict],
    lesson_id: str | None = None,
) -> dict:
    """Save a generated quiz (before answers are submitted)."""
    qid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO quiz_results "
        "(id, user_id, session_id, lesson_id, quiz_type, questions, answers, "
        "score, total, feedback, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, '[]', 0, ?, '', ?)",
        (qid, user_id, session_id, lesson_id, quiz_type, json.dumps(questions), len(questions), now),
    )
    await db.commit()
    return {"id": qid, "quiz_type": quiz_type, "total": len(questions), "created_at": now}


async def get_quiz(
    db: aiosqlite.Connection, quiz_id: str, user_id: str
) -> dict | None:
    cursor = await db.execute(
        "SELECT id, session_id, lesson_id, quiz_type, questions, answers, "
        "score, total, feedback, created_at "
        "FROM quiz_results WHERE id = ? AND user_id = ?",
        (quiz_id, user_id),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    try:
        questions = json.loads(row[4])
    except (json.JSONDecodeError, TypeError):
        questions = []
    try:
        answers = json.loads(row[5])
    except (json.JSONDecodeError, TypeError):
        answers = []
    return {
        "id": row[0],
        "session_id": row[1],
        "lesson_id": row[2],
        "quiz_type": row[3],
        "questions": questions,
        "answers": answers,
        "score": row[6],
        "total": row[7],
        "feedback": row[8],
        "created_at": row[9],
    }


async def save_quiz_results(
    db: aiosqlite.Connection,
    quiz_id: str,
    user_id: str,
    answers: list[dict],
    score: int,
    feedback: str,
):
    """Save graded quiz answers."""
    await db.execute(
        "UPDATE quiz_results SET answers = ?, score = ?, feedback = ? "
        "WHERE id = ? AND user_id = ?",
        (json.dumps(answers), score, feedback, quiz_id, user_id),
    )
    await db.commit()


async def list_quiz_results(
    db: aiosqlite.Connection, user_id: str, session_id: str
) -> list[dict]:
    cursor = await db.execute(
        "SELECT id, lesson_id, quiz_type, score, total, created_at "
        "FROM quiz_results WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
        (user_id, session_id),
    )
    rows = await cursor.fetchall()
    return [
        {
            "id": r[0],
            "lesson_id": r[1],
            "quiz_type": r[2],
            "score": r[3],
            "total": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


async def get_student_stats(db: aiosqlite.Connection, user_id: str, session_id: str) -> dict:
    """Aggregate statistics for appraisal."""
    # Vocabulary stats
    cursor = await db.execute(
        "SELECT COUNT(*), AVG(mastery_level), SUM(times_reviewed), SUM(times_correct) "
        "FROM vocabulary WHERE user_id = ? AND session_id = ?",
        (user_id, session_id),
    )
    vrow = await cursor.fetchone()
    vocab_count = vrow[0] or 0
    avg_mastery = round(vrow[1] or 0, 2)
    total_reviewed = vrow[2] or 0
    total_correct = vrow[3] or 0

    # Quiz stats
    cursor = await db.execute(
        "SELECT COUNT(*), SUM(score), SUM(total) "
        "FROM quiz_results WHERE user_id = ? AND session_id = ? AND score > 0",
        (user_id, session_id),
    )
    qrow = await cursor.fetchone()
    quiz_count = qrow[0] or 0
    quiz_score_sum = qrow[1] or 0
    quiz_total_sum = qrow[2] or 0

    # Lesson count
    cursor = await db.execute(
        "SELECT COUNT(*) FROM lesson_plans WHERE user_id = ? AND session_id = ?",
        (user_id, session_id),
    )
    lrow = await cursor.fetchone()
    lesson_count = lrow[0] or 0

    # Message count
    cursor = await db.execute(
        "SELECT messages FROM tutor_sessions WHERE id = ? AND user_id = ?",
        (session_id, user_id),
    )
    mrow = await cursor.fetchone()
    message_count = 0
    if mrow:
        try:
            msgs = json.loads(mrow[0])
            message_count = len(msgs)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "vocabulary": {
            "total_words": vocab_count,
            "avg_mastery": avg_mastery,
            "total_reviewed": total_reviewed,
            "total_correct": total_correct,
            "accuracy": round(total_correct / total_reviewed * 100, 1) if total_reviewed > 0 else 0,
        },
        "quizzes": {
            "total_taken": quiz_count,
            "total_score": quiz_score_sum,
            "total_questions": quiz_total_sum,
            "avg_score": round(quiz_score_sum / quiz_total_sum * 100, 1) if quiz_total_sum > 0 else 0,
        },
        "lessons_completed": lesson_count,
        "total_interactions": message_count,
    }


async def get_cross_session_stats(
    db: aiosqlite.Connection, user_id: str, target_lang: str
) -> dict:
    """Aggregate statistics across ALL sessions for a given target language."""
    # Find all session IDs for this language
    cursor = await db.execute(
        "SELECT id, name, level, messages, created_at, updated_at "
        "FROM tutor_sessions WHERE user_id = ? AND target_lang = ? "
        "ORDER BY updated_at DESC",
        (user_id, target_lang),
    )
    rows = await cursor.fetchall()
    session_ids = [r[0] for r in rows]

    if not session_ids:
        return {"sessions": [], "totals": {}}

    placeholders = ",".join("?" * len(session_ids))

    # Vocabulary across all sessions
    cursor = await db.execute(
        f"SELECT COUNT(*), AVG(mastery_level), SUM(times_reviewed), SUM(times_correct) "
        f"FROM vocabulary WHERE user_id = ? AND session_id IN ({placeholders})",
        (user_id, *session_ids),
    )
    vrow = await cursor.fetchone()

    # Quiz across all sessions
    cursor = await db.execute(
        f"SELECT COUNT(*), SUM(score), SUM(total) "
        f"FROM quiz_results WHERE user_id = ? AND session_id IN ({placeholders}) AND score > 0",
        (user_id, *session_ids),
    )
    qrow = await cursor.fetchone()

    # Lesson count across all sessions
    cursor = await db.execute(
        f"SELECT COUNT(*) FROM lesson_plans WHERE user_id = ? AND session_id IN ({placeholders})",
        (user_id, *session_ids),
    )
    lrow = await cursor.fetchone()

    # Per-session summaries
    session_summaries = []
    total_messages = 0
    for r in rows:
        msg_count = 0
        try:
            msgs = json.loads(r[3])
            msg_count = len(msgs)
        except (json.JSONDecodeError, TypeError):
            pass
        total_messages += msg_count
        session_summaries.append({
            "name": r[1],
            "level": r[2],
            "messages": msg_count,
            "created": r[4],
            "last_active": r[5],
        })

    vocab_reviewed = vrow[2] or 0
    vocab_correct = vrow[3] or 0
    quiz_total_sum = qrow[2] or 0
    quiz_score_sum = qrow[1] or 0

    return {
        "sessions": session_summaries,
        "totals": {
            "session_count": len(session_ids),
            "vocabulary": {
                "total_words": vrow[0] or 0,
                "avg_mastery": round(vrow[1] or 0, 2),
                "total_reviewed": vocab_reviewed,
                "total_correct": vocab_correct,
                "accuracy": round(vocab_correct / vocab_reviewed * 100, 1) if vocab_reviewed > 0 else 0,
            },
            "quizzes": {
                "total_taken": qrow[0] or 0,
                "total_score": quiz_score_sum,
                "total_questions": quiz_total_sum,
                "avg_score": round(quiz_score_sum / quiz_total_sum * 100, 1) if quiz_total_sum > 0 else 0,
            },
            "lessons_completed": lrow[0] or 0,
            "total_interactions": total_messages,
        },
    }
