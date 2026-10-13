"""Database operations for the Language Tutor.

Manages tutor sessions (separate from regular chat sessions),
vocabulary entries, lesson plans, and quiz results in SQLite.
"""

import os
import re
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta

import aiosqlite

SRS_INITIAL_EASE = float(os.environ.get("SRS_INITIAL_EASE", "2.5"))
# Scheduler: "fsrs" (default) or "sm2" to fall back to the pre-2.6 algorithm.
SRS_ALGO = os.environ.get("SRS_ALGO", "fsrs").strip().lower()
# Target probability of recall at review time. Lower = fewer, longer-spaced
# reviews; higher = more frequent. 0.9 is the FSRS default.
SRS_DESIRED_RETENTION = float(os.environ.get("SRS_DESIRED_RETENTION", "0.9"))
# Ceiling on any interval. FSRS's own default is 36500 days, which is meaningless
# for language study — a word you next see in 11 years is not being learned.
SRS_MAX_INTERVAL = int(os.environ.get("SRS_MAX_INTERVAL", "365"))

logger = logging.getLogger(__name__)


def get_srs_params() -> dict:
    """Current spaced-repetition settings, for the Params panel."""
    return {
        "srs_algo": SRS_ALGO,
        "srs_desired_retention": SRS_DESIRED_RETENTION,
        "srs_max_interval": SRS_MAX_INTERVAL,
        "srs_initial_ease": SRS_INITIAL_EASE,
    }


def update_srs_params(params: dict) -> dict:
    """Update spaced-repetition settings at runtime."""
    global SRS_ALGO, SRS_DESIRED_RETENTION, SRS_MAX_INTERVAL, SRS_INITIAL_EASE

    if "srs_algo" in params:
        val = str(params["srs_algo"]).lower().strip()
        if val in ("fsrs", "sm2"):
            SRS_ALGO = val
    if "srs_desired_retention" in params:
        # Below ~0.7 scheduling becomes erratic; above 0.97 reviews explode.
        SRS_DESIRED_RETENTION = max(0.7, min(0.97, float(params["srs_desired_retention"])))
    if "srs_max_interval" in params:
        SRS_MAX_INTERVAL = max(1, min(36500, int(params["srs_max_interval"])))
    if "srs_initial_ease" in params:
        SRS_INITIAL_EASE = max(1.3, min(4.0, float(params["srs_initial_ease"])))
    return get_srs_params()


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------


async def init_tutor_tables(db: aiosqlite.Connection):
    """Create all tutor-related tables if they don't exist."""

    await db.execute("""
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
        """)

    await db.execute("""
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
        """)

    # Migration: add target_lang column if missing (existing databases)
    try:
        await db.execute(
            "ALTER TABLE vocabulary ADD COLUMN target_lang TEXT NOT NULL DEFAULT ''"
        )
        # Backfill from session's target_lang
        await db.execute(
            "UPDATE vocabulary SET target_lang = "
            "(SELECT target_lang FROM tutor_sessions WHERE tutor_sessions.id = vocabulary.session_id) "
            "WHERE target_lang = ''"
        )
        await db.commit()
    except Exception:
        pass  # column already exists

    # Migration: add example and conjugation columns if missing
    for col_sql in (
        "ALTER TABLE vocabulary ADD COLUMN example     TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE vocabulary ADD COLUMN conjugation TEXT NOT NULL DEFAULT ''",
    ):
        try:
            await db.execute(col_sql)
            await db.commit()
        except Exception:
            pass  # column already exists

    # Migration: add SM-2 SRS columns if missing
    for col_sql in (
        "ALTER TABLE vocabulary ADD COLUMN next_review  TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE vocabulary ADD COLUMN interval_days INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE vocabulary ADD COLUMN ease_factor   REAL    NOT NULL DEFAULT 2.5",
        "ALTER TABLE vocabulary ADD COLUMN lapses        INTEGER NOT NULL DEFAULT 0",
        # FSRS state. NULL stability means "not yet migrated from SM-2" — the
        # scheduler seeds it from interval_days/ease_factor on first review so
        # existing cards keep their place instead of restarting.
        "ALTER TABLE vocabulary ADD COLUMN fsrs_stability  REAL",
        "ALTER TABLE vocabulary ADD COLUMN fsrs_difficulty REAL",
        "ALTER TABLE vocabulary ADD COLUMN fsrs_state      INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE vocabulary ADD COLUMN fsrs_step       INTEGER",
        "ALTER TABLE vocabulary ADD COLUMN last_review     TEXT",
    ):
        try:
            await db.execute(col_sql)
            await db.commit()
        except Exception:
            pass  # column already exists

    # Migration: deduplicate vocabulary and add unique constraint
    # Keep only the row with highest mastery_level per (user_id, target_lang, word)
    try:
        await db.execute("""
            DELETE FROM vocabulary WHERE id NOT IN (
                SELECT id FROM vocabulary v1
                WHERE v1.rowid = (
                    SELECT MAX(v2.rowid) FROM vocabulary v2
                    WHERE v2.user_id = v1.user_id
                      AND v2.target_lang = v1.target_lang
                      AND LOWER(v2.word) = LOWER(v1.word)
                )
            )
        """)
        await db.commit()
    except Exception:
        pass
    try:
        await db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_vocab_user_lang_word "
            "ON vocabulary (user_id, target_lang, LOWER(word))"
        )
        await db.commit()
    except Exception:
        pass

    # Backfill next_review for existing rows that have no value
    try:
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
        await db.execute(
            "UPDATE vocabulary SET next_review = ? WHERE next_review = ''",
            (tomorrow,),
        )
        await db.commit()
    except Exception:
        pass

    await db.execute("""
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
        """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS passages (
            id             TEXT PRIMARY KEY,
            user_id        TEXT NOT NULL DEFAULT '',
            session_id     TEXT NOT NULL,
            title          TEXT NOT NULL DEFAULT '',
            topic          TEXT NOT NULL DEFAULT '',
            target_lang    TEXT NOT NULL DEFAULT '',
            level          TEXT NOT NULL DEFAULT 'A1',
            content        TEXT NOT NULL DEFAULT '',
            seeded_words   TEXT NOT NULL DEFAULT '[]',
            word_count     INTEGER NOT NULL DEFAULT 0,
            unknown_ratio  REAL    NOT NULL DEFAULT 0.0,
            created_at     TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES tutor_sessions(id)
        )
        """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id            TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL DEFAULT '',
            session_id    TEXT NOT NULL,
            lesson_id     TEXT DEFAULT NULL,
            quiz_type     TEXT NOT NULL DEFAULT 'mixed',
            questions     TEXT NOT NULL DEFAULT '[]',
            answers       TEXT NOT NULL DEFAULT '[]',
            score         INTEGER NOT NULL DEFAULT 0,
            score_float   REAL    NOT NULL DEFAULT 0.0,
            total         INTEGER NOT NULL DEFAULT 0,
            feedback      TEXT NOT NULL DEFAULT '',
            created_at    TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES tutor_sessions(id)
        )
        """)

    # Migration: add score_float column if missing (existing databases)
    try:
        await db.execute(
            "ALTER TABLE quiz_results ADD COLUMN score_float REAL NOT NULL DEFAULT 0.0"
        )
        await db.commit()
    except Exception:
        pass  # column already exists

    # Migration: add topic_index column for curriculum tracking
    try:
        await db.execute(
            "ALTER TABLE tutor_sessions ADD COLUMN topic_index INTEGER NOT NULL DEFAULT 0"
        )
        await db.commit()
    except Exception:
        pass  # column already exists

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
        "created_at, updated_at, messages, topic_index FROM tutor_sessions WHERE id = ?",
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
        "topic_index": row[9] or 0,
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


async def set_topic_index(
    db: aiosqlite.Connection, session_id: str, index: int
):
    await db.execute(
        "UPDATE tutor_sessions SET topic_index = ? WHERE id = ?",
        (max(0, index), session_id),
    )
    await db.commit()


async def delete_tutor_session(
    db: aiosqlite.Connection, session_id: str, user_id: str
) -> bool:
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
    example: str = "",
    conjugation: str = "",
) -> dict:
    vid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    await db.execute(
        "INSERT INTO vocabulary "
        "(id, user_id, session_id, target_lang, word, translation, context, phonetic, "
        "part_of_speech, example, conjugation, mastery_level, times_reviewed, times_correct, "
        "next_review, interval_days, ease_factor, lapses, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, 1, ?, 0, ?) "
        "ON CONFLICT (user_id, target_lang, LOWER(word)) DO NOTHING",
        (
            vid, user_id, session_id, target_lang, word, translation,
            context, phonetic, part_of_speech, example, conjugation,
            tomorrow, SRS_INITIAL_EASE, now,
        ),
    )
    await db.commit()
    return {
        "id": vid,
        "word": word,
        "translation": translation,
        "context": context,
        "phonetic": phonetic,
        "part_of_speech": part_of_speech,
        "example": example,
        "conjugation": conjugation,
        "mastery_level": 0,
        "next_review": tomorrow,
        "interval_days": 1,
        "ease_factor": SRS_INITIAL_EASE,
        "lapses": 0,
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
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    for e in entries:
        vid = uuid.uuid4().hex[:12]
        await db.execute(
            "INSERT INTO vocabulary "
            "(id, user_id, session_id, target_lang, word, translation, context, phonetic, "
            "part_of_speech, example, conjugation, mastery_level, times_reviewed, times_correct, "
            "next_review, interval_days, ease_factor, lapses, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, 1, ?, 0, ?) "
            "ON CONFLICT (user_id, target_lang, LOWER(word)) DO UPDATE SET "
            "translation = excluded.translation, "
            "part_of_speech = CASE WHEN excluded.part_of_speech != '' THEN excluded.part_of_speech ELSE part_of_speech END, "
            "example = CASE WHEN excluded.example != '' THEN excluded.example ELSE example END, "
            "conjugation = CASE WHEN excluded.conjugation != '' THEN excluded.conjugation ELSE conjugation END",
            (
                vid, user_id, session_id, target_lang,
                e.get("word", ""), e.get("translation", ""),
                e.get("context", ""), e.get("phonetic", ""),
                e.get("part_of_speech", ""), e.get("example", ""),
                e.get("conjugation", ""),
                tomorrow, SRS_INITIAL_EASE, now,
            ),
        )
        results.append(
            {
                "id": vid,
                "word": e.get("word", ""),
                "translation": e.get("translation", ""),
            }
        )
    await db.commit()
    return results


def _vocab_row_to_dict(r) -> dict:
    return {
        "id": r[0],
        "word": r[1],
        "translation": r[2],
        "context": r[3],
        "phonetic": r[4],
        "part_of_speech": r[5],
        "example": r[6],
        "conjugation": r[7],
        "mastery_level": r[8],
        "times_reviewed": r[9],
        "times_correct": r[10],
        "next_review": r[11] or "",
        "interval_days": r[12] or 1,
        "ease_factor": r[13] or SRS_INITIAL_EASE,
        "lapses": r[14] or 0,
        "created_at": r[15],
        "difficult": (r[14] or 0) >= 3,
    }


_VOCAB_SELECT = (
    "SELECT id, word, translation, context, phonetic, part_of_speech, "
    "example, conjugation, mastery_level, times_reviewed, times_correct, "
    "next_review, interval_days, ease_factor, lapses, created_at "
)


async def list_vocabulary(
    db: aiosqlite.Connection, user_id: str, session_id: str
) -> list[dict]:
    cursor = await db.execute(
        _VOCAB_SELECT
        + "FROM vocabulary WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
        (user_id, session_id),
    )
    rows = await cursor.fetchall()
    return [_vocab_row_to_dict(r) for r in rows]


async def list_vocabulary_by_lang(
    db: aiosqlite.Connection, user_id: str, target_lang: str
) -> list[dict]:
    """List all vocabulary for a user in a given target language (cross-session)."""
    cursor = await db.execute(
        _VOCAB_SELECT
        + "FROM vocabulary WHERE user_id = ? AND target_lang = ? ORDER BY created_at DESC",
        (user_id, target_lang),
    )
    rows = await cursor.fetchall()
    return [_vocab_row_to_dict(r) for r in rows]


async def update_vocabulary_mastery(
    db: aiosqlite.Connection, vocab_id: str, correct: bool
):
    """Update mastery and SM-2 schedule after a quiz answer."""
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
    grade = 5 if correct else 1
    await schedule_review(db, vocab_id, grade)


def _sm2_compute(
    interval_days: int, ease_factor: float, times_correct: int, grade: int
) -> tuple[int, float]:
    """Compute next SM-2 interval and ease_factor.

    grade: 0-5  (0=total blackout, 2=incorrect but recalled, 3=correct with difficulty,
                 5=perfect recall)
    Returns (new_interval_days, new_ease_factor).
    Grades 0-2 are lapses: interval resets to 1.
    Grades 3-5 progress normally.
    """
    is_lapse = grade < 3
    if is_lapse:
        new_interval = 1
    else:
        if times_correct <= 0:
            new_interval = 1
        elif times_correct == 1:
            new_interval = 6
        else:
            new_interval = max(1, round(interval_days * ease_factor))

    new_ef = ease_factor + 0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)
    new_ef = max(1.3, round(new_ef, 3))
    return new_interval, new_ef, is_lapse


def _grade_to_rating(grade: int):
    """Map an SM-2 0–5 grade onto an FSRS rating.

    Keeps the existing call sites (and the quiz grader's 0–5 vocabulary) intact
    while the scheduler underneath changes.
    """
    from fsrs import Rating

    if grade < 3:
        return Rating.Again
    if grade == 3:
        return Rating.Hard
    if grade == 4:
        return Rating.Good
    return Rating.Easy


def _seed_fsrs_from_sm2(interval_days: int, ease_factor: float, lapses: int) -> tuple:
    """Derive initial FSRS (stability, difficulty) from accumulated SM-2 state.

    Without this, switching schedulers would reset every card a learner has
    already driven out to a long interval, forcing them to re-review words they
    know well.

    * stability ≈ the current interval — both express "days until recall
      probability decays to ~90%".
    * difficulty is FSRS's 1 (easiest) – 10 (hardest) scale, mapped from SM-2's
      inverted ease factor, where 1.3 is hardest and ~3.3 easiest. Lapses nudge
      it upward, since a lapsed card is by definition harder than its ease says.
    """
    stability = max(0.5, float(interval_days or 1))
    ease = float(ease_factor or SRS_INITIAL_EASE)
    difficulty = 10.0 - ((ease - 1.3) / 2.0) * 9.0
    difficulty += min(2.0, 0.25 * int(lapses or 0))
    difficulty = max(1.0, min(10.0, round(difficulty, 4)))
    return stability, difficulty


async def fsrs_review(db: aiosqlite.Connection, vocab_id: str, grade: int):
    """Schedule the next review with FSRS.

    FSRS tracks difficulty, stability and retrievability per card, fitted to
    hundreds of millions of real reviews. It needs roughly 20–30% fewer reviews
    than SM-2 for the same retention.

    interval_days and ease_factor are still written so the UI, the stats
    endpoint and a rollback to SRS_ALGO=sm2 all keep working.
    """
    from fsrs import Card, Scheduler, State

    cursor = await db.execute(
        "SELECT interval_days, ease_factor, lapses, fsrs_stability, "
        "fsrs_difficulty, fsrs_state, fsrs_step, last_review, times_reviewed "
        "FROM vocabulary WHERE id = ?",
        (vocab_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return

    (interval_days, ease_factor, lapses, stability, difficulty,
     state, step, last_review, times_reviewed) = row

    if stability is None:
        # First review under FSRS: carry the card's SM-2 history across.
        if (times_reviewed or 0) > 0:
            stability, difficulty = _seed_fsrs_from_sm2(
                interval_days, ease_factor, lapses
            )
            state = State.Review.value
            step = None
            # FSRS derives retrievability from time since last_review. Leaving it
            # NULL means "zero elapsed time", which makes a long-interval card
            # look perfectly retained and sends the next interval to ~5000 days.
            # The card was in fact last seen one interval ago, so say so.
            if not last_review:
                last_review = (
                    datetime.now(timezone.utc)
                    - timedelta(days=max(1, int(interval_days or 1)))
                ).isoformat()
        else:
            stability = difficulty = None  # genuinely new card

    card = Card(
        card_id=int(datetime.now(timezone.utc).timestamp() * 1000),
        state=State(state or State.Learning.value),
        step=step,
        stability=stability,
        difficulty=difficulty,
        last_review=datetime.fromisoformat(last_review) if last_review else None,
    )

    now = datetime.now(timezone.utc)
    scheduler = Scheduler(
        desired_retention=SRS_DESIRED_RETENTION,
        maximum_interval=SRS_MAX_INTERVAL,
    )
    card, _log = scheduler.review_card(card, _grade_to_rating(grade), now)

    new_interval = max(1, (card.due.date() - now.date()).days)
    is_lapse = grade < 3
    await db.execute(
        "UPDATE vocabulary SET next_review = ?, interval_days = ?, "
        "fsrs_stability = ?, fsrs_difficulty = ?, fsrs_state = ?, fsrs_step = ?, "
        "last_review = ?, lapses = lapses + ? WHERE id = ?",
        (
            card.due.date().isoformat(), new_interval,
            card.stability, card.difficulty, card.state.value, card.step,
            now.isoformat(), 1 if is_lapse else 0, vocab_id,
        ),
    )
    await db.commit()


async def schedule_review(db: aiosqlite.Connection, vocab_id: str, grade: int):
    """Schedule the next review using the configured algorithm.

    SRS_ALGO=sm2 restores the previous scheduler; FSRS is the default.
    """
    if SRS_ALGO == "sm2":
        await sm2_review(db, vocab_id, grade)
    else:
        await fsrs_review(db, vocab_id, grade)


async def sm2_review(db: aiosqlite.Connection, vocab_id: str, grade: int):
    """Apply the SM-2 algorithm to schedule the next review for a vocabulary item.

    grade: 0=blackout, 1-2=fail, 3=pass with effort, 4=correct, 5=perfect.
    Updates next_review, interval_days, ease_factor, lapses in the vocabulary row.
    """
    cursor = await db.execute(
        "SELECT interval_days, ease_factor, times_correct FROM vocabulary WHERE id = ?",
        (vocab_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return
    interval_days = row[0] or 1
    ease_factor = row[1] or SRS_INITIAL_EASE
    times_correct = row[2] or 0

    new_interval, new_ef, is_lapse = _sm2_compute(
        interval_days, ease_factor, times_correct, grade
    )
    next_review = (
        (datetime.now(timezone.utc) + timedelta(days=new_interval)).date().isoformat()
    )

    lapse_inc = 1 if is_lapse else 0
    await db.execute(
        "UPDATE vocabulary SET next_review = ?, interval_days = ?, ease_factor = ?, "
        "lapses = lapses + ? WHERE id = ?",
        (next_review, new_interval, new_ef, lapse_inc, vocab_id),
    )
    await db.commit()


async def list_vocabulary_due(
    db: aiosqlite.Connection, user_id: str, target_lang: str = ""
) -> list[dict]:
    """Return vocabulary items due for review today or past-due.

    If target_lang is given, filter to that language; otherwise return all languages.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    if target_lang:
        cursor = await db.execute(
            _VOCAB_SELECT
            + "FROM vocabulary "
            "WHERE user_id = ? AND target_lang = ? AND next_review != '' AND next_review <= ? "
            "ORDER BY next_review ASC",
            (user_id, target_lang, today),
        )
    else:
        cursor = await db.execute(
            _VOCAB_SELECT
            + "FROM vocabulary "
            "WHERE user_id = ? AND next_review != '' AND next_review <= ? "
            "ORDER BY next_review ASC",
            (user_id, today),
        )
    rows = await cursor.fetchall()
    return [_vocab_row_to_dict(r) for r in rows]


async def delete_vocabulary(
    db: aiosqlite.Connection, vocab_id: str, user_id: str
) -> bool:
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
        (
            lid,
            user_id,
            session_id,
            title,
            topic,
            lesson_type,
            target_lang,
            level,
            json.dumps(content),
            now,
        ),
    )
    await db.commit()
    return {"id": lid, "title": title, "topic": topic, "created_at": now}


async def list_all_lesson_plans(
    db: aiosqlite.Connection, user_id: str, limit: int = 200
) -> list[dict]:
    """Every lesson this user has saved, across all sessions.

    The printable-sheets panel is not tied to a tutor session, so it needs the
    whole list; list_lesson_plans is per-session and cannot serve it.
    """
    cursor = await db.execute(
        "SELECT id, title, topic, lesson_type, target_lang, level, created_at "
        "FROM lesson_plans WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, int(limit)),
    )
    return [
        {"id": r[0], "title": r[1], "topic": r[2], "lesson_type": r[3],
         "target_lang": r[4], "level": r[5], "created_at": r[6]}
        for r in await cursor.fetchall()
    ]


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
        (
            qid,
            user_id,
            session_id,
            lesson_id,
            quiz_type,
            json.dumps(questions),
            len(questions),
            now,
        ),
    )
    await db.commit()
    return {
        "id": qid,
        "quiz_type": quiz_type,
        "total": len(questions),
        "created_at": now,
    }


async def get_quiz(db: aiosqlite.Connection, quiz_id: str, user_id: str) -> dict | None:
    cursor = await db.execute(
        "SELECT id, session_id, lesson_id, quiz_type, questions, answers, "
        "score, score_float, total, feedback, created_at "
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
        "score_float": row[7],
        "total": row[8],
        "feedback": row[9],
        "created_at": row[10],
    }


async def save_quiz_results(
    db: aiosqlite.Connection,
    quiz_id: str,
    user_id: str,
    answers: list[dict],
    score: int,
    feedback: str,
    score_float: float | None = None,
):
    """Save graded quiz answers.

    ``score_float`` is the sum of per-question float scores (0.0–1.0 each).
    If omitted, it defaults to ``score`` (integer, backward-compatible).
    """
    if score_float is None:
        score_float = float(score)
    await db.execute(
        "UPDATE quiz_results SET answers = ?, score = ?, score_float = ?, feedback = ? "
        "WHERE id = ? AND user_id = ?",
        (json.dumps(answers), score, score_float, feedback, quiz_id, user_id),
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


async def list_quiz_results_by_lang(
    db: aiosqlite.Connection, user_id: str, target_lang: str
) -> list[dict]:
    """List all quiz results for a user in a given target language (cross-session)."""
    # First, get all session IDs for this language
    cursor = await db.execute(
        "SELECT id FROM tutor_sessions WHERE user_id = ? AND target_lang = ?",
        (user_id, target_lang),
    )
    rows = await cursor.fetchall()
    session_ids = [r[0] for r in rows]

    if not session_ids:
        return []

    placeholders = ",".join("?" * len(session_ids))

    cursor = await db.execute(
        f"SELECT id, session_id, lesson_id, quiz_type, score, total, created_at "
        f"FROM quiz_results WHERE user_id = ? AND session_id IN ({placeholders}) "
        f"ORDER BY created_at DESC",
        (user_id, *session_ids),
    )
    rows = await cursor.fetchall()
    return [
        {
            "id": r[0],
            "session_id": r[1],
            "lesson_id": r[2],
            "quiz_type": r[3],
            "score": r[4],
            "total": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


async def get_student_stats(
    db: aiosqlite.Connection, user_id: str, session_id: str
) -> dict:
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
            "accuracy": (
                round(total_correct / total_reviewed * 100, 1)
                if total_reviewed > 0
                else 0
            ),
        },
        "quizzes": {
            "total_taken": quiz_count,
            "total_score": quiz_score_sum,
            "total_questions": quiz_total_sum,
            "avg_score": (
                round(quiz_score_sum / quiz_total_sum * 100, 1)
                if quiz_total_sum > 0
                else 0
            ),
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
        session_summaries.append(
            {
                "name": r[1],
                "level": r[2],
                "messages": msg_count,
                "created": r[4],
                "last_active": r[5],
            }
        )

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
                "accuracy": (
                    round(vocab_correct / vocab_reviewed * 100, 1)
                    if vocab_reviewed > 0
                    else 0
                ),
            },
            "quizzes": {
                "total_taken": qrow[0] or 0,
                "total_score": quiz_score_sum,
                "total_questions": quiz_total_sum,
                "avg_score": (
                    round(quiz_score_sum / quiz_total_sum * 100, 1)
                    if quiz_total_sum > 0
                    else 0
                ),
            },
            "lessons_completed": lrow[0] or 0,
            "total_interactions": total_messages,
        },
    }


# ---------------------------------------------------------------------------
# Reading passages (comprehensible input)
# ---------------------------------------------------------------------------

# Word characters across Latin, Cyrillic, Greek and the common diacritics. Kept
# deliberately simple: this measures *coverage*, not linguistics, so lemmatising
# would add a dependency for little gain at these sample sizes.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokenise_words(text: str) -> list[str]:
    """Lowercased word tokens, punctuation and digits removed."""
    return [w.lower() for w in _WORD_RE.findall(text or "")]


# Function words — articles, pronouns, prepositions, conjunctions, auxiliaries.
# These are acquired almost immediately and are not "vocabulary to learn", so
# counting them as unknown makes the i+1 ratio meaningless for any learner whose
# saved word list is short (i.e. every beginner, which is who this is for).
_FUNCTION_WORDS: dict[str, set[str]] = {
    "french": set("""le la les un une des du de d au aux a à et ou mais donc or ni car
        je tu il elle on nous vous ils elles me te se lui leur y en
        ce cet cette ces son sa ses mon ma mes ton ta tes notre votre leurs
        qui que quoi dont où est sont suis es sommes êtes était étaient être
        ai as a avons avez ont avait avaient avoir fait faire pas ne plus très
        pour par sur sous dans avec sans chez vers entre comme si oui non
        ça cela celui celle tout tous toute toutes bien aussi encore""".split()),
    "spanish": set("""el la los las un una unos unas de del al a y o pero porque
        yo tu él ella nosotros vosotros ellos ellas me te se le les lo nos os
        que quien cual donde es son soy eres somos sois era eran ser estar
        está están he has ha hemos han había habían tener tiene no sí muy
        para por sobre bajo en con sin hacia entre como si su sus mi mis tu tus
        este esta estos estas ese esa eso todo todos toda todas también más""".split()),
    "german": set("""der die das den dem des ein eine einen einem einer eines
        ich du er sie es wir ihr mich dich sich uns euch mir dir ihm ihnen
        und oder aber denn weil dass ob wenn als wie so auch noch nur schon
        ist sind bin bist seid war waren sein habe hast hat haben hatte hatten
        nicht kein keine in an auf für mit von zu bei nach aus über unter vor
        mein dein ihr unser euer alle alles diese dieser dieses sehr""".split()),
    "italian": set("""il lo la i gli le un uno una del della dei delle di a da in
        con su per tra fra e o ma perché che chi cui dove io tu lui lei noi voi
        loro mi ti si ci vi è sono sei siamo siete era erano essere ho hai ha
        abbiamo avete hanno non sì molto anche ancora già questo questa quello
        mio tuo suo nostro vostro tutto tutti come se quando""".split()),
    "portuguese": set("""o a os as um uma uns umas de do da dos das em no na nos nas
        ao aos à às por para com sem sobre e ou mas porque que quem qual onde
        eu tu ele ela nós vós eles elas me te se lhe lhes nos vos é são sou és
        somos era eram ser estar está estão tenho tem temos têm não sim muito
        meu teu seu nosso vosso todo todos toda todas este esta esse essa como se""".split()),
    "english": set("""the a an of to in on at by for with from and or but so if
        i you he she it we they me him her us them my your his its our their
        is are am was were be been being have has had do does did not no yes
        this that these those what which who whom whose where when how very
        as than then there here all some any more most much many""".split()),
    "romanian": set("""un o unui unei niște al a ai ale cel cea cei cele
        eu tu el ea noi voi ei ele mă te se ne vă îl îi le îmi îți îi
        și sau dar că să nu da este sunt sînt era erau fi fost am ai are avem
        aveți au de la în pe cu fără despre pentru prin sub peste între ca dacă
        acest această acești aceste acel acea meu tău său nostru vostru foarte""".split()),
}


def _function_words(target_lang: str) -> set[str]:
    """Function-word set for a language name, empty if we have no list."""
    return _FUNCTION_WORDS.get((target_lang or "").strip().lower(), set())


def compute_unknown_ratio(
    text: str, known_words: set[str], target_lang: str = ""
) -> tuple[float, int, list[str]]:
    """Measure how much of *text* is new to the learner.

    Returns ``(unknown_ratio, word_count, sample_unknown)``.

    Krashen's *i+1* says input should sit just above current ability — in
    practice about 10% unknown. Below that the learner is not stretched; far
    above it, comprehension collapses and the input stops being comprehensible.
    This is what lets the UI tell the learner whether a passage is pitched right.

    Function words are excluded from the measurement. Without that, a beginner
    with a handful of saved words scores ~95% unknown on an A1 passage — the
    articles and pronouns dominate — and the number tells them nothing.
    """
    tokens = tokenise_words(text)
    if not tokens:
        return 0.0, 0, []

    known = {w.lower() for w in known_words}
    stop = _function_words(target_lang)

    content = {t for t in tokens if t not in stop and len(t) > 1}
    if not content:
        return 0.0, len(tokens), []

    # Distinct words: meeting one hard word ten times is one thing to learn.
    unknown = sorted(content - known)
    ratio = len(unknown) / len(content)
    return round(ratio, 4), len(tokens), unknown[:40]


async def known_words_for(
    db: aiosqlite.Connection, user_id: str, target_lang: str
) -> set[str]:
    """Every word the learner has in their vocabulary for this language."""
    cursor = await db.execute(
        "SELECT word FROM vocabulary WHERE user_id = ? AND target_lang = ?",
        (user_id, target_lang),
    )
    rows = await cursor.fetchall()
    out: set[str] = set()
    for r in rows:
        out.update(tokenise_words(r[0]))
    return out


async def save_passage(
    db: aiosqlite.Connection,
    user_id: str,
    session_id: str,
    title: str,
    topic: str,
    target_lang: str,
    level: str,
    content: str,
    seeded_words: list[str],
    word_count: int,
    unknown_ratio: float,
) -> dict:
    pid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO passages (id, user_id, session_id, title, topic, target_lang, "
        "level, content, seeded_words, word_count, unknown_ratio, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (pid, user_id, session_id, title, topic, target_lang, level, content,
         json.dumps(seeded_words), word_count, unknown_ratio, now),
    )
    await db.commit()
    return {
        "id": pid, "session_id": session_id, "title": title, "topic": topic,
        "target_lang": target_lang, "level": level, "content": content,
        "seeded_words": seeded_words, "word_count": word_count,
        "unknown_ratio": unknown_ratio, "created_at": now,
    }


def _passage_row_to_dict(r) -> dict:
    return {
        "id": r[0], "session_id": r[1], "title": r[2], "topic": r[3],
        "target_lang": r[4], "level": r[5], "content": r[6],
        "seeded_words": json.loads(r[7] or "[]"), "word_count": r[8],
        "unknown_ratio": r[9], "created_at": r[10],
    }


_PASSAGE_SELECT = (
    "SELECT id, session_id, title, topic, target_lang, level, content, "
    "seeded_words, word_count, unknown_ratio, created_at "
)


async def list_passages(
    db: aiosqlite.Connection, user_id: str, session_id: str
) -> list[dict]:
    cursor = await db.execute(
        _PASSAGE_SELECT
        + "FROM passages WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
        (user_id, session_id),
    )
    return [_passage_row_to_dict(r) for r in await cursor.fetchall()]


async def get_passage(
    db: aiosqlite.Connection, user_id: str, passage_id: str
) -> dict | None:
    cursor = await db.execute(
        _PASSAGE_SELECT + "FROM passages WHERE id = ? AND user_id = ?",
        (passage_id, user_id),
    )
    row = await cursor.fetchone()
    return _passage_row_to_dict(row) if row else None


async def delete_passage(
    db: aiosqlite.Connection, user_id: str, passage_id: str
) -> bool:
    cursor = await db.execute(
        "DELETE FROM passages WHERE id = ? AND user_id = ?", (passage_id, user_id)
    )
    await db.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Printable sheets
# ---------------------------------------------------------------------------
#
# Stored as a RECIPE (language, level, kind, sections, seed), never as the
# finished exercises. The generators are deterministic, so the recipe rebuilds
# the identical sheet whenever it is needed: a test printed in March can have
# its answer key reprinted in June and the two still match.


async def init_sheet_tables(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS tutor_sheets (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL DEFAULT '',
            title       TEXT NOT NULL,
            language    TEXT NOT NULL,
            level       TEXT NOT NULL DEFAULT 'A1',
            kind        TEXT NOT NULL DEFAULT 'homework',
            sections    TEXT NOT NULL DEFAULT '[]',
            seed        INTEGER NOT NULL,
            student     TEXT NOT NULL DEFAULT '',
            notes       TEXT NOT NULL DEFAULT '',
            content     TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL
        )
        """
    )
    # A story sheet cannot be rebuilt from a seed — a second model call writes a
    # different story — so its generated text is stored verbatim.
    cur = await db.execute("PRAGMA table_info(tutor_sheets)")
    if "content" not in {r[1] for r in await cur.fetchall()}:
        await db.execute("ALTER TABLE tutor_sheets "
                         "ADD COLUMN content TEXT NOT NULL DEFAULT ''")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_tutor_sheets_user "
        "ON tutor_sheets(user_id, created_at DESC)"
    )
    await db.commit()


_SHEET_COLS = ("id, title, language, level, kind, sections, seed, student, "
               "notes, content, created_at")


def _sheet_row(r) -> dict:
    return {"id": r[0], "title": r[1], "language": r[2], "level": r[3],
            "kind": r[4], "sections": json.loads(r[5] or "[]"), "seed": r[6],
            "student": r[7], "notes": r[8],
            "content": json.loads(r[9]) if r[9] else None,
            "created_at": r[10]}


async def save_sheet(db: aiosqlite.Connection, user_id: str, *, title: str,
                     language: str, level: str, kind: str, sections: list,
                     seed: int, student: str = "", notes: str = "",
                     content: dict | None = None) -> dict:
    sid = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO tutor_sheets (id, user_id, title, language, level, kind, "
        "sections, seed, student, notes, content, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, user_id, title, language, level, kind, json.dumps(sections),
         int(seed), student, notes,
         json.dumps(content, ensure_ascii=False) if content else "", now))
    await db.commit()
    return {"id": sid, "title": title, "language": language, "level": level,
            "kind": kind, "sections": sections, "seed": seed,
            "student": student, "notes": notes, "content": content,
            "created_at": now}


async def list_sheets(db: aiosqlite.Connection, user_id: str,
                      limit: int = 100) -> list[dict]:
    cur = await db.execute(
        f"SELECT {_SHEET_COLS} FROM tutor_sheets WHERE user_id = ? "
        "ORDER BY created_at DESC LIMIT ?", (user_id, int(limit)))
    return [_sheet_row(r) for r in await cur.fetchall()]


async def get_sheet(db: aiosqlite.Connection, user_id: str,
                    sid: str) -> dict | None:
    cur = await db.execute(
        f"SELECT {_SHEET_COLS} FROM tutor_sheets WHERE id = ? AND user_id = ?",
        (sid, user_id))
    row = await cur.fetchone()
    return _sheet_row(row) if row else None


async def delete_sheet(db: aiosqlite.Connection, user_id: str,
                       sid: str) -> bool:
    cur = await db.execute(
        "DELETE FROM tutor_sheets WHERE id = ? AND user_id = ?", (sid, user_id))
    await db.commit()
    return cur.rowcount > 0
