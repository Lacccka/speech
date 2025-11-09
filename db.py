# db.py
from typing import Dict, List, Optional, Sequence

import aiosqlite

DB_PATH = "bot.db"

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    state TEXT DEFAULT 'idle',
    profile_path TEXT,
    current_session INTEGER DEFAULT 0,
    pending_tts_text TEXT
);
"""

CREATE_SPEAKER_REFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS speaker_references (
    reference_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""

CREATE_SAMPLES_TABLE = """
CREATE TABLE IF NOT EXISTS samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""


async def _ensure_column(
    db: aiosqlite.Connection, table: str, column: str, definition: str
) -> None:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    columns = await cursor.fetchall()
    if not any(col[1] == column for col in columns):
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_USERS_TABLE)
        await db.execute(CREATE_SAMPLES_TABLE)
        await db.execute(CREATE_SPEAKER_REFERENCES_TABLE)
        await _ensure_column(db, "users", "current_session", "INTEGER DEFAULT 0")
        await _ensure_column(db, "users", "pending_tts_text", "TEXT")
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_user_session ON samples(user_id, session_id)"
        )
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_speaker_refs_user_position
            ON speaker_references(user_id, position, reference_id)
            """
        )
        await db.commit()


async def get_user(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            SELECT user_id, state, profile_path, current_session, pending_tts_text
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            await db.execute(
                """
                INSERT INTO users (user_id, state, profile_path, current_session, pending_tts_text)
                VALUES (?, 'idle', NULL, 0, NULL)
                """,
                (user_id,),
            )
            await db.commit()
            return (user_id, "idle", None, 0, None)
        return row


async def set_state(user_id: int, state: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET state = ? WHERE user_id = ?", (state, user_id)
        )
        await db.commit()


async def set_profile(user_id: int, profile_path: Sequence[str] | str):
    """Backward compatible wrapper for ``set_speaker_references``."""

    if isinstance(profile_path, str):
        references: Sequence[str] = [profile_path]
    else:
        references = profile_path
    await set_speaker_references(user_id, references)


async def start_user_session(user_id: int) -> int:
    """Increment the user's training session counter and return the new value."""

    await get_user(user_id)  # ensure user exists
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT current_session FROM users WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        current_session = row[0] if row and row[0] is not None else 0
        new_session = current_session + 1
        await db.execute(
            "UPDATE users SET current_session = ? WHERE user_id = ?",
            (new_session, user_id),
        )
        await db.commit()
        return new_session


async def get_current_session(user_id: int) -> int:
    """Return the currently active session for the user."""

    _, _, _, current_session, _ = await get_user(user_id)
    return current_session


async def add_sample(
    user_id: int, file_path: str, session_id: Optional[int] = None
) -> int:
    """Persist metadata about a recorded audio sample.

    Returns the inserted sample identifier.
    """

    if session_id is None or session_id <= 0:
        session_id = await get_current_session(user_id)
        if session_id <= 0:
            session_id = await start_user_session(user_id)

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO samples (user_id, session_id, file_path) VALUES (?, ?, ?)",
            (user_id, session_id, file_path),
        )
        await db.commit()
        return cursor.lastrowid


async def get_user_samples(user_id: int) -> List[Dict[str, object]]:
    """Fetch all stored audio samples for a user."""

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            SELECT sample_id, user_id, session_id, file_path, created_at
            FROM samples
            WHERE user_id = ?
            ORDER BY session_id, sample_id
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "sample_id": row[0],
                "user_id": row[1],
                "session_id": row[2],
                "file_path": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]


async def get_latest_samples(user_id: int, session_id: int) -> List[Dict[str, object]]:
    """Return samples recorded for the specified training session."""

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            SELECT sample_id, user_id, session_id, file_path, created_at
            FROM samples
            WHERE user_id = ? AND session_id = ?
            ORDER BY sample_id
            """,
            (user_id, session_id),
        )
        rows = await cursor.fetchall()
        return [
            {
                "sample_id": row[0],
                "user_id": row[1],
                "session_id": row[2],
                "file_path": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]


async def delete_user_samples(user_id: int, session_id: Optional[int] = None) -> None:
    """Remove stored sample metadata for a user.

    If ``session_id`` is provided, only entries for that session are removed.
    """

    async with aiosqlite.connect(DB_PATH) as db:
        if session_id is None:
            await db.execute("DELETE FROM samples WHERE user_id = ?", (user_id,))
        else:
            await db.execute(
                "DELETE FROM samples WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
        await db.commit()


async def set_pending_tts_text(user_id: int, text: Optional[str]) -> None:
    """Store or clear the pending text awaiting TTS mode selection."""

    await get_user(user_id)  # ensure user exists
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET pending_tts_text = ? WHERE user_id = ?",
            (text, user_id),
        )
        await db.commit()


async def get_pending_tts_text(user_id: int) -> Optional[str]:
    """Retrieve text awaiting TTS mode selection for the user."""

    _, _, _, _, pending = await get_user(user_id)
    return pending


async def get_speaker_references(user_id: int) -> List[str]:
    """Return curated reference file paths for ``user_id``."""

    await get_user(user_id)  # ensure user exists
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            SELECT file_path
            FROM speaker_references
            WHERE user_id = ?
            ORDER BY position, reference_id
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]


def _normalise_references(references: Sequence[str]) -> List[str]:
    return [ref for ref in (ref.strip() for ref in references) if ref]


async def _resequence_positions(
    db: aiosqlite.Connection, user_id: int
) -> Optional[str]:
    cursor = await db.execute(
        """
        SELECT reference_id, file_path
        FROM speaker_references
        WHERE user_id = ?
        ORDER BY position, reference_id
        """,
        (user_id,),
    )
    rows = await cursor.fetchall()
    for new_pos, (reference_id, _) in enumerate(rows):
        await db.execute(
            "UPDATE speaker_references SET position = ? WHERE reference_id = ?",
            (new_pos, reference_id),
        )
    first_path = rows[0][1] if rows else None
    await db.execute(
        "UPDATE users SET profile_path = ? WHERE user_id = ?",
        (first_path, user_id),
    )
    return first_path


async def set_speaker_references(user_id: int, references: Sequence[str]) -> None:
    """Replace stored speaker references for ``user_id``."""

    normalised = _normalise_references(references)
    await get_user(user_id)  # ensure user exists
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM speaker_references WHERE user_id = ?", (user_id,))
        for position, ref in enumerate(normalised):
            await db.execute(
                """
                INSERT INTO speaker_references (user_id, file_path, position)
                VALUES (?, ?, ?)
                """,
                (user_id, ref, position),
            )
        if normalised:
            first_path = normalised[0]
        else:
            first_path = None
        await db.execute(
            "UPDATE users SET profile_path = ? WHERE user_id = ?",
            (first_path, user_id),
        )
        await db.commit()


async def add_speaker_reference(
    user_id: int, file_path: str, *, position: Optional[int] = None
) -> int:
    """Add a single speaker reference and return its identifier."""

    await get_user(user_id)  # ensure user exists
    cleaned = file_path.strip()
    if not cleaned:
        raise ValueError("file_path must not be empty")

    async with aiosqlite.connect(DB_PATH) as db:
        if position is None:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM speaker_references WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            position = row[0] if row and row[0] is not None else 0

        cursor = await db.execute(
            """
            INSERT INTO speaker_references (user_id, file_path, position)
            VALUES (?, ?, ?)
            """,
            (user_id, cleaned, position),
        )
        await _resequence_positions(db, user_id)
        await db.commit()
        return cursor.lastrowid


async def remove_speaker_reference(
    user_id: int,
    *,
    file_path: Optional[str] = None,
    reference_id: Optional[int] = None,
) -> None:
    """Remove stored references either by ``file_path`` or ``reference_id``."""

    if file_path is None and reference_id is None:
        raise ValueError("Either file_path or reference_id must be provided")

    await get_user(user_id)  # ensure user exists
    async with aiosqlite.connect(DB_PATH) as db:
        if reference_id is not None:
            await db.execute(
                "DELETE FROM speaker_references WHERE user_id = ? AND reference_id = ?",
                (user_id, reference_id),
            )
        else:
            await db.execute(
                "DELETE FROM speaker_references WHERE user_id = ? AND file_path = ?",
                (user_id, file_path.strip()),
            )
        await _resequence_positions(db, user_id)
        await db.commit()
