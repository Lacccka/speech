import pytest

aiosqlite = pytest.importorskip("aiosqlite")

import db


@pytest.mark.asyncio
async def test_get_speaker_references_removes_empty_entries(tmp_path, monkeypatch):
    db_path = tmp_path / "bot.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))

    await db.init_db()

    user_id = 123
    # Ensure the user exists in the database
    await db.get_user(user_id)

    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.execute(
            """
            INSERT INTO speaker_references (user_id, file_path, position)
            VALUES (?, '', 0)
            """,
            (user_id,),
        )
        await conn.execute(
            """
            INSERT INTO speaker_references (user_id, file_path, position)
            VALUES (?, ?, 1)
            """,
            (user_id, str(tmp_path / "valid.wav")),
        )
        await conn.commit()

    references = await db.get_speaker_references(user_id)

    assert references == [str(tmp_path / "valid.wav")]

    async with aiosqlite.connect(db.DB_PATH) as conn:
        cursor = await conn.execute(
            "SELECT file_path FROM speaker_references WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()

    assert rows == [(str(tmp_path / "valid.wav"),)]
