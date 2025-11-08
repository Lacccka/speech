# db.py
import aiosqlite

DB_PATH = "bot.db"

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    state TEXT DEFAULT 'idle',
    profile_path TEXT
);
"""


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_USERS_TABLE)
        await db.commit()


async def get_user(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT user_id, state, profile_path FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            await db.execute(
                "INSERT INTO users (user_id, state, profile_path) VALUES (?, 'idle', NULL)",
                (user_id,),
            )
            await db.commit()
            return (user_id, "idle", None)
        return row


async def set_state(user_id: int, state: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET state = ? WHERE user_id = ?", (state, user_id)
        )
        await db.commit()


async def set_profile(user_id: int, profile_path: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET profile_path = ? WHERE user_id = ?",
            (profile_path, user_id),
        )
        await db.commit()
