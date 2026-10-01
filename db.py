import time
from datetime import datetime, timezone

import aiosqlite

DB_PATH = "bluejay.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    goal TEXT NOT NULL,
    mood TEXT NOT NULL,
    quirk TEXT NOT NULL
);

-- One row per `python run.py` invocation (a batch of calls).
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    n_calls INTEGER NOT NULL,
    concurrency INTEGER NOT NULL,
    rps REAL NOT NULL,
    model TEXT NOT NULL
);

-- One row per simulated conversation.
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    persona_id TEXT NOT NULL REFERENCES personas(id),
    status TEXT NOT NULL,              -- 'ok' or 'error'
    error TEXT,
    turns INTEGER NOT NULL,
    ended_by_caller INTEGER,
    started_at TEXT NOT NULL,
    duration_s REAL NOT NULL
);

-- Every message, in order. idx is the position within the call.
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id INTEGER NOT NULL REFERENCES calls(id),
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,                -- 'user' = caller, 'assistant' = receptionist
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_calls_run ON calls(run_id);
CREATE INDEX IF NOT EXISTS idx_messages_call ON messages(call_id, idx);
"""


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


async def init_db(path: str = DB_PATH) -> aiosqlite.Connection:
    db = await aiosqlite.connect(path)
    await db.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
    await db.execute("PRAGMA foreign_keys=ON")
    await db.executescript(SCHEMA)
    await db.commit()
    return db


async def upsert_personas(db: aiosqlite.Connection, personas: list[dict]) -> None:
    await db.executemany(
        "INSERT OR REPLACE INTO personas (id, name, goal, mood, quirk) VALUES (?, ?, ?, ?, ?)",
        [(p["id"], p["name"], p["goal"], p["mood"], p["quirk"]) for p in personas],
    )
    await db.commit()


async def create_run(
    db: aiosqlite.Connection, n_calls: int, concurrency: int, rps: float, model: str
) -> int:
    cur = await db.execute(
        "INSERT INTO runs (started_at, n_calls, concurrency, rps, model) VALUES (?, ?, ?, ?, ?)",
        (iso(time.time()), n_calls, concurrency, rps, model),
    )
    await db.commit()
    return cur.lastrowid


async def finish_run(db: aiosqlite.Connection, run_id: int) -> None:
    await db.execute("UPDATE runs SET finished_at = ? WHERE id = ?", (iso(time.time()), run_id))
    await db.commit()


async def save_call(db: aiosqlite.Connection, run_id: int, result: dict) -> None:
    """Save one finished call and all its messages in a single transaction."""
    cur = await db.execute(
        "INSERT INTO calls (run_id, persona_id, status, error, turns, ended_by_caller,"
        " started_at, duration_s) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            result["persona_id"],
            result["status"],
            result.get("error"),
            result["turns"],
            result.get("ended_by_caller"),
            iso(result["started_at"]),
            result["duration_s"],
        ),
    )
    call_id = cur.lastrowid
    await db.executemany(
        "INSERT INTO messages (call_id, idx, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (call_id, i, m["role"], m["content"], iso(m["ts"]))
            for i, m in enumerate(result["transcript"])
        ],
    )
    await db.commit()
