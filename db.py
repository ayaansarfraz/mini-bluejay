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

-- One judge verdict per call.
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id INTEGER NOT NULL UNIQUE REFERENCES calls(id),
    booking_succeeded INTEGER NOT NULL,
    booked_slot TEXT,
    slot_real INTEGER,                 -- NULL when nothing was booked
    fabricated_info INTEGER NOT NULL,
    reason TEXT NOT NULL,
    judge_model TEXT NOT NULL,
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


async def latest_run_id(db: aiosqlite.Connection) -> int | None:
    async with db.execute("SELECT MAX(id) FROM runs") as cur:
        row = await cur.fetchone()
    return row[0]


async def calls_to_evaluate(db: aiosqlite.Connection, run_id: int) -> list[dict]:
    """Calls in a run that finished ok and have no evaluation yet, with their messages."""
    async with db.execute(
        "SELECT c.id, c.persona_id FROM calls c LEFT JOIN evaluations e ON e.call_id = c.id"
        " WHERE c.run_id = ? AND c.status = 'ok' AND e.id IS NULL ORDER BY c.id",
        (run_id,),
    ) as cur:
        calls = [{"call_id": r[0], "persona_id": r[1]} for r in await cur.fetchall()]
    for call in calls:
        async with db.execute(
            "SELECT role, content FROM messages WHERE call_id = ? ORDER BY idx", (call["call_id"],)
        ) as cur:
            call["messages"] = [{"role": r[0], "content": r[1]} for r in await cur.fetchall()]
    return calls


async def save_evaluation(db: aiosqlite.Connection, call_id: int, ev: dict, model: str) -> None:
    await db.execute(
        "INSERT OR REPLACE INTO evaluations (call_id, booking_succeeded, booked_slot, slot_real,"
        " fabricated_info, reason, judge_model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            call_id,
            int(ev["booking_succeeded"]),
            ev.get("booked_slot"),
            None if ev["slot_real"] is None else int(ev["slot_real"]),
            int(ev["fabricated_info"]),
            ev["reason"],
            model,
            iso(time.time()),
        ),
    )
    await db.commit()


async def report_rows(db: aiosqlite.Connection, run_id: int) -> list[dict]:
    """Every evaluation in a run, joined with its call and persona."""
    async with db.execute(
        "SELECT c.id, c.persona_id, p.name, p.mood, e.booking_succeeded, e.booked_slot,"
        " e.slot_real, e.fabricated_info, e.reason"
        " FROM evaluations e JOIN calls c ON c.id = e.call_id"
        " JOIN personas p ON p.id = c.persona_id WHERE c.run_id = ? ORDER BY c.id",
        (run_id,),
    ) as cur:
        cols = ["call_id", "persona_id", "name", "mood", "booking_succeeded", "booked_slot",
                "slot_real", "fabricated_info", "reason"]
        return [dict(zip(cols, r)) for r in await cur.fetchall()]


async def run_call_stats(db: aiosqlite.Connection, run_id: int) -> dict:
    """Counts for the whole run, including calls the judge never saw."""
    async with db.execute(
        "SELECT COUNT(*),"
        " SUM(c.status = 'error'),"
        " SUM(c.status = 'ok' AND c.ended_by_caller = 0),"
        " SUM(c.status = 'ok' AND e.id IS NULL)"
        " FROM calls c LEFT JOIN evaluations e ON e.call_id = c.id WHERE c.run_id = ?",
        (run_id,),
    ) as cur:
        total, errors, hit_max_turns, unjudged = await cur.fetchone()
    return {
        "total": total or 0,
        "errors": errors or 0,
        "hit_max_turns": hit_max_turns or 0,
        "unjudged": unjudged or 0,
    }
