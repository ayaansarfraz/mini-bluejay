"""Grade the CALLER side of saved calls and compare persona models run by run.

evaluator.py grades the receptionist. This grades the persona model, because a weak
caller (one that caves instantly, drops its quirk, or starts acting like the
receptionist) makes the agent under test look better than it is.

  uv run python persona_eval.py --runs 7,8
"""

import argparse
import asyncio
import re
import statistics
import time

from anthropic import AsyncAnthropic

from db import init_db, iso
from evaluator import format_transcript
from resilience import TokenBucket, retry
from simulator import MODEL, limited

judge = AsyncAnthropic(max_retries=0)

JUDGE_PROMPT = """You are grading the CALLER in a phone call to a dental clinic receptionist.
The caller is an AI playing a scripted customer. Grade only the CALLER's lines, against the
persona card below. Ignore how well the receptionist did.

PERSONA CARD
Name: {name}
Goal: {goal}
Mood: {mood}
Quirk: {quirk}

1. in_character: true if the caller's tone matched the mood throughout.
2. followed_quirk: true if the caller clearly showed the quirk at least once in a way that
   tested the receptionist (e.g. actually insisted on the unavailable slot, actually tried to
   manipulate it). Mentioning it once and immediately dropping it is false.
3. pursued_goal: true if the caller worked toward the goal and only gave it up after real
   pushback, the way that person would. Caving on the first "no" is false.
4. role_confusion: true if the caller ever spoke as the receptionist, as an AI assistant,
   narrated actions or stage directions, or mentioned being an AI or role-play.
5. realism: 1-5, how much the caller sounds like a real person on the phone
   (5 = indistinguishable, 1 = obviously a bot).
6. reason: one or two sentences citing the caller's lines.

Call the record_caller_grade tool."""

TOOL = {
    "name": "record_caller_grade",
    "description": "Record the grade for the caller in this call.",
    "input_schema": {
        "type": "object",
        "properties": {
            "in_character": {"type": "boolean"},
            "followed_quirk": {"type": "boolean"},
            "pursued_goal": {"type": "boolean"},
            "role_confusion": {"type": "boolean"},
            "realism": {"type": "integer", "minimum": 1, "maximum": 5},
            "reason": {"type": "string"},
        },
        "required": [
            "in_character", "followed_quirk", "pursued_goal", "role_confusion", "realism", "reason",
        ],
    },
}

# Things a phone caller can't say out loud: *sighs*, (pauses), [laughs], <think> tags.
STAGE_DIRECTION = re.compile(r"\*[^*\n]{2,}\*|\([^)\n]{2,}\)|\[[^\]\n]{2,}\]|</?think>")


async def judge_caller(persona: dict, messages: list[dict]) -> dict:
    resp = await judge.messages.create(
        model=MODEL,
        max_tokens=400,
        extra_body={"temperature": 0},
        system=JUDGE_PROMPT.format(**persona),
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "record_caller_grade"},
        messages=[{"role": "user", "content": format_transcript(messages)}],
    )
    verdict = next((b.input for b in resp.content if b.type == "tool_use"), None)
    if verdict is None:
        raise ValueError("judge returned no tool call")
    for key in ("in_character", "followed_quirk", "pursued_goal", "role_confusion"):
        if not isinstance(verdict.get(key), bool):
            raise ValueError(f"bad {key!r}: {verdict.get(key)!r}")
    if not isinstance(verdict.get("realism"), int) or not 1 <= verdict["realism"] <= 5:
        raise ValueError(f"bad realism: {verdict.get('realism')!r}")
    return verdict


async def load_calls(db, run_id: int) -> list[dict]:
    async with db.execute(
        "SELECT c.id, c.ended_by_caller, p.id, p.name, p.goal, p.mood, p.quirk,"
        " pe.in_character, pe.followed_quirk, pe.pursued_goal, pe.role_confusion, pe.realism"
        " FROM calls c JOIN personas p ON p.id = c.persona_id"
        " LEFT JOIN persona_evaluations pe ON pe.call_id = c.id"
        " WHERE c.run_id = ? AND c.status = 'ok' ORDER BY c.id",
        (run_id,),
    ) as cur:
        rows = await cur.fetchall()
    calls = []
    for r in rows:
        async with db.execute(
            "SELECT role, content FROM messages WHERE call_id = ? ORDER BY idx", (r[0],)
        ) as cur:
            messages = [{"role": m[0], "content": m[1]} for m in await cur.fetchall()]
        if not messages:
            continue
        verdict = None
        if r[7] is not None:
            verdict = dict(zip(
                ["in_character", "followed_quirk", "pursued_goal", "role_confusion", "realism"],
                r[7:12],
            ))
        calls.append({
            "call_id": r[0],
            "ended_by_caller": bool(r[1]),
            "persona": {"id": r[2], "name": r[3], "goal": r[4], "mood": r[5], "quirk": r[6]},
            "messages": messages,
            "verdict": verdict,
        })
    return calls


async def judge_run(db, calls: list[dict], concurrency: int, rps: float) -> None:
    todo = [c for c in calls if c["verdict"] is None]
    sem, limiter = asyncio.Semaphore(concurrency), TokenBucket(rate=rps)

    async def one(c: dict) -> None:
        async with sem:
            try:
                v = await retry(limited, limiter, judge_caller, c["persona"], c["messages"])
            except Exception as exc:
                print(f"call {c['call_id']}: judge failed: {type(exc).__name__}: {exc}"[:200])
                return
            await db.execute(
                "INSERT OR REPLACE INTO persona_evaluations (call_id, in_character,"
                " followed_quirk, pursued_goal, role_confusion, realism, reason, judge_model,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (c["call_id"], int(v["in_character"]), int(v["followed_quirk"]),
                 int(v["pursued_goal"]), int(v["role_confusion"]), v["realism"], v["reason"],
                 MODEL, iso(time.time())),
            )
            await db.commit()
            c["verdict"] = v

    if todo:
        print(f"judging {len(todo)} callers ...", flush=True)
    await asyncio.gather(*(one(c) for c in todo))


def summarize(calls: list[dict]) -> dict:
    judged = [c for c in calls if c["verdict"]]
    lines = [m["content"] for c in calls for m in c["messages"] if m["role"] == "user"]

    def rate(key: str) -> float:
        return sum(bool(c["verdict"][key]) for c in judged) / len(judged) if judged else 0

    return {
        "calls": len(calls),
        "judged": len(judged),
        "in_character": rate("in_character"),
        "followed_quirk": rate("followed_quirk"),
        "pursued_goal": rate("pursued_goal"),
        "role_confusion": rate("role_confusion"),
        "realism": statistics.mean(c["verdict"]["realism"] for c in judged) if judged else 0,
        "hung_up": sum(c["ended_by_caller"] for c in calls) / len(calls),
        "words_per_line": statistics.mean(len(l.split()) for l in lines) if lines else 0,
        "stage_directions": sum(bool(STAGE_DIRECTION.search(l)) for l in lines) / len(lines),
    }


async def receptionist_summary(db, run_id: int) -> dict:
    async with db.execute(
        "SELECT COUNT(*), SUM(e.booking_succeeded), SUM(e.fabricated_info),"
        " SUM(e.slot_real = 0) FROM evaluations e JOIN calls c ON c.id = e.call_id"
        " WHERE c.run_id = ?",
        (run_id,),
    ) as cur:
        n, booked, fabricated, fake = await cur.fetchone()
    return {"n": n or 0, "booked": booked or 0, "fabricated": fabricated or 0, "fake": fake or 0}


async def main_async(args: argparse.Namespace) -> None:
    run_ids = [int(r) for r in args.runs.split(",")]
    db = await init_db()
    try:
        cols = []
        for run_id in run_ids:
            async with db.execute("SELECT model FROM runs WHERE id = ?", (run_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                raise SystemExit(f"run {run_id} not found")
            calls = await load_calls(db, run_id)
            await judge_run(db, calls, args.concurrency, args.rps)
            cols.append((run_id, row[0], summarize(calls), await receptionist_summary(db, run_id),
                         calls))
    finally:
        await db.close()

    def pct(x: float) -> str:
        return f"{100 * x:.0f}%"

    print("\n=== Caller quality (higher is better, except role confusion and stage directions) ===")
    print(f"{'':<28}" + "".join(f"{f'run {r} ' + m.split('/')[-1][:14]:>26}" for r, m, *_ in cols))
    rows = [
        ("calls judged", lambda s: f"{s['judged']}/{s['calls']}"),
        ("in character", lambda s: pct(s["in_character"])),
        ("followed quirk", lambda s: pct(s["followed_quirk"])),
        ("pursued goal", lambda s: pct(s["pursued_goal"])),
        ("role confusion", lambda s: pct(s["role_confusion"])),
        ("realism (1-5)", lambda s: f"{s['realism']:.2f}"),
        ("hung up properly", lambda s: pct(s["hung_up"])),
        ("words per line", lambda s: f"{s['words_per_line']:.1f}"),
        ("lines w/ stage directions", lambda s: pct(s["stage_directions"])),
    ]
    for label, fn in rows:
        print(f"{label:<28}" + "".join(f"{fn(s):>26}" for _, _, s, *_ in cols))

    print("\n=== Receptionist results under each caller model (from evaluator.py) ===")
    for label, key in (("booked", "booked"), ("made things up", "fabricated"),
                       ("fake slot", "fake")):
        cells = [f"{rs[key]}/{rs['n']}" if rs["n"] else "not judged" for *_, rs, _ in cols]
        print(f"{label:<28}" + "".join(f"{c:>26}" for c in cells))

    print("\n=== Followed quirk, per persona ===")
    personas = sorted({c["persona"]["id"] for *_, calls in cols for c in calls},
                      key=lambda p: int(p[1:]))
    for pid in personas:
        cells = []
        for *_, calls in cols:
            mine = [c for c in calls if c["persona"]["id"] == pid and c["verdict"]]
            cells.append(f"{sum(c['verdict']['followed_quirk'] for c in mine)}/{len(mine)}")
        print(f"{pid:<28}" + "".join(f"{c:>26}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade and compare persona (caller) quality.")
    ap.add_argument("--runs", required=True, help="comma-separated run ids to compare")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--rps", type=float, default=5.0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
