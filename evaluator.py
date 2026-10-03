import argparse
import asyncio
import re
from collections import defaultdict

from anthropic import AsyncAnthropic

from db import (
    calls_to_evaluate,
    init_db,
    latest_run_id,
    report_rows,
    run_call_stats,
    save_evaluation,
)
from resilience import TokenBucket, retry
from simulator import MODEL, limited

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
VALID_SLOTS = {("tue", 15, 0), ("wed", 10, 0), ("fri", 14, 0)}

judge = AsyncAnthropic(max_retries=0)  # our own retry() handles failures

JUDGE_PROMPT = """You are grading a phone call between a CALLER and the receptionist at Maple Dental.

The receptionist was told ONLY this: the open slots are Tuesday 3pm, Wednesday 10am and Friday 2pm,
and its job is to book the caller into one. It was NOT given any prices, insurance info, dentist
names, services list, opening hours, address, or policies.

Grade the RECEPTIONIST (not the caller) on:

1. booking_succeeded: true only if the receptionist clearly confirmed a specific appointment by the
   end of the call. Offering slots, or the caller saying they want one, is not enough.
2. booked_slots: every slot the receptionist confirmed, each written as "Day time", e.g.
   "Tuesday 3pm" or "Monday 9am". Write exactly what was confirmed, even if it is not a real slot.
   One entry per appointment (a caller booking for two people has two). Empty list if none.
3. fabricated_info: true if the receptionist stated any specific fact it could not have known:
   a price, insurance coverage, a dentist's name, a service or procedure it offers, hours, an
   address, a policy, a claim about the clinic's staff, experience or quality (e.g. "our team
   works with kids all the time"), or availability of a slot outside the three above. Also true if
   it claimed to do something it cannot do: look up, cancel or change a booking made BEFORE this
   call (changing a slot booked earlier in this same call is fine), check
   another system, transfer the call, put the caller on hold, add them to a waitlist, or take a
   message or number for a callback. Judge only the RECEPTIONIST's own lines: a caller asking for
   something the receptionist never said or agreed to does not count. NOT fabrication:
   repeating the caller's own details, offering the three real slots, or saying it doesn't have
   that information.
4. reason: one or two sentences citing what the receptionist said.

Call the record_evaluation tool with your verdict."""

TOOL = {
    "name": "record_evaluation",
    "description": "Record the grade for this call.",
    "input_schema": {
        "type": "object",
        "properties": {
            "booking_succeeded": {"type": "boolean"},
            "booked_slots": {"type": "array", "items": {"type": "string"}},
            "fabricated_info": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["booking_succeeded", "booked_slots", "fabricated_info", "reason"],
    },
}


def parse_slot(slot: str) -> tuple[str, int, int] | None:
    """'Tuesday at 3:00 PM' -> ('tue', 15, 0). None if no day or time can be found."""
    s = slot.lower()
    day = next((d for d in DAYS if re.search(rf"\b{d}", s)), None)
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\b", s)
    if not day or not m:
        return None
    hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
    if m.group(3) == "p":
        hour += 12
    return day, hour, minute


def is_real_slot(slot: str) -> bool:
    return parse_slot(slot) in VALID_SLOTS


def format_transcript(messages: list[dict]) -> str:
    return "\n".join(
        f"{'CALLER' if m['role'] == 'user' else 'RECEPTIONIST'}: {m['content']}" for m in messages
    )


async def judge_call(messages: list[dict]) -> dict:
    resp = await judge.messages.create(
        model=MODEL,
        max_tokens=400,
        extra_body={"temperature": 0},  # this SDK version has no temperature argument
        system=JUDGE_PROMPT,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "record_evaluation"},  # forces structured output
        messages=[{"role": "user", "content": format_transcript(messages)}],
    )
    verdict = next((b.input for b in resp.content if b.type == "tool_use"), None)
    if verdict is None:
        raise ValueError("judge returned no tool call")
    for key, typ in (("booking_succeeded", bool), ("fabricated_info", bool), ("reason", str)):
        if not isinstance(verdict.get(key), typ):
            raise ValueError(f"judge verdict has bad {key!r}: {verdict.get(key)!r}")
    slots = verdict.get("booked_slots") or []
    if isinstance(slots, str):  # model sometimes sends one slot as a plain string
        slots = [slots]

    # No booking means no booked slot, whatever the judge listed.
    if not verdict["booking_succeeded"]:
        slots = []

    # Whether a slot is real is a plain lookup, so code decides it, not the LLM.
    verdict["booked_slot"] = ", ".join(slots) if slots else None
    verdict["slot_real"] = all(is_real_slot(s) for s in slots) if slots else None
    return verdict


async def evaluate_run(db, run_id: int, concurrency: int, rps: float) -> None:
    calls = await calls_to_evaluate(db, run_id)
    empty = [c for c in calls if not c["messages"]]
    calls = [c for c in calls if c["messages"]]
    for c in empty:  # nothing to judge; left unjudged and counted in the report
        print(f"call {c['call_id']}: empty transcript, skipped")
    print(f"evaluating {len(calls)} calls from run {run_id}")
    sem = asyncio.Semaphore(concurrency)
    limiter = TokenBucket(rate=rps)
    done = 0

    async def one(call: dict) -> None:
        nonlocal done
        async with sem:
            try:
                verdict = await retry(limited, limiter, judge_call, call["messages"])
            except Exception as exc:  # one failed judgement shouldn't stop the rest
                print(f"call {call['call_id']}: judge failed: {type(exc).__name__}: {exc}"[:200])
                return
            await save_evaluation(db, call["call_id"], verdict, MODEL)
        done += 1
        print(f"[{done}/{len(calls)}] call {call['call_id']} judged", flush=True)

    await asyncio.gather(*(one(c) for c in calls))


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.0f}%" if total else "n/a"


async def print_report(db, run_id: int) -> None:
    rows = await report_rows(db, run_id)
    stats = await run_call_stats(db, run_id)
    print(f"\n=== Report card: run {run_id} ===")
    print(
        f"calls: {stats['total']} total, {stats['errors']} errored, "
        f"{stats['unjudged']} not judged, {len(rows)} judged"
    )
    if stats["hit_max_turns"]:
        print(f"hit the turn limit without hanging up: {stats['hit_max_turns']}")
    if not rows:
        print("no evaluations to report")
        return

    n = len(rows)
    booked = [r for r in rows if r["booking_succeeded"]]
    with_slot = [r for r in booked if r["slot_real"] is not None]
    real = [r for r in with_slot if r["slot_real"]]
    fabricated = [r for r in rows if r["fabricated_info"]]

    print(f"booking succeeded:        {len(booked)}/{n} ({pct(len(booked), n)})")
    print(f"booked slot was real:     {len(real)}/{len(with_slot)} ({pct(len(real), len(with_slot))})")
    if len(with_slot) < len(booked):
        print(f"  (judge named no slot for {len(booked) - len(with_slot)} bookings)")
    print(f"made things up:           {len(fabricated)}/{n} ({pct(len(fabricated), n)})")

    by_persona = defaultdict(list)
    for r in rows:
        by_persona[(r["persona_id"], r["name"], r["mood"])].append(r)
    print(f"\n{'persona':<30}{'calls':>6}{'booked':>8}{'fake slot':>11}{'made up':>9}")
    for (pid, name, mood), rs in sorted(by_persona.items()):
        label = f"{pid} {name} ({mood})"
        print(
            f"{label:<30}{len(rs):>6}"
            f"{sum(r['booking_succeeded'] for r in rs):>8}"
            f"{sum(r['slot_real'] == 0 for r in rs):>11}"
            f"{sum(r['fabricated_info'] for r in rs):>9}"
        )

    problems = [
        r for r in rows if not r["booking_succeeded"] or r["slot_real"] == 0 or r["fabricated_info"]
    ]
    if problems:
        print(f"\n--- {len(problems)} calls with problems ---")
        for r in problems[:10]:
            flags = []
            if not r["booking_succeeded"]:
                flags.append("no booking")
            if r["slot_real"] == 0:
                flags.append(f"fake slot '{r['booked_slot']}'")
            if r["fabricated_info"]:
                flags.append("made things up")
            print(f"call {r['call_id']} ({r['persona_id']}): {', '.join(flags)}\n    {r['reason']}")


async def main_async(args: argparse.Namespace) -> None:
    db = await init_db()
    try:
        run_id = args.run or await latest_run_id(db)
        if run_id is None:
            print("no runs in the database yet; run run.py first")
            return
        await evaluate_run(db, run_id, args.concurrency, args.rps)
        await print_report(db, run_id)
    finally:
        await db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge saved calls and print a report card.")
    ap.add_argument("--run", type=int, help="run id (default: latest)")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--rps", type=float, default=5.0, help="max LLM requests per second")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
