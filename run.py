import argparse
import asyncio
import json
import time
from collections import Counter

from db import create_run, finish_run, init_db, upsert_personas
from evaluator import evaluate_run, print_report
from orchestrator import run_batch
from simulator import MODEL


async def main_async(args: argparse.Namespace) -> None:
    with open("personas.json") as f:
        personas = json.load(f)

    db = await init_db()
    try:
        await upsert_personas(db, personas)
        if args.personas:
            wanted = args.personas.split(",")
            personas = [p for p in personas if p["id"] in wanted]
            if not personas:
                raise SystemExit(f"no personas match {args.personas}")
        run_id = await create_run(db, args.calls, args.concurrency, args.rps, MODEL)
        print(f"run {run_id}: {args.calls} calls, concurrency={args.concurrency}, rps={args.rps}")

        start = time.monotonic()
        results = await run_batch(personas, args.calls, db, run_id, args.concurrency, args.rps)
        elapsed = time.monotonic() - start
        await finish_run(db, run_id)

        ok = [r for r in results if r["status"] == "ok"]
        errors = [r for r in results if r["status"] == "error"]
        print(f"\nrun={run_id} calls={len(results)} ok={len(ok)} errors={len(errors)} time={elapsed:.1f}s")
        if ok:
            print(f"avg turns per call: {sum(r['turns'] for r in ok) / len(ok):.1f}")
            print(f"avg call duration:  {sum(r['duration_s'] for r in ok) / len(ok):.1f}s")
        for msg, n in Counter(r["error"] for r in errors).most_common(5):
            print(f"  {n}x {msg[:150]}")

        if not args.no_eval:
            print()
            await evaluate_run(db, run_id, args.concurrency, args.rps)
            await print_report(db, run_id)
    finally:
        await db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run simulated calls against the receptionist.")
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=10, help="max calls in flight")
    ap.add_argument("--rps", type=float, default=5.0, help="max LLM requests per second")
    ap.add_argument("--no-eval", action="store_true", help="skip judging; run evaluator.py later")
    ap.add_argument("--personas", help="comma-separated persona ids to use, e.g. p9,p10 (default: all)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
