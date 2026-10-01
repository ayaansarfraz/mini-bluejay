import asyncio
import time

import aiosqlite
import httpx

from db import save_call
from resilience import TokenBucket
from simulator import run_conversation


async def run_batch(
    personas: list[dict],
    n_calls: int,
    db: aiosqlite.Connection,
    run_id: int,
    concurrency: int = 10,
    rps: float = 5.0,
) -> list[dict]:
    """Run n_calls simulated conversations concurrently.

    - Semaphore: at most `concurrency` conversations are in flight at once.
    - TokenBucket: at most ~`rps` LLM requests per second across all of them.
    - One failed conversation is recorded as an error and never kills the batch.
    - Each call is saved to the database the moment it finishes, so a Ctrl+C
      keeps everything completed so far.
    """
    sem = asyncio.Semaphore(concurrency)
    limiter = TokenBucket(rate=rps)
    done = 0

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency * 2)) as http:

        async def one(i: int) -> dict:
            nonlocal done
            persona = personas[i % len(personas)]  # cycle through personas
            async with sem:
                started = time.time()
                try:
                    result = await run_conversation(persona, http, limiter)
                    result["status"] = "ok"
                except Exception as exc:
                    result = {
                        "persona_id": persona["id"],
                        "persona_name": persona["name"],
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "transcript": [],
                        "turns": 0,
                    }
                result["call_id"] = i
                result["started_at"] = started
                result["duration_s"] = round(time.time() - started, 2)
                await save_call(db, run_id, result)

            done += 1
            print(
                f"[{done}/{n_calls}] {result['persona_id']} {result['status']} "
                f"{result['turns']} turns {result['duration_s']}s",
                flush=True,
            )
            return result

        return await asyncio.gather(*(one(i) for i in range(n_calls)))
