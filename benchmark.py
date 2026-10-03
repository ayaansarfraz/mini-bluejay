"""Benchmark the persona (caller) model: Claude Haiku API vs self-hosted Qwen3-8B on vLLM.

Every request is one real caller turn: a persona prompt plus a conversation history
taken from calls saved in bluejay.db, so both backends see the same realistic workload.
At each concurrency level we keep exactly C requests in flight and measure:

  ttft        time to first token (how long before the caller starts "speaking")
  latency     time to the full reply
  tok/s/req   decode speed of a single request
  throughput  output tokens/sec across all requests (what batching improves)
  cost        $ per 1,000 simulated calls

No retries here: we want raw behaviour, so failures (e.g. 429s) are counted, not hidden.

  uv run python benchmark.py --backend anthropic
  uv run python benchmark.py --backend vllm --vllm-url https://...modal.direct
"""

import argparse
import asyncio
import json
import os
import random
import sqlite3
import statistics
import time
from datetime import datetime, timezone

import httpx
from anthropic import AsyncAnthropic

from db import DB_PATH
from simulator import MODEL, caller_view, persona_prompt

MAX_TOKENS = 150  # same cap the simulator uses for caller turns
RESULTS_PATH = "benchmark_results.json"

# Prices. Haiku 4.5: $1 / $5 per million input / output tokens.
# Modal GPUs per hour (modal.com/pricing): A10G $0.000306/s, L40S $0.000542/s, L4 $0.000222/s.
HAIKU_IN_PER_TOK = 1.00 / 1e6
HAIKU_OUT_PER_TOK = 5.00 / 1e6
GPU_PER_HOUR = {"A10G": 1.10, "L40S": 1.95, "L4": 0.80}


# ---------- workload ----------


def load_workload(n: int, seed: int) -> tuple[list[dict], float]:
    """Build n caller-turn requests from saved calls, plus the average caller turns per call.

    Each request cuts a real transcript just before one of the caller's lines and asks
    the persona model for that line, which is exactly what the simulator does.
    """
    con = sqlite3.connect(DB_PATH)
    personas = {
        r[0]: {"id": r[0], "name": r[1], "goal": r[2], "mood": r[3], "quirk": r[4]}
        for r in con.execute("SELECT id, name, goal, mood, quirk FROM personas")
    }
    calls: dict[int, dict] = {}
    for call_id, persona_id, role, content in con.execute(
        "SELECT c.id, c.persona_id, m.role, m.content FROM calls c"
        " JOIN messages m ON m.call_id = c.id WHERE c.status = 'ok' ORDER BY c.id, m.idx"
    ):
        calls.setdefault(call_id, {"persona": personas[persona_id], "transcript": []})
        calls[call_id]["transcript"].append({"role": role, "content": content})
    con.close()
    if not calls:
        raise SystemExit("no saved calls in bluejay.db; run run.py first")

    # Every point where the caller is about to speak is a candidate request.
    candidates = [
        (c["persona"], c["transcript"][:i])
        for c in calls.values()
        for i, m in enumerate(c["transcript"])
        if m["role"] == "user"
    ]
    caller_turns_per_call = len(candidates) / len(calls)

    rng = random.Random(seed)
    picked = [rng.choice(candidates) for _ in range(n)]
    workload = [
        {"system": persona_prompt(p), "messages": caller_view(history)} for p, history in picked
    ]
    return workload, caller_turns_per_call


# ---------- backends ----------
# Each returns {"ttft", "latency", "in_tokens", "out_tokens"} for one streamed request.


async def anthropic_request(client: AsyncAnthropic, req: dict) -> dict:
    start = time.perf_counter()
    ttft = None
    async with client.messages.stream(
        model=MODEL, max_tokens=MAX_TOKENS, system=req["system"], messages=req["messages"]
    ) as stream:
        async for text in stream.text_stream:
            if ttft is None and text:
                ttft = time.perf_counter() - start
        final = await stream.get_final_message()
    latency = time.perf_counter() - start
    return {
        "ttft": ttft if ttft is not None else latency,
        "latency": latency,
        "in_tokens": final.usage.input_tokens,
        "out_tokens": final.usage.output_tokens,
    }


async def vllm_request(http: httpx.AsyncClient, url: str, req: dict) -> dict:
    payload = {
        "model": "llm",
        "messages": [{"role": "system", "content": req["system"]}, *req["messages"]],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},  # token counts arrive in the last chunk
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3: answer directly
    }
    start = time.perf_counter()
    ttft = None
    usage = None
    async with http.stream("POST", f"{url}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: ") :])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if ttft is None and choice.get("delta", {}).get("content"):
                    ttft = time.perf_counter() - start
    latency = time.perf_counter() - start
    if usage is None:
        raise RuntimeError("vLLM stream ended without usage")
    return {
        "ttft": ttft if ttft is not None else latency,
        "latency": latency,
        "in_tokens": usage["prompt_tokens"],
        "out_tokens": usage["completion_tokens"],
    }


async def wait_for_vllm(http: httpx.AsyncClient, url: str, timeout_s: float = 900) -> None:
    """A scaled-to-zero Modal server answers 503 until it has booted."""
    print(f"waiting for vLLM at {url} ...", flush=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = await http.get(f"{url}/health", timeout=30)
            if r.status_code == 200:
                print("vLLM is up", flush=True)
                return
        except httpx.TransportError:
            pass
        await asyncio.sleep(5)
    raise SystemExit("vLLM server did not become healthy")


# ---------- measurement ----------


def pctl(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


async def sample_vllm_metrics(http: httpx.AsyncClient, url: str, stop: asyncio.Event) -> dict:
    """Poll vLLM's Prometheus /metrics while a level runs.

    Shows *why* latency grows: how many requests the GPU is actually working on
    (running), how many are queued (waiting), and how full the KV cache is.
    """
    peak = {"running": 0.0, "waiting": 0.0, "kv_cache": 0.0}
    names = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv_cache",
        "vllm:gpu_cache_usage_perc": "kv_cache",  # older vLLM name
    }
    while not stop.is_set():
        try:
            r = await http.get(f"{url}/metrics", timeout=10)
            for line in r.text.splitlines():
                name = line.split("{")[0].split(" ")[0]
                if name in names:
                    key = names[name]
                    peak[key] = max(peak[key], float(line.rsplit(" ", 1)[1]))
        except (httpx.HTTPError, ValueError, IndexError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=1)
        except asyncio.TimeoutError:
            pass
    return {
        "peak_running": int(peak["running"]),
        "peak_waiting": int(peak["waiting"]),
        "peak_kv_cache": round(peak["kv_cache"], 3),
    }


async def run_level(send, workload: list[dict], concurrency: int, probe=None) -> dict:
    """Send every request in `workload`, keeping `concurrency` in flight."""
    sem = asyncio.Semaphore(concurrency)
    ok: list[dict] = []
    errors: list[str] = []

    async def one(req: dict) -> None:
        async with sem:
            try:
                ok.append(await send(req))
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")

    stop = asyncio.Event()
    probe_task = asyncio.create_task(probe(stop)) if probe else None
    wall_start = time.perf_counter()
    await asyncio.gather(*(one(r) for r in workload))
    wall = time.perf_counter() - wall_start
    stop.set()
    server_stats = await probe_task if probe_task else {}

    result = {"concurrency": concurrency, "requests": len(workload), "errors": len(errors),
              "wall_s": round(wall, 2), **server_stats}
    if errors:
        result["error_sample"] = sorted(set(errors))[:3]
    if not ok:
        return result

    decode_speeds = [
        (r["out_tokens"] - 1) / (r["latency"] - r["ttft"])
        for r in ok
        if r["out_tokens"] > 1 and r["latency"] > r["ttft"]
    ]
    out_tokens = sum(r["out_tokens"] for r in ok)
    result.update(
        ttft_p50=round(pctl([r["ttft"] for r in ok], 50), 3),
        ttft_p95=round(pctl([r["ttft"] for r in ok], 95), 3),
        latency_p50=round(pctl([r["latency"] for r in ok], 50), 3),
        latency_p95=round(pctl([r["latency"] for r in ok], 95), 3),
        tok_s_per_request=round(statistics.mean(decode_speeds), 1) if decode_speeds else None,
        throughput_tok_s=round(out_tokens / wall, 1),
        requests_per_s=round(len(ok) / wall, 2),
        avg_in_tokens=round(statistics.mean(r["in_tokens"] for r in ok), 1),
        avg_out_tokens=round(statistics.mean(r["out_tokens"] for r in ok), 1),
    )
    return result


def add_cost(result: dict, backend: str, caller_turns_per_call: float, gpu_per_hour: float) -> None:
    """$ per 1,000 simulated calls (persona side only)."""
    if "requests_per_s" not in result:
        return
    if backend == "anthropic":
        per_request = (
            result["avg_in_tokens"] * HAIKU_IN_PER_TOK + result["avg_out_tokens"] * HAIKU_OUT_PER_TOK
        )
    else:
        # A GPU costs the same per second however busy it is, so the cost of a request
        # is the GPU's price divided by how many requests per second it got through.
        per_request = gpu_per_hour / 3600 / result["requests_per_s"]
    result["cost_per_1k_calls"] = round(per_request * caller_turns_per_call * 1000, 2)


def print_table(backend: str, rows: list[dict]) -> None:
    print(f"\n=== {backend} ===")
    print(f"{'conc':>5}{'ok/req':>9}{'ttft p50':>10}{'ttft p95':>10}{'lat p50':>9}"
          f"{'tok/s/req':>11}{'thru tok/s':>12}{'$/1k calls':>12}{'run/wait/kv':>16}")
    for r in rows:
        if "requests_per_s" not in r:
            print(f"{r['concurrency']:>5}{0:>4}/{r['requests']:<4}  all failed: {r.get('error_sample')}")
            continue
        print(
            f"{r['concurrency']:>5}{r['requests'] - r['errors']:>4}/{r['requests']:<4}"
            f"{r['ttft_p50']:>10.2f}{r['ttft_p95']:>10.2f}{r['latency_p50']:>9.2f}"
            f"{r['tok_s_per_request'] or 0:>11.1f}{r['throughput_tok_s']:>12.1f}"
            f"{r['cost_per_1k_calls']:>12.2f}"
            + (f"{r['peak_running']:>6}/{r['peak_waiting']}/{r['peak_kv_cache']:.0%}"
               if "peak_running" in r else "")
        )
    for r in rows:
        if r.get("errors"):
            print(f"  concurrency {r['concurrency']}: {r['errors']} errors, e.g. {r['error_sample'][0]}")


async def main_async(args: argparse.Namespace) -> None:
    levels = [int(c) for c in args.levels.split(",")]
    probe = None

    label = "haiku" if args.backend == "anthropic" else f"qwen3-8b-{args.gpu.lower()}"
    if args.backend == "anthropic":
        client = AsyncAnthropic(max_retries=0)

        async def send(req: dict) -> dict:
            return await anthropic_request(client, req)

        http = None
    else:
        url = (args.vllm_url or os.getenv("VLLM_URL") or "").rstrip("/")
        if not url:
            raise SystemExit("pass --vllm-url or set VLLM_URL")
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(300, connect=30),
            limits=httpx.Limits(max_connections=max(levels) + 10),
        )
        await wait_for_vllm(http, url)

        async def probe(stop: asyncio.Event) -> dict:
            return await sample_vllm_metrics(http, url, stop)

        async def send(req: dict) -> dict:
            return await vllm_request(http, url, req)

    try:
        # Warm-up: first requests pay one-off costs (connections, caches); don't measure them.
        warm, _ = load_workload(3, seed=999)
        await run_level(send, warm, 3)

        rows = []
        for rep in range(args.repeats):
            for c in levels:
                n = max(args.min_requests, c * args.rounds)
                workload, turns_per_call = load_workload(n, seed=args.seed + c + 1000 * rep)
                print(f"{label}: repeat {rep + 1}/{args.repeats}, concurrency {c}, "
                      f"{n} requests ...", flush=True)
                result = await run_level(send, workload, c, probe)
                add_cost(result, args.backend, turns_per_call, GPU_PER_HOUR.get(args.gpu, 0))
                result["caller_turns_per_call"] = round(turns_per_call, 2)
                result["repeat"] = rep
                rows.append(result)
    finally:
        if http:
            await http.aclose()

    print_table(label, rows)

    record = {
        "backend": args.backend,
        "label": label,
        "gpu": None if args.backend == "anthropic" else args.gpu,
        "model": MODEL if args.backend == "anthropic" else f"Qwen/Qwen3-8B (vLLM, 1x {args.gpu})",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_tokens": MAX_TOKENS,
        "levels": rows,
    }
    try:
        with open(RESULTS_PATH) as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    history.append(record)
    with open(RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nsaved to {RESULTS_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark persona model backends.")
    ap.add_argument("--backend", choices=["anthropic", "vllm"], required=True)
    ap.add_argument("--vllm-url", help="base URL of the vLLM server (or set VLLM_URL)")
    ap.add_argument("--levels", default="1,10,50,100", help="concurrency levels to test")
    ap.add_argument("--rounds", type=int, default=2, help="requests per level = concurrency x rounds")
    ap.add_argument("--min-requests", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="run the whole sweep this many times")
    ap.add_argument("--gpu", choices=sorted(GPU_PER_HOUR), default="A10G",
                    help="GPU the vLLM server runs on (for cost and labels)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
