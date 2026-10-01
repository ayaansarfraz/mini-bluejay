# Project: Mini Bluejay (simulated caller load tester + inference benchmark)

## Why this exists

I'm interviewing at Bluejay (YC X25) for a backend role. First round is a 30 minute backend interview. The recruiter said they're hiring backend engineers and want to move into inference next.

Bluejay tests voice and text AI agents by simulating customers ("digital humans") that call a company's agent, then evaluating how it performed. This project is a small version of their product, built to:

1. Learn the backend concepts their system depends on (async concurrency, rate limiting, retries, queues, evals)
2. Learn the inference basics they're moving into (self-hosting, batching, latency vs throughput)
3. Have something concrete to show the interviewer

I'm new to most of these concepts, so explain things plainly when they come up.

## What I'm building

### Part 1: Simulation backend (Day 1)

Text only, no voice. Built in this order, one piece working before starting the next:

1. **Target agent** (`agent.py`): FastAPI app with one `/chat` endpoint. An LLM plays a dental clinic receptionist with a fixed list of open appointment slots and tries to book the caller.
2. **Personas** (`personas.json`): 5 to 10 fake customers with a name, goal, mood, and quirk (impatient, confused, tries to book a slot that doesn't exist, etc).
3. **Simulator**: a function that runs one full conversation. A second LLM plays a persona and talks to the target agent for up to ~10 turns or until booking is done.
4. **Orchestrator**: runs 100 conversations concurrently using asyncio, with a semaphore to cap concurrency, a token bucket rate limiter for the LLM API, and retries with exponential backoff and jitter.
5. **Storage**: SQLite. Stores each run, persona, every message in order, and timestamps.
6. **Evaluator**: LLM-as-judge with a rubric (did the booking succeed, was the slot real, did the bot make anything up). Outputs a summary report.

End state: `python run.py --calls 100` prints a report card.

**Optional stretch (end of Day 1):** replace the in-memory job list with a Redis queue and run 3 worker processes, with a shared rate limiter in Redis.

### Part 2: Inference layer (Day 2)

- Serve a small open model (e.g. Qwen 2.5 3B) with vLLM on a rented GPU (Modal or RunPod)
- Use it as the persona model in place of the API
- Benchmark self-hosted vs API at concurrency 1, 10, 50, 100
- Measure: time to first token, tokens per second, total throughput, cost per 1,000 simulated calls
- Produce one chart showing the results (expect throughput to rise with concurrency while per-request latency degrades, from continuous batching)

### Part 3: Writeup

GitHub README covering what I built, the benchmark chart, what I learned, and what I'd do differently at Bluejay's scale.

## Stack

- Python, asyncio, FastAPI, uvicorn, httpx
- LLM API (cheapest available model) for Part 1
- SQLite for storage
- Optional: Redis for the queue stretch
- vLLM on Modal or RunPod for Part 2

## Interview context

- Production-scale design I should be able to explain (but don't need to build): API receives a test run, jobs go on a queue (Redis or SQS), workers run call sessions (STT to persona LLM to TTS for voice), recordings to S3, transcripts to Postgres, eval workers score each call, results to a dashboard. Hard parts: provider rate limits, concurrency caps, idempotent retries, backpressure, voice latency budget (under ~1s per turn).
- Coding to be fluent in: async worker pool with a semaphore, token bucket rate limiter, retry with backoff and jitter, consuming a streaming response.
- How I'll use the project: mention it naturally when discussing background or inference, share one concrete benchmark finding, send the link after if they want it.

## Current status

Starting Day 1, step 1 (target agent).

## How to help me

- Casual, direct, concise
- No em dashes
- Give one clear recommendation rather than a list of options
- Explain new terms in plain language with a quick analogy
- Keep me building one piece at a time: get one call working before scaling to many