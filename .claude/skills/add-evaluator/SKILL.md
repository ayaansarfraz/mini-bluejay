---
name: add-evaluator
description: Add or modify an evaluator/metric that scores a simulated call (task success, latency, interruptions, tone, policy compliance). Use when creating any scoring logic over transcripts or call timings.
---

# Add an Evaluator

1. Choose the type: **deterministic** (latency, talk-over, turn counts, WER, tool-call checks) or
   **LLM-judge** (tone, helpfulness, policy). Prefer deterministic whenever possible.
2. Input: a typed `CallRecord` (transcript, timings, tool calls). Output: a typed result with
   `score`, `passed`, `evidence` (turn indices/quotes), and `evaluator_version`.
3. Fail explicitly: raise a typed error on missing/corrupt data. Never return 0 or None for "couldn't score".
4. LLM-judges: fixed rubric in the prompt, temperature 0, structured output, store model + prompt version.
5. Tests: hand-verified happy path, edge cases, determinism (same input -> same output),
   and for judges a small labeled golden set.
6. Run `pytest`, `ruff check .`, `mypy .`.
