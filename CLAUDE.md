# Bluejay – Voice Agent Simulation

Simulates callers (synthetic personas) talking to a voice agent under test, then scores
the calls. Built as an application project for Bluejay.

## Core model
- **Scenario**: goal + success criteria for one call (e.g. "reschedule appointment").
- **Persona**: caller traits (accent, patience, interruptions, noise, emotion).
- **Simulation run**: scenario x persona x agent-under-test -> transcript (+ audio, timings).
- **Evaluator**: scores a run (task success, latency, interruptions, policy/safety, tone).

## Principles
1. Runs are reproducible: every run stores seed, scenario, persona, agent config, model versions.
2. LLMs simulate callers and judge; deterministic code does metrics (latency, talk-over, WER).
3. Evaluators return structured output with evidence (turn indices), never a bare score.
4. Failures are explicit (typed errors); never record a failed call as a 0 score.
5. No secrets in the repo; keys come from env.

## Layout (planned)
- `src/bluejay/scenarios/`, `personas/`, `simulator/`, `evaluators/`, `agents/` (adapters), `reports/`
- `tests/` mirrors `src/`

## Commands
- `uv sync` · `pytest` · `ruff check .` · `ruff format .` · `mypy .`

## Skills (`.claude/skills/`)
add-scenario, add-persona, add-evaluator, run-simulation, review-sim

> Stack assumed Python/uv/pytest; update this file once the stack is settled.
