---
name: run-simulation
description: Run a batch of simulated calls (scenarios x personas) against a voice agent and collect results. Use when asked to execute, rerun, or smoke-test simulations.
---

# Run a Simulation

1. Confirm inputs: agent under test (adapter + config), scenario ids, persona ids, seed, N runs.
2. Smoke-test first: 1 scenario x 1 persona, verify transcript and timings are captured.
3. Run the full matrix; write each run to `runs/<run_id>/` with its config, seed, transcript, scores.
4. Surface failures separately from low scores (infra/adapter errors are not agent failures).
5. Summarize: pass rate per scenario and persona, latency p50/p95, top failure reasons.
6. Never print API keys or call PII; read creds from env.
