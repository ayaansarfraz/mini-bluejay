---
name: add-scenario
description: Add a call scenario (goal, success criteria, expected agent behavior) to the simulation suite. Use when creating or editing a test scenario for the voice agent under test.
---

# Add a Scenario

A scenario defines what the simulated caller wants and how we know the agent succeeded.

1. Define it as a typed model in `src/bluejay/scenarios/` with: `id`, `goal`, `caller_context`
   (facts the caller knows), `success_criteria` (machine-checkable where possible),
   `failure_conditions`, `max_turns`, and `tags` (happy-path, edge-case, adversarial).
2. Criteria must be objective: "agent confirms new time AND calls reschedule tool", not "agent is helpful".
3. Include at least one edge variant (caller changes mind, gives wrong info, asks off-topic).
4. Add a test that the scenario validates and its criteria are well-formed.
5. Run `pytest`, `ruff check .`, `mypy .`.
