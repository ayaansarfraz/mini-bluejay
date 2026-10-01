---
name: review-sim
description: Review simulation code or results for reproducibility, evaluator validity, and silent failures. Use before trusting a score or merging simulator changes.
---

# Review Simulation Work

Check and report with file:line:
- **Reproducibility**: seed, versions, and config stored per run; same inputs rerun consistently.
- **Silent failures**: errors swallowed into 0 scores, empty transcripts counted as passes.
- **Evaluator validity**: criteria objective, judge rubric fixed, evidence cited, no leakage of
  expected answers into the caller or judge prompt.
- **Coverage**: happy path, edge, and adversarial scenarios; persona diversity.
- **Hygiene**: no secrets/PII in logs or committed runs.
