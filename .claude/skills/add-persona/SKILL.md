---
name: add-persona
description: Add a synthetic caller persona (voice, accent, patience, interruption style, background noise, emotion). Use when creating or modifying caller behavior profiles for simulations.
---

# Add a Persona

1. Define in `src/bluejay/personas/` as a typed model: `id`, `speaking_style`, `accent`/voice,
   `speech_rate`, `interruption_rate`, `patience`, `emotion`, `noise_profile`, `system_prompt`.
2. Constrain numeric fields (`ge`/`le`); reject out-of-range values.
3. Personas are orthogonal to scenarios: never bake a goal into a persona.
4. Keep the prompt deterministic apart from the stored seed; log the seed per run.
5. Test: validation bounds, serialization round-trip, persona x scenario combination builds.
6. Run `pytest`, `ruff check .`, `mypy .`.
