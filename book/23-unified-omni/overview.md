# 07 · Unified & Omni Models

The capstone chapter: one model doing both understanding and generation, across all modalities. This is what the closed frontier (GPT-5, Gemini 3) actually is, and the most active open research direction. Theory clarifies the architecture routes to "unification"; practice runs BAGEL / InternVL-U / Qwen3.5-Omni.

## Learning goals

- Understand the fundamental tension — understanding prefers continuous features, generation prefers discrete/diffusion — and its four resolutions
- Read three representative architectures: Janus (decoupled encoding), BAGEL (MoT + hybrid AR/diffusion), Qwen-Omni (Thinker-Talker)
- Understand the emergent abilities unified training brings (world-knowledge editing, cross-modal reasoning)
- Run one open unified model through the full understand + generate + edit loop

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
