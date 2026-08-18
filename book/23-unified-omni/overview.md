# 23 · Unified and Omni Models

**How this connects to Part I.** Gemma 4 is unified on exactly one side: four input modalities, one output. [Chapter 08](../08-fusion-and-masks/index.md) showed how little it takes to get there — project everything into the text embedding space, `masked_scatter` it into the sequence, and the decoder never learns that an image was involved.

Running that trick backwards is much harder, and this chapter is about why. Understanding wants continuous features; generation wants discrete tokens or a diffusion trajectory. Gemma 4 sidesteps the tension by never generating pixels. The models here confront it.

Read this chapter as the answer to a question Part I deliberately left open: *if fusing four modalities into one sequence is that easy, why is generating them back out still an open research problem?*

## Learning goals

- Understand the fundamental tension — understanding prefers continuous features, generation prefers discrete/diffusion — and its four resolutions
- Read three representative architectures: Janus (decoupled encoding), BAGEL (MoT + hybrid AR/diffusion), Qwen-Omni (Thinker-Talker)
- Understand the emergent abilities unified training brings (world-knowledge editing, cross-modal reasoning)
- Run one open unified model through the full understand + generate + edit loop

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
