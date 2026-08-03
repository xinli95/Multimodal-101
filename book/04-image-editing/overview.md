# 04 · Image Editing

A track that became its own discipline after 2025: "talk to a picture in plain language and get precise edits back." This chapter explains how instruction-based editing works (in-context conditioning, reference-image injection) and compares open models (Qwen-Image-Edit, FLUX Kontext) against closed ones (Nano Banana, GPT Image 2).

## Learning goals

- Understand the paradigm shift from inpainting (paint a mask) to instruction editing (pure language)
- Understand in-context conditioning: reference images as sequence conditions in joint attention with the target
- Grasp the two core hard problems: edit locality (don't touch what shouldn't change) and identity consistency
- Run an open editing model and A/B it against closed APIs on the same tasks

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
