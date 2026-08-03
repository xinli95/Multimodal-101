# 01 · VLM — Vision-Language Models

The main arena of multimodal understanding: making an LLM "see" images and video. This chapter nails down the dominant VLM architecture (vision encoder + connector + LLM) and its training recipe, then runs the open-source flagship side by side with closed frontier APIs.

## Learning goals

- Draw the complete dataflow of a modern VLM: image → ViT → connector → LLM context
- Understand the trade-offs of the three connector designs (Linear/MLP, Q-Former, cross-attention)
- Understand why dynamic / native resolution is the key to OCR and document ability
- Run a small Qwen3-VL locally and compare it with Gemini / GPT-5 APIs on the same task set

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
