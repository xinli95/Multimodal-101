# 24 · Applications and Evaluation

**How this connects to Part I.** Every production concern in this chapter is something Part I gave you a number for. Visual-token cost is [chapter 03](../03-image-processor/index.md)'s soft-token menu — 280 per image, 70 per video frame, and the [OCR cliff](../04-vision-tower/notebooks/02_image_understanding.ipynb) that decides whether your documents are readable at all. Latency and throughput are [chapter 09](../09-generation-and-serving/index.md)'s cache curve and its 15× batch scaling. "Should I fine-tune?" is [chapter 10](../10-finetuning/index.md)'s freeze-or-adapt decision.

The point of this chapter is that those are not separate topics. A multimodal RAG system that retrieves ten page images has just spent 2,800 context tokens before the question is even asked; whether that is affordable is a question you can now answer from first principles instead of by benchmarking blindly.

## Learning goals

- Build multimodal RAG: document parsing + multimodal embedding retrieval + VLM generation, with the token budget worked out in advance
- Understand the core loop of a multimodal agent: screenshot understanding → decision → action (GUI agents)
- Develop evaluation instincts: automatic metrics, LLM-as-judge, arena blind tests — where each applies and where each traps you
- Learn the production concerns: visual token cost, latency, caching, model routing

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
