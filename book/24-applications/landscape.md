# Landscape · Applications & Evaluation

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

## Multimodal RAG components

| Component | Open picks | Notes |
|---|---|---|
| Document parsing | DeepSeek-OCR-2, olmOCR-2, MinerU | See the [understanding landscape](../landscape.md) |
| Visual retrieval embeddings | ColQwen family, Jina-CLIP, BGE-VL | Late interaction scores best but indexes are large |
| Vector stores | Qdrant, LanceDB, Milvus | All support multi-vector / late interaction now |
| Generation end | Qwen3-VL / closed VLM APIs | See [Part I](../00-orientation/index.md) |

## Agent frameworks and models

- GUI-grounding specialists: the UI-TARS line (ByteDance), CogAgent; general VLMs (Qwen3-VL, Claude computer use, Gemini) now ship GUI ability built in
- Frameworks: browser-use, agent SDKs (every major API has a computer-use tool)

## Evaluation infrastructure

| Tool/Benchmark | Purpose |
|---|---|
| VLMEvalKit, lmms-eval | Batch evaluation frameworks for open VLMs |
| MMMU, OCRBench, Video-MME | Standard understanding benchmarks |
| GenEval, VBench, OmniDocBench | Generation / document benchmarks |
| LMArena (Vision/Image/Video) | Human-preference blind testing; the most intuitive trend read |

## Where things are heading

1. In multimodal RAG, "parse vs. visual retrieval" has no winner yet; hybrid retrieval is the pragmatic choice.
2. GUI agents are the fastest-commercializing VLM track of 2026.
3. Evaluation is moving from "one score" to "dimensions + task-specific custom eval sets"; every notebook in this book follows that practice.
