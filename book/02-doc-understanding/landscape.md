# Landscape · Document Understanding & OCR

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

## Open specialized models

| Model | Org | Size | License | Highlights |
|---|---|---|---|---|
| **DeepSeek-OCR-2** | DeepSeek | ~3B | MIT | Released 2026-01; grounded Markdown, high throughput — a top open pick for "PDF → Markdown" |
| **olmOCR-2** | AI2 | 8B | Apache 2.0 | Fully open (data/training/eval); OmniDocBench average 83+ |
| PaddleOCR-VL-1.5 | Baidu | ~0.9B | Apache 2.0 | SOTA-class results at a tiny parameter count; edge/batch friendly |
| Chandra | Datalab | 9B | - | Strong multilingual |
| MinerU / Marker | community | pipeline | mind AGPL/GPL | Practical tools from the classical-pipeline school |

## General VLMs on documents

- Qwen3-VL / InternVL3.5 OCR is already strong; light document workloads may not need a specialist at all.
- Closed APIs (Gemini, GPT-5, Claude) remain the robustness ceiling for complex tables/handwriting, but cost more and lack coordinate grounding (hallucination-prone and hard to trace).

## Quick chooser

| Scenario | Pick |
|---|---|
| Bulk PDF processing (own GPUs) | DeepSeek-OCR-2 / olmOCR-2 |
| Edge or CPU-only environment | PaddleOCR-VL-1.5 |
| Few, high-value, extremely complex documents | Closed VLM APIs |
| Traceability required (bbox provenance) | DeepSeek-OCR-2 (grounded output) |

## Benchmarks

- OmniDocBench (most comprehensive), OCRBench v2, olmOCR-Bench
