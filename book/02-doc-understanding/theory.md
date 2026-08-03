# Theory · Document Understanding

## 1. From classical OCR to end-to-end VLMs

- Classical pipeline: detection → recognition → layout analysis → reading-order recovery (the classic PaddleOCR architecture)
- End-to-end paradigm: one VLM outputs Markdown/HTML/JSON directly (Nougat 2023 opened this line → today's olmOCR, DeepSeek-OCR)
- The trade: end-to-end saves plumbing but can hallucinate; classical pipelines are controllable but errors cascade

## 2. Key design questions

- **Resolution strategy**: documents are high-resolution, information-dense inputs — tiling vs. NaViT variable-length patches (the PaddleOCR-VL route)
- **Output representation**: Markdown (reader-friendly) vs. HTML (table fidelity) vs. grounded output with coordinates (traceable — the key to fighting hallucination)
- **Reading order**: how to linearize multi-column, cross-page, figure-mixed layouts

## 3. Optical context compression (DeepSeek-OCR's core insight)

- The idea: render a 1000-word page as an image and ~100 visual tokens suffice to reconstruct it nearly losslessly → vision is an efficient compression channel for text
- The implication: OCR is not just an application — it may be a general mechanism for cheap long-context in LLMs ("store memories as images")
- A beautiful closed loop of "understand → compress → regenerate"; worth a careful read

## 4. Evaluation

- **OmniDocBench**: the most comprehensive document-parsing benchmark today (layout, formulas, tables, multilingual)
- olmOCR-Bench: unit-test-style evaluation
- Benchmark trap: Markdown formatting differences get mis-scored as content errors

## Key papers / projects

| Paper/Project | Year | Why read it |
|---|---|---|
| Nougat | 2023 | Opened end-to-end document conversion |
| GOT-OCR 2.0 | 2024 | The "unified OCR" concept |
| DeepSeek-OCR / OCR-2 | 2025–2026 | Optical context compression |
| olmOCR / olmOCR-2 | 2025 | The fully-open route (data + weights + eval) |
| PaddleOCR-VL / -VL-1.5 | 2025–2026 | NaViT encoder + lightweight efficiency route |
