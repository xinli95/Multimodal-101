# 02 · Document Understanding & OCR

The most practical deployment of VLMs: PDFs / scans / tables / charts → structured text. Also the entry point of multimodal RAG. This chapter covers the design of OCR-specialized VLMs — especially the clever idea of "optical context compression" — and builds a complete PDF → Markdown pipeline.

## Learning goals

- Understand how general VLMs differ from OCR-specialized models (resolution strategy, output format, grounding)
- Understand the core insight of DeepSeek-OCR: compressing long text context through visual tokens
- Build a PDF → Markdown pipeline and evaluate it OmniDocBench-style
- Know when to reach for a specialized OCR model vs. a general VLM API

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
