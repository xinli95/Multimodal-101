# Landscape · Multimodal Understanding

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

Part I studies one model in depth. This page is the counterweight: where Gemma 4 sits among everything else that reads images, documents, video and audio. The generation side has its own landscape pages in Part II ([20](20-image-generation/landscape.md), [21](21-video-generation/landscape.md), [22](22-audio-generation/landscape.md), [23](23-unified-omni/landscape.md)).

## The vision encoders underneath almost everything

| Model | Org | License | Notes |
|---|---|---|---|
| SigLIP 2 | Google | Apache 2.0 | The most common vision tower in current open VLMs |
| CLIP (ViT-L/14) | OpenAI | MIT | Classic baseline, still the best thing to teach with |
| DINOv2/v3 | Meta | Apache 2.0 | Self-supervised line, strong for dense prediction |

Gemma 4's tower is trained in-house rather than bolted on from a released SigLIP checkpoint, which is part of why its preprocessing (no ImageNet normalisation, learned 2D position table) differs from the LLaVA-lineage default. See [chapter 03](03-image-processor/index.md) and [chapter 04](04-vision-tower/index.md).

## Open VLMs

| Model | Org | Sizes | License | Highlights |
|---|---|---|---|---|
| **Gemma 4** | Google | E2B / E4B / 26B-A4B / 31B | Gemma terms | **Part I's subject.** Text+image+video everywhere, audio on E2B/E4B; 128K–256K context; PLE, MoE, variable-aspect-ratio vision |
| **Qwen3-VL** | Alibaba | 2B → 235B-A22B (MoE) | Apache 2.0 | Open flagship; benchmarks against Gemini 2.5 Pro / GPT-5; strong across OCR, grounding, video, agents |
| **InternVL3.5** | Shanghai AI Lab | 1B → 241B-A28B | MIT/Apache (per size) | Open SOTA contender; reasoning and efficiency both emphasized |
| Llama 4 multimodal | Meta | multi-size MoE | Llama license | Big ecosystem, strong in English |
| Molmo | AI2 | 1B–72B | Apache 2.0 | Fully open data; unique pointing ability |
| Pixtral | Mistral | 12B/124B | Apache 2.0 | The European representative |
| Phi-4 multimodal | Microsoft | ~5.6B | MIT | The on-device small-model representative |

The three design-space axes worth comparing while reading Part I:

| Axis | Gemma 4 | Qwen3-VL | InternVL3.5 | LLaVA (2023) |
|---|---|---|---|---|
| Image → tokens | Fixed budget, native aspect ratio, 3×3 pooling | Native resolution, 2×2 merge, sequence grows with image | Fixed tiles + thumbnail | One 224×224 square |
| Spatial position | Learned 2D table + 2D RoPE | M-RoPE (time/height/width) | Standard 1D over tiles | 1D over 576 patches |
| Connector | Pooler + linear embedder | MLP merger | MLP + pixel shuffle | Single MLP |

## Closed frontier

| Model | Org | Multimodal traits |
|---|---|---|
| Gemini 3.x Pro | Google | Unified multimodal from day one — text/image/audio/video natively; widely considered the strongest multimodal understanding. Gemma 4 is the open sibling of this line, which is part of why it is worth studying |
| GPT-5.x | OpenAI | Native multimodal input, strong reasoning chains |
| Claude (Opus/Sonnet/Fable) | Anthropic | Image input + document understanding, strong on engineering tasks |

## Documents and OCR

| Model | Org | Size | License | Highlights |
|---|---|---|---|---|
| **DeepSeek-OCR-2** | DeepSeek | ~3B | MIT | Released 2026-01; grounded Markdown, high throughput — a top open pick for "PDF → Markdown" |
| **olmOCR-2** | AI2 | 8B | Apache 2.0 | Fully open (data/training/eval); OmniDocBench average 83+ |
| PaddleOCR-VL-1.5 | Baidu | ~0.9B | Apache 2.0 | SOTA-class results at a tiny parameter count; edge/batch friendly |
| Chandra | Datalab | 9B | - | Strong multilingual |
| MinerU / Marker | community | pipeline | mind AGPL/GPL | Practical tools from the classical-pipeline school |

General VLMs are now good enough at OCR that light document workloads may not need a specialist — and Gemma 4's soft-token menu ([chapter 03](03-image-processor/index.md)) is exactly the knob that decides whether it will read your small print. Closed APIs remain the robustness ceiling for complex tables and handwriting, but cost more and lack coordinate grounding.

| Scenario | Pick |
|---|---|
| Bulk PDF processing (own GPUs) | DeepSeek-OCR-2 / olmOCR-2 |
| Edge or CPU-only environment | PaddleOCR-VL-1.5 |
| Few, high-value, extremely complex documents | Closed VLM APIs |
| Traceability required (bbox provenance) | DeepSeek-OCR-2 (grounded output) |

Benchmarks: OmniDocBench (most comprehensive), OCRBench v2, olmOCR-Bench.

## Speech understanding

| Model | Org | Notes |
|---|---|---|
| Whisper (large-v3 / turbo) | OpenAI | The encoder-decoder ASR baseline everything is measured against |
| Gemma 4 E2B/E4B audio tower | Google | Chunked-attention encoder inside a general multimodal model — [chapter 05](05-audio-and-video/index.md) |
| Parakeet / Canary | NVIDIA | The CTC/transducer streaming line |

## Where things are heading

1. Open and closed models are roughly tied on standard benchmarks; the closed edge concentrates in long-tail robustness and native video/audio fusion.
2. "Thinking VLMs" (multimodal CoT + RLVR) are the default configuration for new models — including all four Gemma 4 sizes.
3. Audio-in as a first-class modality (rather than a bolted-on ASR pipeline) is spreading downward from frontier models; Gemma 4 E2B/E4B is the accessible open example.
4. GUI/computer-use agents are the fastest-growing deployment scenario for VLMs.
5. Watch Qwen3.8 (>1T-parameter multimodal, previewed at WAIC 2026-07; open weights promised but not yet released).
