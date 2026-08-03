# Landscape · VLM

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

## Open source

| Model | Org | Sizes | License | Highlights |
|---|---|---|---|---|
| **Qwen3-VL** | Alibaba | 2B → 235B-A22B (MoE) | Apache 2.0 | Open flagship; benchmarks against Gemini 2.5 Pro / GPT-5; strong across OCR, grounding, video, agents |
| **InternVL3.5** | Shanghai AI Lab | 1B → 241B-A28B | MIT/Apache (per size) | Open SOTA contender; reasoning and efficiency both emphasized |
| Llama 4 multimodal | Meta | multi-size MoE | Llama license | Big ecosystem, strong in English |
| Molmo | AI2 | 1B–72B | Apache 2.0 | Fully open data; unique pointing ability |
| Pixtral | Mistral | 12B/124B | Apache 2.0 | The European representative |
| Phi-4 multimodal | Microsoft | ~5.6B | MIT | The on-device small-model representative |

**Teaching picks**: use the small Qwen3-VL sizes locally (🟡 consumer GPU); for architecture study, the Qwen3-VL + InternVL3.5 tech reports together cover the mainstream design space.

## Closed frontier

| Model | Org | Multimodal traits |
|---|---|---|
| Gemini 3.x Pro | Google | Unified multimodal architecture from day one — text/image/audio/video natively; widely considered the strongest multimodal understanding |
| GPT-5.x | OpenAI | Native multimodal input, strong reasoning chains |
| Claude (Opus/Sonnet/Fable) | Anthropic | Image input + document understanding, strong on engineering tasks |

## Where things are heading

1. Open and closed models are roughly tied on standard benchmarks; the closed edge concentrates in long-tail robustness and native video/audio fusion.
2. "Thinking VLMs" (multimodal CoT + RLVR) are the default configuration for new models.
3. GUI/computer-use agents are the fastest-growing deployment scenario for VLMs.
4. Watch Qwen3.8 (>1T-parameter multimodal, previewed at WAIC 2026-07; open weights promised but not yet released).
