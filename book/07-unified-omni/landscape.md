# Landscape · Unified Multimodal Models

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

## Open / open-weight

| Model | Org | Size | License | Coverage |
|---|---|---|---|---|
| **Qwen3.5-Omni** | Alibaba | from 30B MoE | Apache 2.0 (open builds) | In: text/image/audio/video; out: text + speech; 256k context, 113-language ASR; beats Gemini 3.1 Pro on key audio tasks |
| **BAGEL** | ByteDance Seed | 7B active / 14B | Apache 2.0 | Understand + generate + edit; MoT architecture; Hyper-BAGEL acceleration (no v2 — still the 2025 original) |
| **InternVL-U** | Shanghai AI Lab | 4B | MIT | Released 2026-03; understanding/reasoning/generation/editing in one — **the local-practice pick** (small) |
| Janus-Pro | DeepSeek | 1B/7B | MIT | Early-2025 model; use for architecture teaching, not as SOTA |
| NExT-OMNI | research | - | open | Discrete flow matching any-to-any; frontier reference |

## Closed frontier

| Model | Org | Form |
|---|---|---|
| GPT-5.x + GPT Image 2 | OpenAI | Native multimodal input + unified image generation/editing |
| Gemini 3.x + Nano Banana Pro | Google | Unified architecture from day one, omni in and out |
| Qwen3.8-Max (preview) | Alibaba | >1T-parameter multimodal, previewed 2026-07; open weights promised, not yet delivered |

## Where things are heading

1. The entire closed frontier is now unified models; open source followed en masse through 2026 (InternVL-U, the Qwen-Omni line).
2. "Thinking before generation" is becoming the new lever on generation quality.
3. Real-time full-duplex voice interaction is the omni models' main deployment form (assistants, companionship, support).
4. Understanding-generation mutual benefit is not yet fully proven — watch the ROVER/Unison line of benchmarks.
