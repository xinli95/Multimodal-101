# Landscape · Image Generation

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

## Open / open-weight

| Model | Org | Size | License | Highlights |
|---|---|---|---|---|
| **FLUX.2 [dev]** | Black Forest Labs | 32B | Non-commercial (license required for commercial) | Highest-quality open-weight tier |
| **FLUX.2 [klein]** | Black Forest Labs | 4B/9B | Apache 2.0 | Released 2026-01; sub-0.5s generation on consumer GPUs (~13GB VRAM) — **the teaching pick** |
| **Qwen-Image-2512** | Alibaba | 20B | Apache 2.0 | Strongest open model in blind tests; standout Chinese/English text rendering |
| SD 3.5 | Stability AI | 2.5B–8B | Community license | The widest ecosystem baseline |
| HiDream-I1 | HiDream | 17B | MIT | Alternative |

## Closed frontier

| Model | Org | Highlights |
|---|---|---|
| **GPT Image 2** | OpenAI | Arena #1; dense text rendering, precise instructed edits, 10-reference composition |
| Nano Banana Pro (Gemini image) | Google | Instructed editing fused with world knowledge |
| Midjourney V8 | Midjourney | The aesthetics ceiling |
| MAI-Image-2.5 | Microsoft | Top of the arena tier |

⚠️ Retired: Imagen 4 API shuts down 2026-08-17; DALL-E 3 has been replaced by the GPT Image line.

## Where things are heading

1. The open-closed gap has narrowed to "leaderboard position" rather than "usable or not"; open wins on control, on-prem, and cost.
2. LLM-ification of the text encoder (world-knowledge injection) is this generation's shared choice.
3. The generation/editing boundary is dissolving — new models ship T2I and I2I together (see chapter 04).
