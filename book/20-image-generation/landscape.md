# Landscape · Image Generation

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

## Generation · open / open-weight

| Model | Org | Size | License | Highlights |
|---|---|---|---|---|
| **FLUX.2 [dev]** | Black Forest Labs | 32B | Non-commercial (license required for commercial) | Highest-quality open-weight tier |
| **FLUX.2 [klein]** | Black Forest Labs | 4B/9B | Apache 2.0 | Released 2026-01; sub-0.5s generation on consumer GPUs (~13GB VRAM) — **the teaching pick** |
| **Qwen-Image-2512** | Alibaba | 20B | Apache 2.0 | Strongest open model in blind tests; standout Chinese/English text rendering |
| SD 3.5 | Stability AI | 2.5B–8B | Community license | The widest ecosystem baseline |
| HiDream-I1 | HiDream | 17B | MIT | Alternative |

## Generation · closed frontier

| Model | Org | Highlights |
|---|---|---|
| **GPT Image 2** | OpenAI | Arena #1; dense text rendering, precise instructed edits, 10-reference composition |
| Nano Banana Pro (Gemini image) | Google | Instructed editing fused with world knowledge |
| Midjourney V8 | Midjourney | The aesthetics ceiling |
| MAI-Image-2.5 | Microsoft | Top of the arena tier |

⚠️ Retired: Imagen 4 API shuts down 2026-08-17; DALL-E 3 has been replaced by the GPT Image line.

## Generation · where things are heading

1. The open-closed gap has narrowed to "leaderboard position" rather than "usable or not"; open wins on control, on-prem, and cost.
2. LLM-ification of the text encoder (world-knowledge injection) is this generation's shared choice.
3. The generation/editing boundary is dissolving — new models ship T2I and I2I together (see the editing half of this page).

---

## Editing · open / open-weight

| Model | Org | License | Highlights |
|---|---|---|---|
| **Qwen-Image-Edit** (latest 2511+ builds) | Alibaba | Apache 2.0 | #1 at precise text editing; all-round appearance + semantic edits — the open default |
| **FLUX.2 family (editing built in)** | BFL | klein: Apache 2.0 | Since FLUX.2, generation/editing are one model with multi-reference; Kontext [dev] is the previous generation |
| Step1X-Edit, SeedEdit (open builds) | StepFun / ByteDance | varies | Alternatives |

## Editing · closed frontier

| Model | Org | Highlights |
|---|---|---|
| **Nano Banana Pro** | Google (Gemini image) | Among the strongest all-round instructed editors; character consistency, world knowledge |
| **GPT Image 2** | OpenAI | Precise local edits + up to 10 reference images |
| SeedEdit / Seedream | ByteDance | The leading closed line from China |

## Editing · quick chooser

| Scenario | Pick |
|---|---|
| Replacing text inside images (especially Chinese) | Qwen-Image-Edit |
| On-prem + fine-tunable | Qwen-Image-Edit (LoRA-customizable edit behavior) |
| Maximum character consistency for commercial content | Nano Banana Pro / GPT Image 2 |
| Sub-second interactive editing | FLUX.2 [klein] |

## Editing · where things are heading

1. "Editing" is being absorbed into "generation": standalone editors will disappear into unified multimodal models (chapter 23).
2. Fine-tuning open editors (e.g. Qwen-Image-Edit + LoRA) can beat closed generalists in vertical domains — a tutorial experiment worth running.
