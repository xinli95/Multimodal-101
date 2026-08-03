# Landscape · Image Editing

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

## Open / open-weight

| Model | Org | License | Highlights |
|---|---|---|---|
| **Qwen-Image-Edit** (latest 2511+ builds) | Alibaba | Apache 2.0 | #1 at precise text editing; all-round appearance + semantic edits — the open default |
| **FLUX.2 family (editing built in)** | BFL | klein: Apache 2.0 | Since FLUX.2, generation/editing are one model with multi-reference; Kontext [dev] is the previous generation |
| Step1X-Edit, SeedEdit (open builds) | StepFun / ByteDance | varies | Alternatives |

## Closed frontier

| Model | Org | Highlights |
|---|---|---|
| **Nano Banana Pro** | Google (Gemini image) | Among the strongest all-round instructed editors; character consistency, world knowledge |
| **GPT Image 2** | OpenAI | Precise local edits + up to 10 reference images |
| SeedEdit / Seedream | ByteDance | The leading closed line from China |

## Quick chooser

| Scenario | Pick |
|---|---|
| Replacing text inside images (especially Chinese) | Qwen-Image-Edit |
| On-prem + fine-tunable | Qwen-Image-Edit (LoRA-customizable edit behavior) |
| Maximum character consistency for commercial content | Nano Banana Pro / GPT Image 2 |
| Sub-second interactive editing | FLUX.2 [klein] |

## Where things are heading

1. "Editing" is being absorbed into "generation": standalone editors will disappear into unified multimodal models.
2. Fine-tuning open editors (e.g. Qwen-Image-Edit + LoRA) can beat closed generalists in vertical domains — a tutorial experiment worth running.
