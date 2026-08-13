# Theory · Image Generation

## 1. The standard three-piece kit

1. **VAE**: pixels ↔ latent space (8x/16x downsampling); generation happens in latent space
2. **Text encoder**: CLIP/T5 → the new trend is using an LLM/VLM directly (FLUX.2 uses a Mistral-3 VLM, Qwen-Image uses Qwen2.5-VL), injecting world knowledge straight into the conditioning
3. **Backbone**: U-Net (SD1.5/SDXL era) → **MMDiT** (SD3/FLUX: text and image tokens in dual-stream joint attention)

## 2. Training-objective evolution

- DDPM noise prediction → v-prediction → **Rectified Flow / Flow Matching** (SD3, FLUX, Qwen-Image all use it)
- Intuition: straighten the curved denoising trajectory — training is more stable and few-step sampling gets better

## 3. Inference and acceleration

- Sampler genealogy: DDIM → DPM-Solver → Flow Euler
- **Step distillation**: LCM, Turbo/Lightning, adversarial distillation → how FLUX.2 [klein] hits sub-second generation
- CFG and guidance distillation (the FLUX dev line bakes CFG into the model, halving inference compute)

## 4. Controllability and fine-tuning

- **LoRA**: low-rank adaptation, the de-facto standard for style/character customization
- ControlNet / IP-Adapter: structural control and image reference (gradually replaced by in-context conditioning in the DiT era — §6 below)
- Why text rendering is hard: tokenizer granularity + training data; how GPT Image 2 / Qwen-Image broke through

## 5. Evaluation (generation)

- Limits of automatic metrics: FID is effectively dead; GenEval / DPG-Bench (compositionality) and arena blind tests (human preference) are the mainstream
- Key axes: prompt adherence, text rendering, multi-subject consistency, aesthetics

## 6. Editing: paradigm evolution

- **The inpainting era**: user paints a mask + local repaint (SD inpainting)
- **InstructPix2Pix (2022)**: first pure-instruction editing — training triplets (source, instruction, result) synthesized with GPT-3 + Prompt-to-Prompt
- **The in-context conditioning era** (FLUX Kontext, Qwen-Image-Edit, GPT Image, Nano Banana): reference images are encoded as token sequences and jointly attended with the target inside the DiT — multi-reference support falls out naturally

## 7. Editing: the core technical problems

- **Edit locality**: how to guarantee "change only what was asked" — the quality of constructed edit pairs + how conditions are injected architecturally
- **Identity consistency**: keeping a person/IP stable across edits (character consistency is the #1 commercial requirement)
- **Semantic vs. appearance edits**: style transfer / viewpoint rotation (semantic) stresses different capabilities than adding/removing objects (appearance)
- **Text editing**: precise replacement of text inside images (Qwen-Image-Edit's strength, rooted in its VLM text encoder + dual-path conditioning)

## 8. Editing and unified models

- Editing is the intersection of understanding and generation: the model must first *read* the source image (understanding), then reconstruct most of it (generation)
- Hence unified models (GPT Image, Nano Banana, BAGEL, InternVL-U) have a natural advantage at editing — foreshadowing chapter 23

## 9. Editing: evaluation

- GEdit-Bench, ImgEdit, arena-style blind tests
- Three human-eval axes: instruction adherence / fidelity of unedited regions / overall quality

## Key papers

| Paper | Year | Why read it |
|---|---|---|
| DDPM | 2020 | The starting point |
| Latent Diffusion | 2022 | Where SD comes from |
| DiT | 2022 | The Transformer backbone |
| SD3 (MMDiT + RF) | 2024 | The paper that fixed the modern architecture |
| FLUX.1/.2 technical material | 2024–2026 | The strongest current open practice |
| Qwen-Image tech report | 2025 | Chinese text rendering + VLM as text encoder |
| InstructPix2Pix | 2022 | Opened instruction editing; its data-synthesis idea still echoes |
| FLUX.1 Kontext | 2025 | The in-context editing representative |
| Qwen-Image-Edit tech report | 2025 | Dual-path conditioning + text editing |
| Nano Banana / GPT Image blogs & system cards | 2025–2026 | Reference points for the closed capability frontier |
