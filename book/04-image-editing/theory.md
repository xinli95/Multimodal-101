# Theory · Image Editing

## 1. Editing paradigm evolution

- **The inpainting era**: user paints a mask + local repaint (SD inpainting)
- **InstructPix2Pix (2022)**: first pure-instruction editing — training triplets (source, instruction, result) synthesized with GPT-3 + Prompt-to-Prompt
- **The in-context conditioning era** (FLUX Kontext, Qwen-Image-Edit, GPT Image, Nano Banana): reference images are encoded as token sequences and jointly attended with the target inside the DiT — multi-reference support falls out naturally

## 2. Core technical problems

- **Edit locality**: how to guarantee "change only what was asked" — the quality of constructed edit pairs + how conditions are injected architecturally
- **Identity consistency**: keeping a person/IP stable across edits (character consistency is the #1 commercial requirement)
- **Semantic vs. appearance edits**: style transfer / viewpoint rotation (semantic) stresses different capabilities than adding/removing objects (appearance)
- **Text editing**: precise replacement of text inside images (Qwen-Image-Edit's strength, rooted in its VLM text encoder + dual-path conditioning)

## 3. The connection to unified models

- Editing is the intersection of understanding and generation: the model must first *read* the source image (understanding), then reconstruct most of it (generation)
- Hence unified models (GPT Image, Nano Banana, BAGEL, InternVL-U) have a natural advantage at editing — foreshadowing chapter 07

## 4. Evaluation

- GEdit-Bench, ImgEdit, arena-style blind tests
- Three human-eval axes: instruction adherence / fidelity of unedited regions / overall quality

## Key papers

| Paper | Year | Why read it |
|---|---|---|
| InstructPix2Pix | 2022 | Opened instruction editing; its data-synthesis idea still echoes |
| FLUX.1 Kontext | 2025 | The in-context editing representative |
| Qwen-Image-Edit tech report | 2025 | Dual-path conditioning + text editing |
| Nano Banana / GPT Image blogs & system cards | 2025–2026 | Reference points for the closed capability frontier |
