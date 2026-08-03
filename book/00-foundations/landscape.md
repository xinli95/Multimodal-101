# Landscape · Foundation Components

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

## Vision encoders (the backbone of VLMs / generators)

| Model | Org | License | Notes |
|---|---|---|---|
| SigLIP 2 | Google | Apache 2.0 | The most common vision tower in current open VLMs |
| CLIP (ViT-L/14) | OpenAI | MIT | Classic baseline, best for teaching |
| DINOv2/v3 | Meta | Apache 2.0 | Self-supervised line, strong for dense prediction |

## Text encoders (conditioning side of generators)

- T5/Flan-T5 remains widely used for text-to-image conditioning; the newer trend (FLUX.2, Qwen-Image) is to use an LLM/VLM directly as the text encoder.

## Paradigm status in one paragraph

- **Understanding**: autoregressive LLM + vision encoder is the overwhelming mainstream (see chapter 01).
- **Image/video generation**: Flow Matching + DiT is now standard equipment (chapters 03/05).
- **Audio generation**: codec-token autoregressive LMs dominate (chapter 06).
- **Unified models**: AR + diffusion hybrids (BAGEL) and discrete flow matching (NExT-OMNI) are both being explored (chapter 07).
