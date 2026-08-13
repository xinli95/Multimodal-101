# 20 · Image Generation and Editing

**How this connects to Part I.** Gemma 4 takes pixels *in* and produces text. This chapter runs the arrow the other way: text in, pixels out. The vocabulary transfers more than you would expect — a DiT patchifies its latent exactly the way Gemma 4's vision tower patchifies an image, and both then run a transformer over the resulting token grid. What differs is the objective (denoise/transport rather than predict-next-token) and the fact that the token grid has to be turned back into pixels at the end.

Generation and editing were separate chapters until FLUX.2 and Qwen-Image-Edit made them one model, so they are one chapter here. In theory we walk DDPM → Latent Diffusion → DiT → Flow Matching, then instruction editing and in-context conditioning. In practice we run FLUX.2 [klein] locally and compare against closed APIs.

## Learning goals

- Fully understand the three-piece kit of modern text-to-image models: VAE, text encoder, DiT backbone
- Understand Classifier-Free Guidance, samplers, and step distillation (why klein renders in 0.5s)
- Understand the shift from inpainting (paint a mask) to instruction editing (pure language), and in-context conditioning: reference images as sequence conditions in joint attention with the target
- Grasp editing's two hard problems: edit locality (don't touch what shouldn't change) and identity consistency
- Run FLUX.2 [klein] / SD 3.5 locally; practice prompting and parameter tuning; train a style LoRA of your own

## Contents

- [theory.md](theory.md) — principles and key papers (§1–5 generation, §6–9 editing)
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
