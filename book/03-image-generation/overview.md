# 03 · Image Generation

The first chapter on the generation side. In theory we walk the evolution line DDPM → Latent Diffusion → DiT → Flow Matching (seeded in chapter 00); in practice we run FLUX.2 [klein] locally and compare against closed APIs like GPT Image 2.

## Learning goals

- Fully understand the three-piece kit of modern text-to-image models: VAE, text encoder, DiT backbone
- Understand Classifier-Free Guidance, samplers, and step distillation (why klein renders in 0.5s)
- Run FLUX.2 [klein] / SD 3.5 locally; practice prompting and parameter tuning
- Understand LoRA fine-tuning and train a style LoRA of your own

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
