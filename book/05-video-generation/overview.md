# 05 · Video Generation

Image generation extended into time: Video DiT, 3D causal VAE, spatiotemporal attention. In practice we run Wan 2.2 / HunyuanVideo 1.5 locally and call the Veo 3.1 API — and learn why video is today's most compute-hungry modality.

## Learning goals

- Understand the three new problems video adds over images: temporal consistency, compute explosion, audio-video sync
- Master the design trade-offs of 3D causal VAEs and spatiotemporal attention (full 3D vs. factorized)
- Understand why I2V (image-to-video) is the workhorse path in production
- Run an open video model locally (short low-res clips) and compare with the Veo API

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
