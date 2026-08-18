# 21 · Video Generation

**How this connects to Part I.** [Chapter 05](../05-audio-and-video/index.md) took video *in*, and the striking thing was how little machinery it needed: sample 32 frames, run each through the image tower, write `MM:SS` timestamps into the prompt as text. No temporal attention, no 3D convolution. Understanding a video, it turns out, is mostly understanding a handful of pictures in order.

Generating one is not. The moment you have to *produce* frame 17 rather than merely read it, temporal consistency stops being free — and everything in this chapter exists to buy it back. That asymmetry is the most useful thing to carry across from Part I: a model that reads video can get away with frame sampling; a model that writes video cannot.

The vocabulary still transfers. A video DiT patchifies its latent the way [chapter 04](../04-vision-tower/index.md)'s vision tower patchifies an image, then attends over the resulting grid — only now the grid has a time axis, and the attention bill grows accordingly.

## Learning goals

- Understand the three new problems video adds over images: temporal consistency, compute explosion, audio-video sync
- Master the design trade-offs of 3D causal VAEs and spatiotemporal attention (full 3D vs. factorized)
- Understand why I2V (image-to-video) is the workhorse path in production
- Run an open video model locally (short low-res clips) and compare with the Veo API

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
