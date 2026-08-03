# Theory · Video Generation

## 1. From image to video: three new problems

1. **Temporal consistency**: object identity, motion physics, lighting continuity
2. **Compute explosion**: tokens = space × time; 5 seconds of 720p is already hundreds of thousands of tokens → every design decision orbits compute savings
3. **Audio-video sync**: lip sync and sound-effect alignment (the watershed capability of the current generation)

## 2. Core components

- **3D Causal VAE**: joint spatiotemporal compression (typically 4x temporal + 8x8 spatial); the causal design lets the first frame encode independently → the foundation of I2V
- **Video DiT**:
  - Factorized attention (alternating spatial and temporal layers): cheaper, the early mainstream
  - Full 3D attention: better but expensive; standard on flagships
  - 3D RoPE position encoding
- **Flow Matching** as the training objective (identical to the image chapter — it transfers directly)

## 3. Conditioning and control

- T2V vs. **I2V** (first-frame conditioning — the production workhorse: compose the shot with an image model first, then let the video model move it)
- Motion control: camera trajectories, reference videos, pose sequences
- Multi-reference input (Seedance 2.5 takes 50 references): anchoring characters/scenes/styles
- Joint audio generation: LTX-2's single-forward-pass audio+video vs. Veo's cascaded approach

## 4. Long video and efficiency

- Autoregressive blockwise / streaming generation (the Self-Forcing, CausVid line)
- Distillation: few-step video variants (FastWan et al.)
- Why a 30-second single shot (Seedance 2.5) was a 2026 milestone

## 5. Evaluation

- VBench / VBench-2.0 (dimensional automatic eval), arena blind tests
- Physical plausibility remains the shared weakness (fluids, collisions, hand interactions)

## Key papers

| Paper | Year | Why read it |
|---|---|---|
| Sora tech report | 2024 | Popularized spacetime patches |
| Wan 2.1/2.2 tech reports | 2025 | The most complete technical disclosure in open video |
| HunyuanVideo / 1.5 | 2024–2025 | Another full open implementation |
| LTX-Video / LTX-2 | 2024–2026 | Real-time generation + joint audio-video |
| Self-Forcing | 2025 | The autoregressive long-video direction |
