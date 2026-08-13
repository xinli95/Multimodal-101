# Theory · Unified Multimodal Models

## 1. Why unification is hard

- Understanding prefers **continuous, semantic** visual features (CLIP-family encoders)
- Generation prefers **detail-faithful** representations (VQ tokens or VAE latents + diffusion)
- One representation that must both "understand" and "paint" is an inherent conflict → every architectural difference is a different answer to this tension

## 2. The four architecture routes

1. **Pure autoregressive + discrete visual tokens** (Chameleon, Emu3): the most elegant; image quality chronically suffers
2. **Decoupled encoders** (Janus/Janus-Pro): understanding goes through SigLIP, generation through VQ, sharing one LLM — a simple, effective compromise
3. **AR + Diffusion hybrid** (BAGEL; GPT Image presumably in this family): the LLM handles understanding and planning, a diffusion expert handles pixels; BAGEL's MoT (Mixture-of-Transformer-Experts) lets the two share attention context
4. **Discrete flow matching unification** (NExT-OMNI et al.): the research frontier — one training objective for all modalities

## 3. The special architecture for real-time speech

- **Thinker-Talker** (Qwen3-Omni/3.5-Omni): the Thinker produces text/reasoning while the Talker streams speech tokens in parallel — solving "think while speaking"
- Full-duplex dialogue: interruption handling, bidirectional streams
- The link back to chapter 22's codec LMs: what the Talker generates *are* codec tokens

## 4. What unification buys (why bother)

- **World knowledge injected into generation**: when editing, the model *knows* what things should look like (the core of the Nano Banana demos)
- **Cross-modal reasoning chains**: think before painting (thinking-before-generation)
- **Interleaved generation**: mixed text-image output, visual chains of thought
- Open question: does unification actually help both sides? (Benchmarks like ROVER show mutual benefit is not yet consistent)

## 5. Evaluation

- Understanding side reuses the MMMU family; generation side GenEval/GEdit; unified-specific: ROVER, Unison

## Key papers

| Paper | Year | Why read it |
|---|---|---|
| Chameleon | 2024 | The pure-AR representative |
| Janus / Janus-Pro | 2024–2025 | Decoupled encoding, clearest for teaching |
| BAGEL (Emerging Properties in Unified Multimodal Pretraining) | 2025 | MoT + emergent-ability analysis — this chapter's core close-read |
| Qwen3-Omni / Qwen3.5-Omni tech reports | 2025–2026 | Thinker-Talker real-time omni |
| InternVL-U | 2026 | A 4B unified model you can reproduce locally |
