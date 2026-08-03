# Theory · VLM

## 1. Architecture evolution

- **Dual-tower alignment** (CLIP, 2021): retrieval/classification only, no dialogue
- **Frozen LLM + bridge**:
  - Flamingo (2022): inject visual features via gated cross-attention
  - BLIP-2 (2023): Q-Former compresses visual features into a fixed number of query tokens
  - LLaVA (2023): the minimalist route — a single MLP projector maps ViT features straight into the LLM embedding space. **This simplest route ultimately won**; it is worth pondering why (data quality > architectural cleverness)
- **The modern standard** (Qwen-VL family, InternVL family): SigLIP-class vision tower + MLP connector + strong LLM, jointly trained with all parameters unfrozen

## 2. Key techniques

- **Dynamic / native resolution**: from fixed 224px to arbitrary-resolution tiling (Qwen2-VL's naive dynamic resolution, InternVL's tiles) — directly determines the ceiling for OCR and document understanding
- **Visual token compression**: a high-res image easily costs thousands of tokens; compression strategies (pooling, pixel-shuffle, Q-Former) trade cost against fine-grained perception
- **Video input**: frame sampling + timestamp encoding (e.g. M-RoPE / time-aware position encoding); long-video understanding rides on long context
- **Multimodal position encoding**: Qwen's M-RoPE — factorize position into time/height/width dimensions

## 3. Training recipe (the three-stage paradigm)

1. **Alignment pre-training**: freeze LLM and ViT, train only the connector (image-text pairs)
2. **Multi-task / continued pre-training**: unfreeze everything; mix OCR, grounding, VQA, interleaved data
3. **SFT + RL**: instruction following and preference alignment; since 2025, **multimodal RLVR** (verifiable rewards) is standard — reasoning VLMs (chain-of-thought over images) are now the default

## 4. Capability boundaries and evaluation

- Benchmarks: MMMU (college-level multi-discipline), MMBench, OCRBench, DocVQA, Video-MME
- Known weak spots: precise spatial relations, counting, hallucination (describing objects that aren't there)
- Frontier directions: GUI agents (screen understanding + action), 3D grounding, very long video

## Key papers

| Paper | Year | Why read it |
|---|---|---|
| CLIP | 2021 | Where it all starts |
| Flamingo | 2022 | The cross-attention bridge route |
| BLIP-2 | 2023 | The Q-Former route |
| LLaVA / LLaVA-1.5 | 2023 | Proof that the minimalist route wins |
| Qwen2-VL → Qwen3-VL tech reports | 2024–2025 | The full picture of a modern SOTA architecture |
| InternVL3.5 tech report | 2025 | The other open-source pole; cascaded RL training |
