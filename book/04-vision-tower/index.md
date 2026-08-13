# 04 · Vision Tower — From Patches to Soft Tokens

**Position in the pipeline**: `pixel_values ──► Gemma4VisionModel ──► pooler ──► Gemma4MultimodalEmbedder ──► soft tokens`

This is the chapter where an image becomes something the language model can read. Four stages, each with a design decision worth understanding:

1. **Patch embedder** — linear projection of each 16×16 patch, plus a *learned* 2D position embedding looked up by `(x, y)` coordinates from a table with 10,240 slots per axis
2. **Encoder** — 16 transformer layers with **2D RoPE**: half the head dimensions rotate by the x coordinate, the other half by y. This is what makes "the cat is left of the dog" representable
3. **Pooler** — average-pool 3×3 neighbourhoods *by position*, collapsing 2,520 patches into 280 soft tokens
4. **Multimodal embedder** — project from vision hidden size into the text embedding space, so the result can be dropped into the token sequence like any other embedding

## What you will learn

1. Why a grid needs two position signals (a learned absolute table *and* rotary relative encoding) and what each one buys
2. How 2D RoPE is implemented by splitting head dimensions between axes — and how to write it yourself in fifteen lines
3. How pooling by position, rather than by sequence index, survives variable aspect ratios and padding
4. Where the projection into text space happens, and why that single module is the direct descendant of LLaVA's MLP projector
5. **Design space**: LLaVA's MLP vs. BLIP-2's Q-Former vs. Flamingo's cross-attention vs. Qwen-VL's M-RoPE + native resolution vs. InternVL's tiling — and what each optimises for

## Source map

| Symbol in `modeling_gemma4.py` | Role |
|---|---|
| `Gemma4VisionPatchEmbedder` | Patch projection + learned 2D position table (`_position_embeddings`) |
| `Gemma4VisionRotaryEmbedding`, `apply_multidimensional_rope` | 2D RoPE: per-axis rotation of half the head dims |
| `Gemma4VisionAttention`, `Gemma4VisionEncoderLayer`, `Gemma4VisionEncoder` | The stack |
| `Gemma4VisionPooler`, `_avg_pool_by_positions` | 3×3 pooling driven by patch coordinates |
| `Gemma4VisionModel` | The assembled tower |
| `Gemma4MultimodalEmbedder` | Vision hidden size → text embedding space |
| `Gemma4Model.get_image_features` | The one call that runs all of the above |
| `Gemma4ClippableLinear`, `Gemma4RMSNorm` | Numerical-stability building blocks used throughout |

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_vision_tower_anatomy.ipynb` | Hook every stage and print the shape as one image walks through; reimplement `apply_multidimensional_rope` and `assert_close`; visualise which patches pool into which soft token | 🟢 CPU (mini config) / 🟡 for real weights |
| `02_image_understanding.ipynb` | What the tower buys you in practice with real E2B weights: VQA, OCR, structured document extraction, and grounding with drawn boxes | 🟡 24GB VRAM |
| [`03_compare_qwen3vl.ipynb`](notebooks/03_compare_qwen3vl.ipynb) | The design-space control: Qwen3-VL's native-resolution + 2×2-merge approach on the same tasks, with visual-token counts side by side | 🟡 12GB+ VRAM |
