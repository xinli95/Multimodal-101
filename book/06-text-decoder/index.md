# 06 · Text Decoder — Attention You Can Afford

**Position in the pipeline**: `inputs_embeds ──► Gemma4TextModel ──► hidden states`

At 128K–256K context, the decoder's design is dominated by one constraint: the KV cache. Full attention at every layer with full-size heads is unaffordable, so Gemma 4 spends its budget unevenly and deliberately. Most layers see a 512-token sliding window; a minority get full global attention with *larger* heads (`global_head_dim=512` vs `head_dim=256`). The last few layers do not own KV projections at all — they reuse an earlier layer's. Some global layers do not have a separate value projection either; the key projection output is reused as the value.

Each of those is a specific trade of quality against memory, and each is a few lines of readable code.

## What you will learn

1. How `layer_types` assigns each layer to `sliding_attention` or `full_attention`, and how to read the resulting pattern off a real checkpoint
2. Why a sliding-window layer needs a *different* RoPE than a global layer, and how `Gemma4TextRotaryEmbedding` serves both
3. QK-norm: RMSNorm applied to queries, keys **and** values per head — including the scale-free value norm — and what instability it prevents
4. Cross-layer KV sharing (`num_kv_shared_layers`, `store_full_length_kv`) and K=V projections (`attention_k_eq_v`): what they save and what they cost
5. `final_logit_softcapping` and `Gemma4TextScaledWordEmbedding` — the small numerical details that make training stable
6. Measure the KV cache yourself and see the savings in megabytes rather than in prose

## Source map

| Symbol in `modeling_gemma4.py` | Role |
|---|---|
| `Gemma4TextAttention` | GQA + q/k/v RMSNorm, sliding vs. global branching, KV sharing, K=V |
| `Gemma4TextRotaryEmbedding` (`compute_default_rope_parameters`) | Per-layer-type RoPE |
| `sliding_window_mask_function` | The window as a boolean predicate over `(q_idx, kv_idx)` |
| `Gemma4TextMLP`, `Gemma4TextDecoderLayer` | The block |
| `Gemma4TextModel.forward` | The loop, the shared-KV dict, and mask construction |
| `Gemma4ForCausalLM` | lm_head + logit softcapping |
| `repeat_kv`, `eager_attention_forward`, `apply_rotary_pos_emb` | The reference implementations to read before the fused kernels |

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_decoder_anatomy.ipynb` | Print the real `layer_types` pattern and head-dim assignment; reimplement `sliding_window_mask_function` and check it against the library; measure KV cache growth per layer type and quantify what sharing saves | 🟢 CPU (mini config) / 🟡 for real measurements |
