# 07 · PLE and MoE — Two Ways to Spend Parameters

**Position in the pipeline**: inside every `Gemma4TextDecoderLayer`.

Two mechanisms that both answer "how do we get more capability without paying for it at every token", from opposite directions.

**Per-Layer Embeddings (PLE)** is the unusual one, and the reason this chapter exists. Instead of a single embedding at the input that every layer must preserve through 30 blocks of residual stream, Gemma 4 feeds *each decoder layer* its own auxiliary embedding signal. It is built from two components, summed and scaled by `1/√2`:

- **token identity** — look `input_ids` up in a second embedding table, `embed_tokens_per_layer`, whose weight is `[vocab_size_per_layer_input, num_hidden_layers × hidden_size_per_layer_input]`: every layer's slice is packed into one row per token
- **context aware** — project the actual `inputs_embeds` through a linear layer, scale by `1/√hidden_size`, reshape per layer, and RMSNorm it

For multimodal positions there are no `input_ids` to look up — a soft token is not in the vocabulary — so only the context-aware half contributes. That detail is a small window into how the whole design has to accommodate non-text tokens.

**MoE** is the familiar one: in the 26B-A4B checkpoint, eligible layers replace the dense MLP with a router and a bank of experts, of which `top_k_experts` fire per token. 26B of parameters, ~4B of compute.

## What you will learn

1. Both halves of the PLE pipeline, reimplemented from `input_ids` to the per-layer tensor, checked against the library
2. Why the packed `[vocab, num_layers × ple_dim]` layout exists and how it is reshaped
3. What actually happens to the model's output when you zero PLE out — an ablation you can run in a minute
4. The MoE path: routing, expert dispatch, `top_k_experts`, `moe_intermediate_size`, and what expert load imbalance looks like when you plot it
5. `use_double_wide_mlp` and the fused gate/up projection — a reminder that most "architecture" is layout

## Source map

| Symbol in `modeling_gemma4.py` | Role |
|---|---|
| `Gemma4TextModel.get_per_layer_inputs` | The token-identity half |
| `Gemma4TextModel.project_per_layer_inputs` | The context-aware half, the `1/√2` merge, and the multimodal fallback |
| `Gemma4TextScaledWordEmbedding` | Also used for the PLE table, with its own `embed_scale` |
| `Gemma4TextRouter`, `Gemma4TextExperts` | The MoE path |
| `Gemma4TextDecoderLayer.forward` | Where the per-layer input is actually consumed |
| `Gemma4PreTrainedModel._resize_per_layer_embeddings` | What vocabulary resizing has to do about the second table |

Config fields: `vocab_size_per_layer_input`, `hidden_size_per_layer_input` (256), `enable_moe_block`, `num_experts`, `top_k_experts`, `moe_intermediate_size`, `use_double_wide_mlp`.

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_ple_ablation.ipynb` | Reimplement both PLE halves and `assert_close`; then zero the per-layer input and watch generation degrade — the fastest way to believe a mechanism matters | 🟡 24GB (E2B) |
| `02_moe_routing.ipynb` | A small MoE config with random weights: trace a token through router → experts → combine, then plot expert load across a batch | 🟢 CPU |
