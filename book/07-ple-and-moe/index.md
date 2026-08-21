# 07 · PLE and MoE — Two Ways to Spend Parameters

**Position in the pipeline**: inside every `Gemma4TextDecoderLayer`.

Two mechanisms that both answer "how do we get more capability without paying for it at every token", from opposite directions — and neither is on in every checkpoint. PLE runs on **E2B and E4B** (`hidden_size_per_layer_input=256`) and is switched off on the large models (`0`). MoE is the reverse: **26B-A4B only**. Chapter 01 §5 has the full table. That split is the first thing to explain, and it turns out to explain itself: PLE buys capacity where parameters are scarce, MoE buys it where compute is scarce.

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
| [`Gemma4TextModel.get_per_layer_inputs`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1744) | The token-identity half |
| [`Gemma4TextModel.project_per_layer_inputs`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1788) | The context-aware half, the `1/√2` merge, and the multimodal fallback |
| [`Gemma4TextScaledWordEmbedding`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1465) | Also used for the PLE table, with its own `embed_scale` |
| [`Gemma4TextRouter`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1339), [`Gemma4TextExperts`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1300) | The MoE path |
| [`Gemma4TextDecoderLayer.forward`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1405) | Where the per-layer input is actually consumed |
| [`Gemma4PreTrainedModel._resize_per_layer_embeddings`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1573) | What vocabulary resizing has to do about the second table |

Config fields: [`vocab_size_per_layer_input`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L180), [`hidden_size_per_layer_input`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L181) (256), [`enable_moe_block`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L186), [`num_experts`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L188), [`top_k_experts`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L189), [`moe_intermediate_size`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L190), [`use_double_wide_mlp`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L187).

## Walkthrough — PLE

### 1. It is a gate, not an addition

The docs describe PLE as "an auxiliary residual signal into each decoder layer", which is true but undersells it. Here is what a layer actually does with its per-layer input, at the very end of `Gemma4TextDecoderLayer.forward`:

```python
if self.hidden_size_per_layer_input:
    residual = hidden_states
    hidden_states = self.per_layer_input_gate(hidden_states)   # hidden_size -> 256
    hidden_states = self.act_fn(hidden_states)                 # gelu_pytorch_tanh
    hidden_states = hidden_states * per_layer_input            # <- elementwise gate
    hidden_states = self.per_layer_projection(hidden_states)   # 256 -> hidden_size
    hidden_states = self.post_per_layer_input_norm(hidden_states)
    hidden_states = residual + hidden_states
```

The layer squeezes its own hidden state down to 256 dimensions, activates it, **multiplies elementwise by the per-layer embedding**, projects back up, normalises, and adds. The per-layer embedding is not added to the stream — it *modulates a low-rank bottleneck view of the stream*.

That distinction matters. An additive signal would inject the same content regardless of what the layer computed. A multiplicative gate lets the embedding say *"for this token, at this layer, amplify these 256 directions and suppress those"*. It is a per-token, per-layer, learned conditioning channel, costing two thin projections per layer.

### 2. The token-identity half

```python
self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
    config.vocab_size_per_layer_input,                                   # 262144
    config.num_hidden_layers * config.hidden_size_per_layer_input,       # 35 * 256 = 8960
    self.padding_idx,
    embed_scale=config.hidden_size_per_layer_input**0.5,                 # sqrt(256) = 16
)
```

A **second full-vocabulary embedding table**, whose row for a token packs that token's contribution to all 35 layers end to end. Look it up once, reshape, done:

```python
return self.embed_tokens_per_layer(input_ids).reshape(
    *input_ids.shape, self.config.num_hidden_layers, self.hidden_size_per_layer_input)
```

The packed layout is not incidental — it turns 35 separate embedding lookups into one gather. On E2B that table is 262144 × 8960 ≈ 2.35B parameters, comfortably larger than the rest of the model. This is the trick that makes "E2B" mean *effective* 2B: the table is enormous but only one row per token is ever touched, so it can live in slower memory while the compute-heavy weights stay resident. That is also why PLE belongs to the small sizes (chapter 01 §5) — it buys per-token capacity that costs memory rather than FLOPs, which is exactly the trade an on-device model wants and a 31B server model does not.

### 3. The context-aware half

```python
per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
per_layer_projection = per_layer_projection.reshape(*inputs_embeds.shape[:-1],
                                                    self.config.num_hidden_layers,
                                                    self.hidden_size_per_layer_input)
per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

if per_layer_inputs is None:
    return per_layer_projection

return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale   # 2**-0.5
```

One `nn.Linear(hidden_size, num_layers * ple_dim)` applied to the input embeddings, scaled by `1/√hidden_size`, reshaped per layer, RMSNormed. Then the two halves are summed and scaled by `1/√2` — the standard variance-preserving combination of two roughly unit-variance signals.

Note what each half knows. The token-identity half depends only on *which token this is*. The context-aware half depends on `inputs_embeds`, which for a multimodal position is a **soft token from the vision or audio tower**. So the gate for an image position is computed from the image content.

### 4. The multimodal branch, and a genuinely strange fallback

```python
if per_layer_inputs is None:
    return per_layer_projection
```

Soft tokens have no `input_ids` — a vision token is not in the vocabulary — so there is nothing to look up in `embed_tokens_per_layer`. Multimodal positions get the context-aware half only. Chapter 08 shows how `Gemma4Model.forward` assembles a per-layer-inputs tensor with pad embeddings standing in for the multimodal spans.

And then there is this, in `get_per_layer_inputs` when you call the model with `inputs_embeds` but no `input_ids`:

```python
input_ids = ((inputs_embeds[:, :, None, :]
              == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5)
             .all(dim=3).nonzero()[:, 2])
```

It **inverts the embedding by brute-force exact match against the entire vocabulary** to recover the token IDs. A `[batch, seq, 262144, hidden]` broadcast comparison — with a `try/except` that tells you what went wrong when your embeddings are not bit-identical to table rows:

> *Since Gemma4 needs to reverse the embedding to compute another embedding, make sure you provide exact `inputs_embeds`*

This exists so `generate(inputs_embeds=...)` works at all, and the docstring on `forward` tells you to avoid it:

> *If calling the `forward` with `inputs_embeds` instead of `input_ids`, you should probably precompute them and forward them along `inputs_embeds`, otherwise recomputing them needs to reverse the main embedding, which is expensive.*

Take the advice. Precompute `per_layer_inputs` and pass it. This is a good illustration of the cost of a second, parallel embedding path: every API that assumed "embeddings are the model's entry point" needs a special case.

Two guards enforce the contract:

```python
if (input_ids is None) ^ (inputs_embeds is not None): raise ValueError(...)
if input_ids is not None and per_layer_inputs is not None: raise ValueError(...)
```

### 5. Resizing the vocabulary touches two tables

`Gemma4PreTrainedModel` carries `_resize_per_layer_embeddings` alongside the usual `resize_token_embeddings`. Add a special token to a PLE model and you must grow `embed_tokens` **and** `embed_tokens_per_layer`, whose row width is `num_layers × 256`. Relevant in chapter 10 if you add tokens during fine-tuning.

## Walkthrough — MoE

### 6. Dense and sparse, added together

`enable_moe_block` is `True` only on 26B-A4B. The surprise is that the MoE block does not *replace* the dense MLP:

```python
residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)                        # dense MLP, always runs

if self.enable_moe_block:
    hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

    hidden_states_flat = residual.reshape(-1, residual.shape[-1])   # note: pre-MLP states
    _, top_k_weights, top_k_index = self.router(hidden_states_flat)
    hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
    hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
    hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2.reshape(residual.shape))

    hidden_states = hidden_states_1 + hidden_states_2            # dense + sparse
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

The dense MLP is a **shared expert** that every token gets, and the routed experts add specialisation on top. This is the DeepSeekMoE-style arrangement, and it solves a real MoE failure mode: with pure routing, every expert has to independently relearn the common-case transformation, wasting capacity. Give everyone a shared path and the routed experts can afford to be genuinely specialised.

Two details in the plumbing. The router reads `residual` — the hidden state *before* the pre-FFN norm and the dense MLP — so routing decisions are made on the layer's input, not on a partially transformed intermediate. And the two branches get separate norms (`post_feedforward_layernorm_1` / `_2`) before being summed, so their scales can differ.

The shapes for 26B-A4B: `intermediate_size = 2112` for the dense path, `moe_intermediate_size = 704` per expert, `num_experts = 128`, `top_k_experts = 8`. Eight experts × 704 = 5632 of routed width, plus 2112 of shared width, per token — out of 128 × 704 = 90,112 available. That is the 26B/4B ratio.

### 7. The router

```python
hidden_states = self.norm(hidden_states)                       # RMSNorm, with_scale=False
hidden_states = hidden_states * self.scale * self.scalar_root_size   # learned per-dim scale, then 1/sqrt(hidden)
expert_scores = self.proj(hidden_states)                       # [tokens, 128]
router_probabilities = F.softmax(expert_scores, dim=-1, dtype=torch.float32)
top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.config.top_k_experts, dim=-1)
top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)       # renormalise the kept 8
top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
```

Softmax-then-top-k-then-renormalise is the standard recipe. Two Gemma-specific touches: a parameter-free RMSNorm followed by a *learned per-dimension* `scale` before projection (the router gets to decide which features are worth routing on), and a learned `per_expert_scale` applied after renormalisation — a per-expert gain the model can use to down-weight experts that are systematically over-selected. Note there is no auxiliary load-balancing loss here; balancing lives in training, not inference.

### 8. Expert dispatch

```python
expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

for expert_idx in expert_hit:
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
    current_state = hidden_states[token_idx]
    gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
    current_hidden_states = self.act_fn(gate) * up
    current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
    final_hidden_states.index_add_(0, token_idx, current_hidden_states)
```

Expert weights are stored as **3D tensors** — `gate_up_proj` is `[num_experts, 2*intermediate, hidden]` — not 128 separate `nn.Linear` modules. That is what lets a distributed backend do a grouped GEMM over all experts at once (see `base_model_ep_plan`'s `"grouped_gemm"` in chapter 01).

This reference loop iterates only over experts that were actually selected (`expert_hit`), gathers their tokens, runs a SwiGLU, scales by the routing weight, and scatter-adds. It is readable and slow; `@use_experts_implementation` on the class swaps in a fused kernel when one is available. Read this version to understand MoE, then never run it in production.

## Design space

**PLE** is close to unique. The nearest relatives:

- **Adapters / LoRA** also inject low-rank per-layer signals, but the signal is *learned per task* and identical for every token. PLE's is per token, from a table, learned in pretraining.
- **Prefix tuning / prompt tuning** add per-layer learned vectors, again token-independent.
- **Gemma 3n's PLE** is the direct ancestor; Gemma 4 inherits it for the same reason (on-device inference where memory bandwidth beats FLOPs).
- **Mixture-of-depths / early-exit** also vary per-token compute, but by skipping layers rather than conditioning them.

The right way to think about PLE is as **a very large, very sparse parameter store that costs memory instead of compute** — and therefore as a technique whose value depends entirely on your deployment. On a phone with slow flash and a small NPU, trading FLOPs for a table lookup is a win. On an H100 serving batched requests, the table is just weight to move, which is why 31B sets it to 0.

**MoE** is well-trodden by comparison. Gemma 4's version — shared dense expert plus 128 fine-grained routed experts at top-8 — is essentially the DeepSeekMoE recipe, and sits opposite Mixtral's coarse 8-experts-top-2. Fine-grained routing gives combinatorially more expert *combinations* per token (C(128,8) is astronomically larger than C(8,2)) at the cost of more routing overhead and harder load balancing.

Read together, the two mechanisms are the same idea at different scales: **most parameters should not run for most tokens.** PLE achieves it with an embedding table indexed by token identity; MoE achieves it with a learned router. Gemma 4 uses the first where memory is the constraint and the second where compute is.

## Check yourself

1. PLE is described as a residual signal. Where in the layer is it actually applied, and why is "gate" a better word than "add"?
2. `embed_tokens_per_layer` is bigger than the rest of E2B put together. Why is that acceptable, and what does it imply about where it should be stored?
3. An image soft token reaches a decoder layer. Which half of its per-layer input is present, and which is missing?
4. Why does `get_per_layer_inputs` contain code that inverts an embedding table, and how do you avoid triggering it?
5. In the MoE block, the router reads `residual` rather than the post-MLP hidden state. Why does that matter?
6. 26B-A4B: 128 experts, top-8, `moe_intermediate_size=704`, dense `intermediate_size=2112`. How much FFN width does one token actually see, and what fraction of the available width is that?
7. Why does adding a special token to E2B require resizing two embedding tables?

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_ple_ablation.ipynb` | Reimplement both PLE halves and `assert_close`; then zero the per-layer input and watch generation degrade — the fastest way to believe a mechanism matters | 🟡 24GB (E2B) |
| `02_moe_routing.ipynb` | A small MoE config with random weights: trace a token through router → experts → combine, then plot expert load across a batch | 🟢 CPU |
