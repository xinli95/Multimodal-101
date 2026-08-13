# 06 · Text Decoder — Attention You Can Afford

**Position in the pipeline**: `inputs_embeds ──► Gemma4TextModel ──► hidden states`

At 128K–256K context, the decoder's design is dominated by one constraint: the KV cache. Full attention at every layer with full-size heads is unaffordable, so Gemma 4 spends its budget unevenly and deliberately. Most layers see a sliding window (512 tokens on E2B, 1024 on the large sizes); a minority get full global attention with *larger* heads (`global_head_dim=512` vs `head_dim=256`). On E2B, the last 20 of 35 layers do not own KV projections at all — they reuse an earlier layer's. On the large sizes, global layers do not have a separate value projection; the key projection output is reused as the value.

The two families make opposite bets — E2B goes MQA plus aggressive cross-layer sharing, the large models keep more KV heads but collapse K and V — and comparing them is the fastest way to understand what each trick actually costs.

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

## Walkthrough

### 1. The attention schedule

Chapter 01 showed `layer_types` being generated. Here is what it means in practice, straight off the checkpoints:

| | E2B | 31B | 26B-A4B |
|---|---|---|---|
| layers | 35 | 60 | 30 |
| global (`full_attention`) at | 4, 9, 14, 19, 24, 29, 34 | 5, 11, …, 59 | 5, 11, 17, 23, 29 |
| ratio | 1 in 5 | 1 in 6 | 1 in 6 |
| `sliding_window` | 512 | 1024 | 1024 |

Roughly **five sixths of every forward pass sees only a local window.** Global attention — the part whose cost grows quadratically with context — happens seven times in E2B, ten times in the 31B. That is the single biggest reason a 256K context is affordable at all.

The window itself is defined as a predicate rather than a matrix:

```python
def sliding_window_mask_function(sliding_window: tuple[int, int]) -> Callable:
    def inner_mask(batch_idx, head_idx, q_idx, kv_idx) -> bool:
        left_window_size, right_window_size = sliding_window
        dist = q_idx - kv_idx
        left_mask  = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask
    return inner_mask
```

A function of `(q_idx, kv_idx)`, not a `[seq, seq]` tensor. The masking utilities compose these predicates and only materialise a tensor when the attention backend needs one — and FlashAttention never does, it takes the window size as an integer. Note that the function is symmetric in form: it supports a right window too, which is what the audio tower reuses it for (chapter 05) and what `use_bidirectional_attention` needs (chapter 08).

### 2. Two attention regimes, two RoPEs, two head sizes

```python
self.is_sliding = self.layer_type == "sliding_attention"
self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
```

**Global layers get bigger heads**: 512 instead of 256. Rarer, but individually more expressive — the model spends its long-range budget on quality where it spends it at all.

And each regime gets its own rotary embedding. `Gemma4TextRotaryEmbedding` builds one buffer set *per layer type*:

```python
for layer_type in self.layer_types:
    rope_params = self.config.rope_parameters[layer_type]
    ...
    if layer_type == "full_attention" and rope_type == "proportional":
        rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"
    curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, **rope_init_fn_kwargs)
    self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
```

`forward` then selects by name: `inv_freq = getattr(self, f"{layer_type}_inv_freq")`. From chapter 01, the two settings are:

- `sliding_attention` → `rope_type="default"`, θ = 10,000
- `full_attention` → `rope_type="proportional"`, θ = 1,000,000, `partial_rotary_factor=0.25`

The logic is worth stating plainly. A sliding layer's largest possible relative distance is 512, so classic θ=10,000 frequencies resolve it fine and stretching them would waste resolution. A global layer must distinguish positions up to 262,144 apart, so it needs the much flatter θ=1,000,000 — and it rotates only a quarter of its 512 dimensions, leaving the rest position-independent. **Every layer gets exactly the positional resolution its span requires.** Models that use one RoPE everywhere are over-stretching their local layers to serve their global ones.

### 3. QK-norm, and a value norm without parameters

```python
self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
self.scaling = 1.0
```

The same trio as the vision tower. And note `self.scaling = 1.0` — the customary `1/√d` is **gone**, because normalising q and k already fixes their magnitudes. The scale factor was only ever compensating for growth that no longer happens.

Order matters in `forward`: project → norm → RoPE → transpose. Normalising before rotation keeps the rotation acting on unit-ish vectors.

### 4. Cross-layer KV sharing

The most aggressive memory trick in the model, and about fifteen lines.

```python
first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
prev_layers = config.layer_types[:first_kv_shared_layer_idx]
self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])

if not self.is_kv_shared_layer:
    self.k_norm = ...; self.v_norm = ...
    self.k_proj = nn.Linear(...)
    self.v_proj = nn.Linear(...) if not self.use_alternative_attention else None
```

On E2B, `num_kv_shared_layers = 20` of 35 layers, so layers 15–34 have **no `k_proj`, no `v_proj`, no `k_norm`, no `v_norm` at all**. They read the keys and values computed by the last non-sharing layer *of their own type*:

```python
if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.layer_type]
```

`store_full_length_kv` marks which layers are the donors — the last sliding layer and the last global layer before the sharing boundary. Two donors, one per layer type, because a sliding layer's keys and a global layer's keys are rotated with different RoPEs and are not interchangeable.

The comment on the shared branch explains a subtlety that would otherwise look like a missed optimisation:

> *We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache once we are past the sliding window — so we always use `shared_kv_states` instead, even when `past_key_values` is not None.*

The donor's cache has already forgotten what a consumer might need. Correctness beats reuse.

And the loader is told to expect the missing weights rather than warn about them:

```python
for i, layer in enumerate(self.layers):
    if layer.self_attn.is_kv_shared_layer:
        self._keys_to_ignore_on_load_unexpected.extend(
            [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")])
```

**K=V** is the large models' variant of the same idea:

```python
self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
...
value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
```

On global layers only, the value projection is deleted and the key projection's output is used as both. Halves the KV cache for exactly the layers whose cache is full-length. Note it is applied **before** the norms and RoPE diverge — keys get `k_norm` + rotation, values get `v_norm` and no rotation, so the two are not identical downstream despite sharing a projection.

### 5. The detail that makes the whole design click

`Gemma4TextMLP.__init__`:

```python
first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
```

Read that twice. **The double-wide MLP is applied to exactly the layers that gave up their KV projections.** On E2B, layers 0–14 have a 6144-wide MLP and their own keys and values; layers 15–34 have a **12288-wide** MLP and no keys or values of their own.

This is not two independent tricks that happen to coexist. It is one decision: *in the upper half of the network, stop paying for attention parameters and cache, and spend the savings on feed-forward capacity instead.* It reflects a real finding about transformers — upper layers do more feature transformation and less information routing — and Gemma 4 has quietly built it into the architecture. You cannot see this from the config alone; `use_double_wide_mlp: true` looks like a global flag. Only the three lines above reveal that it is conditional.

### 6. Odds and ends that matter

**Embedding scale.** `Gemma4TextScaledWordEmbedding` multiplies the lookup by `√hidden_size`, with a comment preserving a piece of institutional memory:

```python
# Gemma4 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5.
```

The scale is stored as a buffer so it is applied in the same precision the original implementation used. Reproducing a model bit-for-bit means reproducing its rounding.

**`layer_scalar`.** Every decoder layer ends with `hidden_states *= self.layer_scalar`, a `register_buffer` initialised to 1.0 and loaded from the checkpoint — a per-layer residual scale that costs one multiply.

**`final_logit_softcapping = 30.0`** on all sizes, applied in `Gemma4ForCausalLM`: `logits = tanh(logits / cap) * cap`. The same smooth ceiling as the audio tower's attention softcap. It bounds the entropy of the output distribution and stops any single token's logit running away.

## Design space

| Technique | Who | What it saves | What it costs |
|---|---|---|---|
| **GQA** | nearly everyone | KV cache ÷ (heads/kv_heads) | mild quality loss at high ratios |
| **MQA** (`kv_heads=1`) | Gemma 4 E2B, PaLM | maximum GQA saving | more; usually needs training from scratch with it |
| **Sliding + global mix** | Gemma 2/3/4, Mistral, Character.ai | most layers become O(n·w) | long-range info must route through the few global layers |
| **Cross-layer KV sharing** | Gemma 4, CLA, YOCO | cache ÷ (layers/donors) | upper layers cannot form new attention patterns |
| **K=V** | Gemma 4 large | cache ÷ 2 on those layers | keys and values can no longer specialise separately |
| **MLA** | DeepSeek | cache ↓ via low-rank latent | a genuinely different attention formulation |

Gemma 4's contribution is not any single row — it is that it applies four of them at once, and *differently per size*. E2B, which must fit on a phone, picks MQA plus 20 shared layers. The large models, which have quality budget to protect, keep 16 or 8 KV heads and take the milder K=V saving on global layers only. Same codebase, opposite ends of the trade-off curve, selected entirely by config.

The honest limitation: cross-layer sharing means the top 57% of E2B's layers cannot decide *what to look at* — only what to do with what the layer below looked at. Whether that hurts is an empirical question, and the double-wide MLP is the compensation.

## Check yourself

1. Why do sliding and global layers need different RoPE θ? What breaks if you give both θ=10,000?
2. `self.scaling = 1.0` — where did `1/√d` go?
3. Why are there two entries in `shared_kv_states` rather than one?
4. E2B: 35 layers, 20 shared, 1 KV head, head_dim 256/512. Roughly how much KV cache per token, and what would it be without sharing?
5. Under `attention_k_eq_v`, keys and values come from one projection. Name two things that still differ between them downstream.
6. Which layers get the double-wide MLP, and what is the argument for that choice?

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_decoder_anatomy.ipynb` | Print the real `layer_types` pattern and head-dim assignment; reimplement `sliding_window_mask_function` and check it against the library; measure KV cache growth per layer type and quantify what sharing saves | 🟢 CPU (mini config) / 🟡 for real measurements |
