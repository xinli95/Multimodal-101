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

## Walkthrough

`Gemma4VisionModel.forward` is short enough to quote almost in full, and it is the spine of the chapter:

```python
pooling_kernel_size = self.config.pooling_kernel_size
output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)

padding_positions = (pixel_position_ids == -1).all(dim=-1)
inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
output = self.encoder(inputs_embeds=inputs_embeds,
                      attention_mask=~padding_positions,
                      pixel_position_ids=pixel_position_ids, **kwargs)
hidden_states, pooler_mask = self.pooler(output.last_hidden_state, pixel_position_ids,
                                         padding_positions, output_length)
hidden_states = hidden_states[pooler_mask]      # strip padding
if self.config.standardize:
    hidden_states = (hidden_states - self.std_bias.float()) * self.std_scale.float()
```

Notice what the `(-1, -1)` sentinel from chapter 03 buys: one line, `padding_positions = (pixel_position_ids == -1).all(dim=-1)`, and every subsequent stage knows which of the 2520 slots are real. The positions tensor is doing the job an attention mask would normally do, *and* the job of a coordinate system, simultaneously.

### 1. Patch embedder: where [0,1] becomes [-1,1]

```python
def forward(self, pixel_values, pixel_position_ids, padding_positions):
    # Gemma4 applies no normalization and instead scales in model code
    pixel_values = 2 * (pixel_values - 0.5)
    hidden_states = self.input_proj(pixel_values)
    position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
    return hidden_states + position_embeddings
```

There it is — the missing normalisation from chapter 03, one line of arithmetic inside the model. `input_proj` is a single bias-free `nn.Linear(3 * 16², hidden_size)`: 768 → 768 on E2B, 768 → 1152 on the large sizes. That is the entire "vision stem". No convolutions, no ResNet, no patch-merging pyramid.

### 2. Two position signals, and why both

The learned table is deliberately enormous:

```python
self.position_embedding_table = nn.Parameter(torch.ones(2, self.position_embedding_size, self.hidden_size))
```

Shape `(2, 10240, hidden_size)` — one table per axis, 10,240 slots each. At 16 pixels per patch that covers images up to ~164,000 pixels on a side. It will never be exhausted, which is the point: the lookup is by *absolute coordinate*, so nothing has to be interpolated when the aspect ratio changes. Contrast with ViT's fixed 14×14 grid of position embeddings, which every high-resolution VLM has to bilinearly resize at load time.

The lookup sums the two axes:

```python
clamped_positions = pixel_position_ids.clamp(min=0)
x_emb = F.embedding(clamped_positions[..., 0], self.position_embedding_table[0])
y_emb = F.embedding(clamped_positions[..., 1], self.position_embedding_table[1])
position_embeddings = x_emb + y_emb
position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
```

Note the `clamp(min=0)` and its comment: padding positions are `-1`, which would be an invalid index, so they are clamped to 0 purely so the lookup does not crash — and then zeroed unconditionally. Clamp-then-mask is a pattern you will see three more times in this codebase; it exists because a gather has to be valid even for entries you are about to throw away.

**Then RoPE does it again, relatively.** The learned table gives absolute position. `Gemma4VisionRotaryEmbedding` gives relative position, per axis, and the docstring in `compute_default_rope_parameters` flags the subtlety:

```python
# The reference implementation computes RoPE frequencies INDEPENDENTLY
# for each spatial dimension using the partitioned head_dim (head_dim // ndim),
# so both x and y dimensions get identical frequency ranges.
# This is different from splitting the global inv_freq between dimensions.
spatial_dim = dim // 2
inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2).float() / spatial_dim))
```

Read that carefully, because it is the easy thing to get wrong. You do **not** take the usual 1D `inv_freq` of length `head_dim/2` and give half of it to x and half to y — that would give the two axes different frequency bands and make horizontal and vertical distance incomparable. Instead each axis gets its *own* full frequency spectrum computed over `head_dim/2` channels. With `head_dim=64`, `spatial_dim=32`, and each of x and y rotates 32 channels across the same range of frequencies. Symmetric by construction.

`forward` then loops over the two axes and concatenates:

```python
for i in range(2):
    dim_position_ids = position_ids[:, :, i]
    freqs = (inv_freq_expanded.float() @ dim_position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    all_cos.append(emb.cos()); all_sin.append(emb.sin())
cos = torch.cat(all_cos, dim=-1)
```

and application splits the head dimension into `ndim` equal parts, rotating each with its own axis's cos/sin:

```python
ndim = position_ids.shape[-1]                                   # 2
num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
x_parts   = torch.split(x,   [num_rotated_channels_per_dim] * ndim, dim=-1)
y_parts = [apply_rotary_pos_emb(x=x_parts[k], cos=cos_parts[k], sin=sin_parts[k], unsqueeze_dim=unsqueeze_dim)
           for k in range(ndim)]
return torch.cat(y_parts, dim=-1)
```

`apply_multidimensional_rope` is generic in `ndim` — pass 3D positions and it would rotate three-way. That generality is presumably why the audio tower and any future video-native variant can share the machinery.

So: **absolute position lives in the residual stream (added once, at the stem); relative position lives in the attention scores (applied at every layer).** They answer different questions — "where is this patch on the page" versus "how far apart are these two patches" — and Gemma 4 wants both.

### 3. The encoder block

`Gemma4VisionEncoderLayer` is a Gemma-style block: RMSNorm before *and* after each sublayer (`input_layernorm` / `post_attention_layernorm`, `pre_feedforward_layernorm` / `post_feedforward_layernorm`), a gated MLP, and no causal mask — `self.is_causal = False`, since an image has no reading order.

Two details are Gemma 4-specific.

**QKV norm, including a scale-free value norm:**

```python
self.q_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
self.k_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
```

Normalising queries and keys before attention is now common practice (it stops logits from drifting during long training runs). Normalising *values* is rarer, and doing it without a learnable scale — `with_scale=False`, so it is a pure normalisation with no parameters — is rarer still. You will meet the identical trio in the text decoder (chapter 06).

**Clippable linears.** Every projection in the vision and audio towers is a `Gemma4ClippableLinear`:

```python
if self.use_clipped_linears:
    hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
hidden_states = self.linear(hidden_states)
if self.use_clipped_linears:
    hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
```

The bounds are **buffers loaded from the checkpoint**, initialised to ±inf, so a model that was not trained with clipping is unaffected. E2B ships `use_clipped_linears: true` for its vision tower; the 31B ships `false`. This is activation clipping baked into the released weights — a quantisation-and-stability measure that only exists because someone measured the ranges during training.

### 4. Pooler: 3×3 by coordinate, not by index

The naive way to pool a patch grid is to reshape and average. That fails here, because the grid is a different shape for every image and the tensor is padded. So the pooler builds the pooling operation as a **matrix multiply driven by the coordinates**:

```python
k = int((input_seq_len // length) ** 0.5)                       # 3
clamped_positions = pixel_position_ids.clamp(min=0)
max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
output = weights.transpose(1, 2) @ hidden_states.float()
```

Line by line: integer-divide each `(x, y)` by 3 to get the coordinates of its 3×3 cell; flatten those cell coordinates to a single index using the image's own width; one-hot that index into a `(patches → soft tokens)` assignment matrix, scaled by 1/9; and matmul. Padding patches were zeroed before the call (`masked_fill`), so they contribute nothing to any average.

It is an unusual way to write average pooling, and it is the right one: it works for any grid shape, any padding pattern, in a single batched op with no Python loop over images. The returned `mask` (`torch.logical_not((weights == 0).all(dim=1))`) marks which output slots got at least one real patch — that is the `pooler_mask` used to strip padding from the final output.

Then a scaling step with a comment that tells you exactly what went wrong once:

```python
# The scaling expands the activation magnitude, which can exceed the float16 range, so it is
# computed in float32 and the pooled features are returned in float32.
hidden_states = hidden_states.float() * self.root_hidden_size
```

Multiplying by `√hidden_size` (≈27.7 at 768) can overflow fp16's 65504 ceiling. The pooler therefore returns **float32** regardless of the model's dtype, and the caller casts back after standardising. If you ever build your own tower, this is the class of bug that is nearly impossible to find from a loss curve.

`standardize` (large sizes only) then applies checkpoint-loaded `std_bias` and `std_scale` — a learned whitening of the soft tokens before they leave the tower.

### 5. Into text space

```python
class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(self, multimodal_config, text_config):
        self.multimodal_hidden_size = getattr(multimodal_config, "output_proj_dims", multimodal_config.hidden_size)
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, text_config.hidden_size, bias=False)
        self.embedding_pre_projection_norm = Gemma4RMSNorm(self.multimodal_hidden_size, eps=self.eps, with_scale=False)

    def forward(self, inputs_embeds):
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))
```

**That is the whole connector: an RMSNorm and one bias-free linear.** After all the machinery above — budget solving, dual position encoding, coordinate-driven pooling — the actual bridge into the language model is the most boring module in the file. This is LLaVA's lesson, still holding in 2026: the projector does not need to be clever; everything else does.

The class is shared with audio, which is why it reads `output_proj_dims` when present (the audio tower's own output width, 1536) and falls back to `hidden_size` otherwise. Two instances exist per model, `embed_vision` and `embed_audio` (chapter 01 §4), which is exactly why they are separately freezable in chapter 10.

`Gemma4Model.get_image_features` ties it together:

```python
vision_outputs = self.vision_tower(pixel_values=pixel_values, pixel_position_ids=image_position_ids, **kwargs)
vision_outputs.pooler_output = self.embed_vision(inputs_embeds=vision_outputs.last_hidden_state)
return vision_outputs
```

The result is a flat sequence of soft tokens, in the text embedding space, ready for chapter 08 to scatter into the prompt.

## Design space

| Model | Image → LLM tokens | Spatial position | Connector |
|---|---|---|---|
| **Flamingo** (2022) | Perceiver resampler → 64 latents | 1D over latents | Gated cross-attention layers inside the LLM |
| **BLIP-2** (2023) | Q-Former → 32 query tokens | learned queries | Q-Former + linear |
| **LLaVA** (2023) | 576 fixed patches | 1D learned | **A single MLP** |
| **Qwen2/3-VL** | Native resolution, 2×2 merge | M-RoPE (t/h/w) | MLP merger |
| **InternVL 3.5** | Tiles + thumbnail | 1D per tile | MLP + pixel shuffle |
| **Gemma 4** | Budgeted patches, 3×3 pooled | Learned 2D table **+** 2D RoPE | RMSNorm + linear |

Two axes are worth separating.

**Compression**: everyone compresses, they just disagree about where. Q-Former compresses with a *learned* module and a fixed output size — expressive, but a bottleneck that has to be trained and a fixed budget regardless of content. Pixel shuffle and 2×2 merge compress by *rearranging* channels, which is free and lossless-ish but only by integer factors. Gemma 4's 3×3 average pooling is the crudest option available and it works, which is a data point about how much of the heavy lifting the encoder has already done.

**Position**: LLaVA flattens a grid to a line and hopes the LLM figures it out. Qwen's M-RoPE and Gemma 4's 2D RoPE both encode the grid honestly, and they arrive at nearly the same place from different directions — M-RoPE partitions head dimensions across (time, height, width), Gemma 4 across (x, y) with a shared frequency spectrum per axis. Gemma 4 additionally keeps an absolute learned table, which Qwen drops. The extra parameters are cheap (10240 × hidden × 2) and buy resolution-independent absolute positioning.

Where Gemma 4 is genuinely unusual is *not* using a released SigLIP checkpoint. Most open VLMs bolt on SigLIP-so400m and adapt around its 224/384 fixed grid. Gemma 4 trained its own tower with the budget scheme built in from the start, which is why its preprocessing looks nothing like everyone else's.

## Check yourself

1. The vision tower gets both a learned absolute position table *and* 2D RoPE. Delete one — what specifically degrades in each case?
2. Why does each axis get its own full frequency spectrum instead of splitting a single `inv_freq` in half?
3. `clamp(min=0)` appears in both `_position_embeddings` and `_avg_pool_by_positions`. What would happen without it, and why is the clamped value irrelevant?
4. The pooler returns float32 even when the model is bfloat16. Why, and what is the magic number that forces it?
5. An image produced 266 real soft tokens out of a 280 budget. What is in the other 14 slots after `hidden_states[pooler_mask]`?
6. The connector is one RMSNorm plus one linear. Given everything upstream, argue why that is sufficient — then argue why BLIP-2 thought it was not.

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_vision_tower_anatomy.ipynb` | Hook every stage and print the shape as one image walks through; reimplement `apply_multidimensional_rope` and `assert_close`; visualise which patches pool into which soft token | 🟢 CPU (mini config) / 🟡 for real weights |
| `02_image_understanding.ipynb` | What the tower buys you in practice with real E2B weights: VQA, OCR, structured document extraction, and grounding with drawn boxes | 🟡 24GB VRAM |
| [`03_compare_qwen3vl.ipynb`](notebooks/03_compare_qwen3vl.ipynb) | The design-space control: Qwen3-VL's native-resolution + 2×2-merge approach on the same tasks, with visual-token counts side by side | 🟡 12GB+ VRAM |
