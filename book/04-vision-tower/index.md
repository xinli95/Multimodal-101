# 04 · Vision Tower — From Patches to Soft Tokens

Chapter 03 stopped after turning an image into patch vectors and coordinates. Those values still describe only small pieces of colour. This chapter asks the next question: **how does Gemma 4 turn those local patch vectors into contextual visual features that fit the language model's embedding width?**

```text
chapter 03: image processor
pixel_values + image_position_ids
              │
              ▼
┌──────────────────────────────────────────────┐
│ Gemma4VisionModel                           │
│                                              │
│ patch embedder → vision encoder → 3×3 pooler │
└──────────────────────────────────────────────┘
              │ contextual visual features
              ▼
Gemma4MultimodalEmbedder
RMSNorm → linear projection to text width
              │
              ▼
visual embeddings ready for the language model
              │
              ▼
chapter 08: place them into reserved prompt positions
```

The **vision tower** is `Gemma4VisionModel`: patch embedder, transformer encoder, and pooler. The **connector** is the following `Gemma4MultimodalEmbedder`: an RMSNorm and a learned linear projector. Keeping that boundary clear makes the architecture much easier to follow.

## What you will learn

1. What one row of `pixel_values` means before it enters the model
2. How patch embedding and the vision encoder turn local pixels into contextual features
3. Why image patches need position information, explained as absolute address versus relative geometry
4. Why Gemma 4 pools 3×3 patch groups only after the encoder
5. How `Gemma4MultimodalEmbedder` projects vision features into the text embedding space
6. How this connector differs from LLaVA's projector, BLIP-2's Q-Former, and Flamingo's cross-attention

## One image through the whole path

Use one running example before reading any implementation. At the default 280-token tier, chapter 03 turns a 640×480 image into a 912×672 image. The resized image contains a 57×42 grid, or 2,394 real patches. The processor pads that to 2,520 rows so all images at this tier share one input shape.

For the E2B model, the shapes are:

| Stage | Shape for this one image | What changed? |
|---|---:|---|
| image processor output | `[1, 2520, 768]` | 2,394 real 16×16 RGB patches plus 126 zero-padding rows |
| patch embedder | `[1, 2520, 768]` | each raw patch becomes a learned vision feature; width happens to remain 768 on E2B |
| 16-layer vision encoder | `[1, 2520, 768]` | shape stays the same, but every real patch now contains context from other patches |
| 3×3 pooler | `[1, 280, 768]` | nine neighbouring patch features are averaged into one candidate soft token |
| remove pooled padding | `[266, 768]` | only the real 19×14 pooled grid remains |
| multimodal embedder | `[266, 1536]` | token count stays 266; feature width changes from vision width to text width |

This table is the chapter in miniature. The encoder changes the **meaning** of each row, the pooler changes the **number** of rows, and the connector changes the **width** of each row.

### What are the 2,520 rows?

`2,520 = 280 × 3²` is the default tier's maximum number of pre-pooling patches. It is not the number of real patches in every image. In this example:

- rows 0–2,393 contain real patch pixels;
- rows 2,394–2,519 are zero padding;
- the matching `image_position_ids` contain real `(x, y)` coordinates for the first group and `(-1, -1)` for the padding group.

The tower converts the `(-1, -1)` test into a Boolean padding mask. The encoder uses the mask so real patches do not attend to padding, and the pooler uses it so zero rows do not become visual tokens. The position tensor therefore supplies two plain pieces of information: where a real patch came from, and whether a row is padding.

## The architecture, one stage at a time

### 1. Patch embedder: pixels become model features

Each processor row is a flattened 16×16 RGB square:

```text
16 × 16 × 3 = 768 pixel values
```

The patch embedder performs three operations:

```python
pixel_values = 2 * (pixel_values - 0.5)       # [0, 1] → [-1, 1]
hidden_states = self.input_proj(pixel_values) # raw patch → vision hidden width
hidden_states = hidden_states + position_embeddings
```

#### Is `[0, 1] → [-1, 1]` normalization?

In ordinary mathematical language, yes: it is a fixed affine rescaling and centring step. The potential confusion comes from the Transformers API. Chapter 03 said `do_normalize=False` because the **image processor** does not apply the usual per-channel operation `(x - image_mean) / image_std`. It only divides bytes by 255. The model then applies the same `2x - 1` transform to every channel.

So both statements are true:

- there is no ImageNet-style, per-channel mean/std normalization in the processor;
- the model still recentres values from `[0, 1]` to `[-1, 1]` before the learned projection.

`input_proj` is a bias-free linear layer. It takes the 768 numbers in one raw patch and produces one vector in the vision model's hidden width: 768 on E2B and 1,152 on the larger models. It does this independently for every patch.

The old phrase “vision stem” simply meant **the entrance to the learned vision network**. A ResNet-style stem would first run several convolutions and gradually reduce spatial resolution. Gemma 4 does not do that: chapter 03 has already cut the image into patches, so the learned entrance is just one linear projection. “No convolutions, no ResNet, no patch-merging pyramid” describes that architectural choice; it does not imply that the tower has no feature extraction. The transformer encoder does the feature extraction next.

### 2. Position: address first, geometry during attention

A patch's 768 colour values do not reveal where it came from. The same blue patch might be sky at the top of an image or water at the bottom. After flattening a grid into a sequence, the model therefore needs explicit position information.

Gemma 4 supplies it in two ways. A useful first-reading intuition is:

| Position signal | Informal question it helps answer | Where it enters |
|---|---|---|
| learned 2D position embedding | “Where is this patch in the image?” | added once to the patch feature |
| 2D RoPE | “What is the horizontal and vertical relationship between two patches?” | applied to queries and keys in every attention layer |

The learned embedding has one lookup table for x and another for y. For a patch at `(x, y)`, Gemma 4 looks up `x_embedding[x]` and `y_embedding[y]`, adds them, and then adds the result to the patch feature. Padding coordinates are `(-1, -1)`; they are masked to a zero position embedding.

During self-attention, 2D RoPE uses those same x and y coordinates differently. Half of each attention head carries horizontal position and half carries vertical position. This makes attention sensitive to spatial displacement: two patches with the same horizontal separation receive the same x-relative rotation even if both move elsewhere in the image.

The “absolute address versus relative geometry” distinction is an intuition, not a claim that the two mechanisms have perfectly isolated responsibilities after training. What matters on a first read is simpler: the learned embedding gives every patch a location-aware starting representation, while 2D RoPE makes the attention calculation itself aware of the 2D grid. The exact frequency construction and `clamp-then-mask` implementation belong in [`01_vision_tower_anatomy.ipynb`](notebooks/01_vision_tower_anatomy.ipynb); they are not prerequisites for following the architecture.

### 3. Vision encoder: patches gain context

After patch embedding, each patch knows its own pixels and position, but it still does not know what the rest of the image contains. The vision encoder supplies that context.

E2B uses 16 transformer blocks; the larger Gemma 4 variants use 27. Each block has the familiar two-part structure:

```text
patch features
    │
    ├─ self-attention ──► mix information across all real patches
    │
    └─ gated MLP ───────► transform each patch feature
         with normalization and residual connections around both
```

Vision attention is not causal. A patch near the top-left may attend to a patch near the bottom-right in the same layer. Repeating this process turns local evidence into contextual evidence. A patch that initially contains only brown pixels might become useful as “part of the dog's ear” after combining nearby texture, outline, face, and scene information.

The sequence shape does not change through the encoder. On E2B, `[1, 2520, 768]` enters and `[1, 2520, 768]` leaves. That does **not** mean nothing happened: every vector has been rewritten by 16 layers of attention and MLP computation. Padding rows remain present for a regular batch shape, but the attention mask prevents them from participating as image content.

### 4. Pooler: nine contextual patch features become one soft token

Sending all 2,394 real patch features in the running example to the language model would consume too much context. Gemma 4 reduces the sequence by pooling each spatial 3×3 group:

```text
57×42 contextual patch grid
          │ average each 3×3 neighbourhood
          ▼
19×14 visual-token grid = 266 real soft tokens
```

The order matters. Gemma 4 pools **after** the transformer, so it averages features that have already seen the rest of the image—not nine raw pixel patches. The encoder gets fine-grained evidence first; the language model receives the cheaper summary.

Why does the pooler use coordinates? Because a padded batch is a one-dimensional tensor, while “these nine patches are neighbours” is a two-dimensional fact. Integer-dividing `(x, y)` by 3 assigns every patch to its 3×3 group. The `(-1, -1)` padding marker prevents padded rows from being treated as real groups. After pooling, `pooler_mask` removes the 14 unused outputs from the 280-output ceiling, leaving the 266 real soft tokens shown above.

This is ordinary average pooling expressed in a way that works for different grid shapes within one batch. The anatomy notebook reconstructs the exact one-hot matrix multiplication for readers who want the implementation.

### 5. Multimodal embedder: project vision width to text width

The pooler has produced the right **number** of visual tokens, but their vectors still use the vision tower's coordinate system and hidden width. The language model expects vectors with its own embedding width. `Gemma4MultimodalEmbedder` is the learned adapter between them:

```python
class Gemma4MultimodalEmbedder(nn.Module):
    def forward(self, inputs_embeds):
        normalized = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normalized)
```

It contains:

1. a scale-free RMSNorm, which puts incoming feature magnitudes on a stable scale;
2. one bias-free linear layer, which learns the mapping from vision width to text width.

For E2B, the running example changes from `[266, 768]` to `[266, 1536]`. The connector does **not** reduce 266 to a smaller count. It changes the representation of each token so the language model can accept it alongside 1,536-wide word embeddings. In chapter 08, those 266 vectors replace the 266 image-placeholder positions reserved by chapter 02.

Yes: in the simplest architectural description, Gemma 4 is a vision encoder followed by a projector into text space. The code calls that projector `Gemma4MultimodalEmbedder`; “connector” is the broader literature term.

## Now read `Gemma4VisionModel.forward`

With the stages established, the short forward pass becomes a summary rather than a puzzle:

```python
padding_positions = (pixel_position_ids == -1).all(dim=-1)
output_length = pixel_values.shape[-2] // self.config.pooling_kernel_size**2

# 1. raw patch pixels → positioned vision features
hidden_states = self.patch_embedder(
    pixel_values, pixel_position_ids, padding_positions
)

# 2. contextualise every real patch
hidden_states = self.encoder(
    inputs_embeds=hidden_states,
    attention_mask=~padding_positions,
    pixel_position_ids=pixel_position_ids,
).last_hidden_state

# 3. spatial 3×3 reduction, then discard pooled padding
hidden_states, pooler_mask = self.pooler(
    hidden_states, pixel_position_ids, padding_positions, output_length
)
hidden_states = hidden_states[pooler_mask]
```

`Gemma4Model.get_image_features` then performs stage 4:

```python
vision_outputs = self.vision_tower(
    pixel_values=pixel_values,
    pixel_position_ids=image_position_ids,
)
vision_outputs.pooler_output = self.embed_vision(
    inputs_embeds=vision_outputs.last_hidden_state
)
```

The division between the two classes is now explicit: `vision_tower` understands and compresses the image; `embed_vision` maps the resulting vectors to the language model's width.

## Implementation details worth knowing later

These details matter when reproducing the source, but they are not the architectural backbone:

- **Large coordinate table.** The learned x and y tables each have 10,240 entries. This avoids resizing one fixed square position grid when aspect ratios change.
- **Q/K/V normalization.** Attention normalizes queries, keys, and values; the value norm has no learned scale. This is a numerical-stability choice, not a new architectural stage.
- **Clippable linears.** E2B can clamp activations around linear layers using checkpoint-loaded bounds. Larger variants disable this path.
- **Float32 pooling.** The pooler multiplies averaged features by `√hidden_size`. It performs this step in float32 to avoid fp16 overflow.
- **Optional output standardization.** The larger vision towers apply checkpoint-loaded bias and scale after pooling; E2B does not.

The first notebook tests these details with a miniature CPU model.

## Source map

| Symbol in `modeling_gemma4.py` | Role |
|---|---|
| [`Gemma4VisionPatchEmbedder`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L579) | `[0,1] → [-1,1]`, patch projection, learned x/y embeddings |
| [`Gemma4VisionAttention`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L917) | non-causal self-attention with 2D RoPE |
| [`Gemma4VisionEncoderLayer`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L986), [`Gemma4VisionEncoder`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1030) | 16- or 27-layer transformer stack |
| [`Gemma4VisionPooler`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L624) | coordinate-aware 3×3 average pooling |
| [`Gemma4VisionModel`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2017) | patch embedder + encoder + pooler |
| [`Gemma4MultimodalEmbedder`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2085) | RMSNorm + projection from vision width to text width |
| [`Gemma4Model.get_image_features`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2212) | vision tower followed by the connector |

## Design space: what does a connector actually connect?

Chapter 03 compared **image sampling strategies**: fixed canvas, tiling, dynamic resolution, and Gemma 4's selected pixel budget. Those choices determine how many patch features reach the vision tower. This chapter compares the next decision: **how are vision features compressed, mapped, and presented to the language model?** The two design spaces are adjacent, but they are not duplicates.

The word “connector” is used loosely in VLM papers. It may mean only a width-changing projector, a learned token compressor, or an entire cross-attention path added inside the LLM. Separate three questions when comparing designs:

1. Does it reduce the number of visual tokens?
2. How does it map vision feature width to language feature width?
3. Are visual vectors inserted into the text sequence, or kept separately for cross-attention?

| Family | What happens between vision encoder and LLM? | Token-count effect | How vision reaches the LLM | Main trade-off |
|---|---|---|---|---|
| [**LLaVA**](https://arxiv.org/abs/2304.08485) | Apply a linear layer in the original model, or a small MLP in later versions, independently to each patch feature | none in the basic connector; one output per retained patch | visual vectors join the input embedding sequence | cheapest and easiest to train, but the LLM pays for every visual patch |
| [**BLIP-2**](https://arxiv.org/abs/2301.12597) | 32 learned queries cross-attend to all image features in a Q-Former, then project the 32 results | dense patch sequence → fixed 32 | the query outputs condition the language model as a compact visual prefix | content-adaptive compression, but adds a substantial module and its own pretraining objectives |
| [**Flamingo**](https://arxiv.org/abs/2204.14198) | a Perceiver Resampler produces 64 visual latents per image/video; gated cross-attention layers are inserted throughout the LLM | variable vision grid → fixed 64 latents per visual input | text states repeatedly cross-attend to separate visual memory | strong for interleaved images/video and few-shot prompts, but modifies the LLM architecture and adds recurring cross-attention cost |
| [**Qwen2-VL**](https://arxiv.org/abs/2409.12191) | neighbouring 2×2 vision features are concatenated and mapped by a learned merger/MLP | four patch features → one visual token | merged vectors enter the language sequence | retains a variable-resolution grid with a learned local merge; cost still follows image area |
| [**InternVL 1.5**](https://arxiv.org/abs/2404.16821) | pixel shuffle moves a 2×2 spatial neighbourhood into channels, then an MLP maps it to language width | four features → one per tile neighbourhood; total still grows with tile count | projected tile and thumbnail vectors enter the language sequence | keeps high-resolution tile detail, but many tiles can still create a long visual prefix |
| **Gemma 4** | average contextual features over 3×3 spatial groups, then apply RMSNorm + one linear projector | nine patch features → one soft token, capped by the chosen tier | projected vectors replace reserved image positions | very cheap and predictable; uniform averaging is less adaptive than learned queries |

The key comparison is now concrete:

- **LLaVA's connector mainly aligns feature width.** Basic LLaVA does not ask the connector to decide which patches matter.
- **BLIP-2 and Flamingo learn a fixed-size summary.** Learned queries or latents choose what to carry forward, adding capacity and training complexity.
- **Qwen2-VL, InternVL, and Gemma 4 preserve a spatial grid while reducing it locally.** Qwen and InternVL use learned rearrangement/MLP paths; Gemma 4 uses the simpler 3×3 average after its encoder has contextualised the patches.
- **Flamingo fuses differently.** It keeps visual memory outside the text sequence and adds cross-attention inside the LLM; the other rows primarily turn vision outputs into input-like embeddings.

This section is self-contained. The notebooks are evidence and implementation practice, not missing prerequisites: notebook 01 opens Gemma 4's exact RoPE and pooling code; notebook 02 tests what the tower buys on VQA/OCR/grounding; notebook 03 compares Gemma 4's fixed budget with Qwen3-VL's dynamic-resolution token counts on similar tasks.

## Review questions, with answers

1. **What is one row of `pixel_values`?** It is one flattened 16×16 RGB patch: `16 × 16 × 3 = 768` rescaled pixel values. Its `(x, y)` coordinate lives separately in `image_position_ids`.

2. **Where do the 2,520 rows at the default tier come from?** The maximum is 280 soft tokens times nine pre-pooling patches per token. A particular image may use fewer real rows; the rest are zero padding marked by position `(-1, -1)`.

3. **Why are both coordinates and a padding mask needed?** Coordinates preserve the 2D grid for position embedding and spatial pooling. A mask prevents padded rows from taking part in attention or surviving pooling. Gemma 4 derives that mask from coordinates equal to `(-1, -1)`.

4. **Why use a learned position embedding and 2D RoPE?** The learned x/y embeddings make the initial patch features location-aware. RoPE makes each attention layer sensitive to horizontal and vertical displacement between patches. They inject position at different points in the computation.

5. **Why pool after the encoder instead of before it?** Pooling after the encoder lets fine-grained patches exchange information first. The 3×3 average then compresses contextual features. Pooling raw patches earlier would discard local evidence before the transformer could use it.

6. **What does `Gemma4MultimodalEmbedder` change?** It changes feature width, not token count. On E2B, `[266, 768]` becomes `[266, 1536]` through RMSNorm and a learned linear projection.

7. **Is `[0,1] → [-1,1]` normalization?** Broadly yes—it is fixed centring and rescaling. It is not the per-channel mean/std normalization controlled by the image processor's `do_normalize` flag.

8. **What happens to the 14 unused outputs when an image produces 266 real tokens under a 280 ceiling?** `pooler_mask` removes them. They are not passed to the connector and do not occupy language-model context positions.

## Notebooks

| Notebook | What it adds beyond the chapter | Hardware |
|---|---|---|
| [`01_vision_tower_anatomy.ipynb`](notebooks/01_vision_tower_anatomy.ipynb) | Implementation deep dive: hook every stage, reproduce 2D RoPE, visualise 3×3 assignment, and inspect float32 pooling | 🟢 CPU (mini config) / 🟡 real weights |
| [`02_image_understanding.ipynb`](notebooks/02_image_understanding.ipynb) | Behavioural experiments with E2B weights: VQA, OCR, structured extraction, grounding, and a soft-token sweep | 🟡 24GB VRAM |
| [`03_compare_qwen3vl.ipynb`](notebooks/03_compare_qwen3vl.ipynb) | Design-space experiment: compare Qwen3-VL's dynamic resolution and visual-token counts on similar tasks | 🟡 12GB+ VRAM |
