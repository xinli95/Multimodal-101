# 08 · Fusion and Masks — Where the Modalities Actually Meet

**Position in the pipeline**: between the towers and the decoder — the single point where four modalities become one sequence.

**Mental-model checkpoint:** fusion is not another encoder or connector. By this point every modality has already been understood, compressed, and projected to `D_text`; fusion only places those vectors into one sequence and specifies who may attend to whom.

This is the chapter the whole first half has been building toward, and it is smaller than you expect. The soft tokens from chapters 04 and 05 are already in the text embedding space. The placeholder runs from chapter 02 are already sitting in `input_ids`. Fusion is: find the placeholders, scatter the soft tokens over them, and then fix the attention mask so the model treats the result correctly.

The mask is the interesting part. A sentence is causal — token *n* may not see token *n+1*. But an image is not a sequence; there is no reason the top-left patch should be forbidden from seeing the bottom-right one. So Gemma 4 supports `use_bidirectional_attention="vision"`: **vision tokens attend bidirectionally within their block, while text stays causal.** The resulting 4D mask has a very particular shape, and looking at it as a heatmap is worth more than any amount of description.

Note that this is **not on in every size**. The 31B and 26B-A4B checkpoints set `"vision"`; **E2B and E4B leave it unset**, so their image tokens are as causal as their text. You therefore get two different masks out of the same code path depending on which checkpoint you loaded — which makes this the easiest chapter in Part I to run as a controlled experiment.

## What you will learn

1. How `get_placeholder_mask` finds image/video/audio positions from either `input_ids` or `inputs_embeds`, and why the `inputs_embeds` path has to compare against an embedded token
2. Why placeholder IDs are rewritten to `pad_token_id` before the embedding lookup — and what index error that avoids
3. How `masked_scatter` writes soft tokens into `inputs_embeds`, and how shape mismatches show up when the placeholder count and the soft-token count disagree
4. What `mm_token_type_ids` carries and why the model wants it in addition to the token IDs
5. How the bidirectional-vision mask is built (`create_masks_for_vision_model`, `get_block_sequence_ids_for_mask`), what a "block" is, and what happens with two images in one prompt
6. Plot the final 4D mask and read the architecture off the picture

## Source map

| Symbol in `modeling_gemma4.py` | Role |
|---|---|
| [`Gemma4Model.get_placeholder_mask`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2231) | Locating image / video / audio positions |
| [`Gemma4Model.forward`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2279) | The `masked_scatter` fusion and the `pad_token_id` rewrite |
| [`create_masks_for_vision_model`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2112) | The bidirectional-vision mask |
| [`get_block_sequence_ids_for_mask`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2167) | Grouping contiguous multimodal runs into blocks |
| [`sliding_window_mask_function`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1915) | Composed with the above for sliding layers (chapter 06) |
| [`Gemma4ModelOutputWithPast`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L78), [`Gemma4CausalLMOutputWithPast`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L109) | What comes back out |

Config fields: [`use_bidirectional_attention`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L179) (`"vision"` / `"all"` / unset), [`image_token_id`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L335), [`audio_token_id`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L339), [`video_token_id`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L336).

## Walkthrough

### 1. Finding the placeholders

```python
def get_placeholder_mask(self, input_ids=None, inputs_embeds=None):
    if input_ids is not None:
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        special_audio_mask = input_ids == self.config.audio_token_id
    else:
        special_image_mask = (inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device))).all(-1)
        ...
    return special_image_mask, special_video_mask, special_audio_mask
```

The `input_ids` path is an integer comparison. The `inputs_embeds` path — used when a caller supplies embeddings directly — has to *embed the placeholder ID and compare vectors*, which is the same awkwardness PLE runs into in chapter 07 and for the same reason: once you accept embeddings as an entry point, any information that lived in the IDs has to be recovered.

### 2. The `pad_token_id` swap

```python
image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
multimodal_mask = image_mask | video_mask | audio_mask

llm_input_ids = None
if inputs_embeds is None:
    llm_input_ids = input_ids.clone()
    # Replace image id with PAD if the image token if OOV, to avoid index-errors
    llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, llm_input_ids)
    inputs_embeds = self.get_input_embeddings()(llm_input_ids)
```

Placeholder IDs (258880–258884) can exceed the embedding table's row count, so they are rewritten to `pad_token_id` (0) before lookup. The resulting embeddings at those positions are meaningless — they are about to be overwritten — but they must be *valid*, because an out-of-range index is a crash, not a warning.

PLE gets the same treatment, with the pad embedding substituted explicitly:

```python
pad_embedding = self.language_model.embed_tokens.weight[self.config.text_config.pad_token_id, :]
llm_inputs_embeds = torch.where(multimodal_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds)
per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)
```

### 3. Fusion is one line, guarded

The same three-step pattern runs for images, video and audio:

```python
image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

n_image_tokens = image_mask.sum()
image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
torch_compilable_check(
    inputs_embeds[image_mask].numel() == image_features.numel(),
    f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}")

inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
```

**That `masked_scatter` is the fusion.** Everything in chapters 03–05 exists to produce `image_features`; everything in chapter 02 exists to produce `image_mask`; this line puts them together. After it, there is no such thing as an "image" anymore — just a sequence of embeddings, some of which came from pixels.

The `torch_compilable_check` is the error you will meet if you build inputs by hand. It fires when the placeholder count and the soft-token count disagree, which happens whenever the processor's prediction and the tower's actual output diverge — exactly the failure chapter 02's `_compute_audio_num_tokens` docstring was guarding against. `torch_compilable_check` rather than a plain `assert` because a Python `assert` on a tensor value graph-breaks under `torch.compile`.

Audio has one extra step, because the audio tower emits a padded batch:

```python
audio_features = audio_features[audio_mask_from_encoder.to(audio_features.device)]
```

The encoder's own validity mask strips padding before the scatter — the same job `pooler_mask` does for vision in chapter 04. Three modalities, three padding conventions, one flat sequence at the end.

### 4. Blocks: grouping contiguous multimodal runs

```python
def get_block_sequence_ids_for_mask(mm_token_type_ids, device):
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    return torch.where(is_vision, vision_group_ids, -1)
```

`mm_token_type_ids` — produced by the processor, which is why chapter 02 noted `model_input_names` appends it — labels each position by modality. This function turns those labels into **block IDs**: every maximal contiguous run of vision tokens gets a distinct integer, and text gets `-1`.

Two images in one prompt therefore get block IDs 0 and 1, and `-1` for everything else. Tokens in block 0 may see each other; tokens in block 1 may see each other; a token in block 0 may not see block 1 bidirectionally. Without this, a prompt with two images would let the first image peek forward at the second.

### 5. The mask, and the thing everybody gets wrong

Here is the finding that a paper figure would never show you. The docstring says it outright:

```python
def create_masks_for_vision_model(...):
    """Create full_attention and sliding_attention masks with correct composition.

    For global (full attention) layers:  causal only (no bidirectional)
    For local (sliding window) layers:  AND(sliding_window, OR(causal, blockwise))

    Unlike Gemma 3 (which applies bidirectional attention on all layers), Gemma 4
    explicitly disables bidirectional attention on global attention layers.
    """
    full_mask = create_causal_mask(**mask_kwargs)

    sliding_mask = create_causal_mask(
        **mask_kwargs,
        or_mask_function=blockwise_overlay(padded_block_sequence_ids),
        and_mask_function=sliding_window_overlay(config.sliding_window),
    )
    return {"full_attention": full_mask, "sliding_attention": sliding_mask}
```

**Bidirectional vision attention happens only on the sliding-window layers.** The global layers — the 1-in-6 that see the whole sequence — remain strictly causal, for image tokens as much as text.

Once you see the reason it is obvious. A global layer's KV cache is the expensive one and the one `generate` reuses across decoding steps; a strictly causal mask is what makes that cache valid, because entry *n* never depends on anything after *n*. Allow bidirectionality there and every image token's cached key would depend on later tokens, and incremental decoding would be wrong. The sliding layers can afford it: their window is small, and the blockwise overlay is confined inside image spans that are fully present at prefill.

So Gemma 4 gets most of the benefit of bidirectional image attention — patches attending to each other freely at 5 of every 6 layers — with none of the cache invalidity. Gemma 3 applied it everywhere; this is a deliberate, documented correction.

Read the composition carefully, because the order is load-bearing:

```
sliding = AND( sliding_window , OR( causal , blockwise ) )
```

`OR(causal, blockwise)` says *"you may attend backwards, **or** to anything in your own image block"*. `AND(sliding_window, …)` then clips the result to the window. A large image block is therefore **not** fully bidirectional — a 280-token image inside a 512-token window is, but two 1120-token images would be clipped. Composition order also explains the `maybe_pad_block_sequence_ids` dance: `or_mask_function` bypasses the internal padding that `create_causal_mask` normally applies, so the block IDs have to be padded to `kv_length` by hand first.

And the branch that decides which path runs:

```python
use_bidir = text_config.use_bidirectional_attention == "vision"
if use_bidir and mm_token_type_ids is not None:
    block_sequence_ids = get_block_sequence_ids_for_mask(mm_token_type_ids, device=inputs_embeds.device)
    causal_mask_mapping = create_masks_for_vision_model(block_sequence_ids=block_sequence_ids, **mask_kwargs)
else:
    # Smaller Gemma models (use_bidirectional_attention=None) or text-only inputs use standard causal masking
    causal_mask_mapping = create_masks_for_generate(**mask_kwargs)
```

Two conditions. On **E2B and E4B** the first is false (chapter 01 §5), so their image tokens are fully causal — the top-left patch cannot see the bottom-right one, ever. And on any size, a text-only prompt takes the plain path. **Running the same code with two different checkpoints produces two structurally different masks**, which is what makes this chapter's notebook a controlled experiment rather than a demonstration.

Note also that the result is a **dict keyed by layer type**, not a tensor. The decoder looks up `attention_mask[self.layer_type]` per layer. That is the mechanism by which one model runs two mask regimes at once.

## Design space

| Model | Image tokens attend | Where |
|---|---|---|
| **LLaVA, most open VLMs** | causally | everywhere |
| **Gemma 3** | bidirectionally | all layers |
| **Gemma 4 (31B, 26B-A4B)** | bidirectionally | sliding layers only |
| **Gemma 4 (E2B, E4B)** | causally | everywhere |
| **PaliGemma** | bidirectionally (prefix-LM) | all layers, image is a prefix |
| **Flamingo / Llama 3.2 V** | n/a — features live outside the sequence | cross-attention layers |

The underlying question is whether the causal constraint, which exists because text has a reading order, should apply to data that does not. The purist answer is no; the practical answer is that bidirectionality fights KV caching. Prefix-LM designs (PaliGemma) resolve it by making the image a fixed prefix that is always fully present. Gemma 4 resolves it by *layer type* — a finer-grained answer that keeps cacheable layers causal and spends bidirectionality where it is cheap.

That E2B declines the feature entirely is its own data point: at small scale, the simplicity is apparently worth more than the modelling gain.

## Check yourself

1. Why are placeholder IDs rewritten to `pad_token_id` before the embedding lookup, given the embeddings are immediately overwritten?
2. You build inputs by hand and get "Image features and image tokens do not match, tokens: 280, features: 266". What did you do wrong?
3. A prompt with two images. What are the block IDs, and what would go wrong without them?
4. Why is bidirectional attention disabled on global layers specifically? Give the caching argument.
5. Two 1120-token images at `sliding_window=1024`. Is either fully bidirectional within its block?
6. The same code, run on E2B and on 31B with an identical prompt, produces different masks. Which line decides, and what are the two outcomes?
7. `causal_mask_mapping` is a dict, not a tensor. Who consumes it, and why does it need to be a dict?

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_mask_heatmap.ipynb` | Build a two-image, one-audio prompt; plot the final 4D attention mask as a heatmap and read off the causal text region, the bidirectional image blocks, and the sliding window; then diff `inputs_embeds` before and after `masked_scatter` to see fusion happen | 🟢 CPU (mini config) |
