# 08 · Fusion and Masks — Where the Modalities Actually Meet

**Position in the pipeline**: between the towers and the decoder — the single point where four modalities become one sequence.

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
| `Gemma4Model.get_placeholder_mask` | Locating image / video / audio positions |
| `Gemma4Model.forward` | The `masked_scatter` fusion and the `pad_token_id` rewrite |
| `create_masks_for_vision_model` | The bidirectional-vision mask |
| `get_block_sequence_ids_for_mask` | Grouping contiguous multimodal runs into blocks |
| `sliding_window_mask_function` | Composed with the above for sliding layers (chapter 06) |
| `Gemma4ModelOutputWithPast`, `Gemma4CausalLMOutputWithPast` | What comes back out |

Config fields: `use_bidirectional_attention` (`"vision"` / `"all"` / unset), `image_token_id`, `audio_token_id`, `video_token_id`.

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_mask_heatmap.ipynb` | Build a two-image, one-audio prompt; plot the final 4D attention mask as a heatmap and read off the causal text region, the bidirectional image blocks, and the sliding window; then diff `inputs_embeds` before and after `masked_scatter` to see fusion happen | 🟢 CPU (mini config) |
