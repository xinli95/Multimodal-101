# 02 · Text I/O — Tokenizer, Chat Template, Placeholders

**Position in the pipeline**: `messages ──► chat template ──► tokenizer ──► input_ids`

The multimodal part of a multimodal model starts, counterintuitively, in the *text* pipeline. Before any pixel is touched, the chat template has already decided where the image goes: it expands a single `{"type": "image"}` entry into a run of 280 identical placeholder token IDs, wrapped in begin/end-of-image markers. Chapter 08 will overwrite those placeholder embeddings with real visual features. Everything in between is bookkeeping on a flat sequence of integers.

## What you will learn

1. How `Gemma4Processor` composes a tokenizer, an image processor, a video processor and an audio feature extractor behind a single `__call__`
2. How `apply_chat_template` turns a `messages` list into a string, and how the placeholder runs are computed — including the *dynamic* audio case, where token count depends on duration
3. The special-token map, and why the model has to temporarily rewrite placeholder IDs to `pad_token_id` before embedding lookup
4. Gemma 4's native `system` role, its configurable thinking mode, and its function-calling format
5. Why `padding_side="left"` is mandatory for batched generation

## The special tokens

| Token | ID | Purpose |
|---|---|---|
| `boi_token_id` | 255999 | begin-of-image |
| `boa_token_id` | 256000 | begin-of-audio |
| `image_token_id` | 258880 | image placeholder — repeated `image_seq_length` (default **280**) times |
| `audio_token_id` | 258881 | audio placeholder — repeated `ceil(duration_ms / 40)` times, capped at 750 |
| `eoi_token_id` | 258882 | end-of-image |
| `eoa_token_index` | 258883 | end-of-audio |
| `video_token_id` | 258884 | video placeholder |

The 40ms per audio token is not arbitrary: it falls out of the audio tower's 4× temporal downsampling applied to 10ms frames (chapter 05).

## Source map

| File | Symbol | Role |
|---|---|---|
| `processing_gemma4.py` | `Gemma4Processor.__call__` | The front door for all four modalities |
| | `prepare_inputs_layout`, `validate_inputs` | Ordering and consistency of interleaved inputs |
| | `replace_image_token`, `replace_audio_token`, `replace_video_token` | Placeholder-run expansion |
| | `_get_num_multimodal_tokens`, `_compute_audio_num_tokens` | How many placeholders each input is worth |
| `modeling_gemma4.py` | `Gemma4Model.get_placeholder_mask` | The consumer side: finding those placeholders again |
| | `Gemma4TextScaledWordEmbedding` | Embedding lookup scaled by `√hidden_size` |

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_tokens_and_template.ipynb` | Expand a multimodal `messages` list one step at a time — raw template output, token IDs, decoded pieces — and locate every placeholder run by eye. Then a tool-calling round trip and a thinking-mode comparison | 🟢 CPU, tokenizer only |
| [`02_tokenize_everything.ipynb`](notebooks/02_tokenize_everything.ipynb) | The wider view: how text, images, audio and video each become tokens across the field, and what the compression ratios look like | 🟢 CPU |
