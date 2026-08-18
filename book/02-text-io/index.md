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

## Walkthrough

### 1. `Gemma4Processor` is four processors in a trench coat

```python
class Gemma4Processor(ProcessorMixin):
    def __init__(self, feature_extractor, image_processor, tokenizer, video_processor,
                 chat_template=None, image_seq_length: int = 280,
                 audio_seq_length: int = 750, audio_ms_per_token: int = 40, **kwargs):
```

Four sub-processors, one object. `AutoProcessor.from_pretrained` builds all four from a single `processor_config.json`, and `processor(text=..., images=..., videos=..., audio=...)` routes each argument to its owner and then merges the results into one `BatchFeature`. This is the pattern for every multimodal model in the library; learn it once here.

Two of the constructor's defaults are the numbers this whole chapter turns on:

- `image_seq_length = 280` — how many placeholder tokens one image is worth
- `audio_ms_per_token = 40` — how many milliseconds one audio token is worth

The comment in the source explains where 40 comes from, and it is a load-bearing fact for chapter 05: *"the SSCP convolution's 4× time reduction on 10ms frames."*

### 2. The chat template emits **one** `<|image|>`, not 280

This is the single most common misunderstanding about multimodal chat templates, so it is worth being precise. Pull the template and read it:

```python
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
print(proc.chat_template)   # 386 lines of Jinja
```

In the message loop, a content item of type `image` produces exactly this:

```jinja
{%- elif item.get('type') in ['image', 'image_url'] -%}
    {{- '<|image|>' -}}
{%- elif item.get('type') in ['audio', 'input_audio'] -%}
    {{- '<|audio|>' -}}
{%- elif item.get('type') == 'video' -%}
    {{- '<|video|>' -}}
```

One marker per attachment. The expansion into a run of 280 happens later, in the **processor**, once the image processor has actually looked at the image and decided how many soft tokens it is worth:

```python
def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
    num_soft_tokens = image_inputs["num_soft_tokens_per_image"][image_idx]
    return f"{self.boi_token}{self.image_token * num_soft_tokens}{self.eoi_token}"
```

So the pipeline is **template → one marker → processor → `<boi>` + N×`<image>` + `<eoi>`**, and N is data-dependent. That ordering is why `apply_chat_template(..., tokenize=True)` needs the images in hand; you cannot tokenize a multimodal prompt without processing its attachments first.

Audio does the same thing, but computes N from the waveform by *simulating the encoder*:

```python
def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
    mask = audio_inputs["input_features_mask"][audio_idx]
    # Simulate two stride-2 conv blocks on the mask
    t = len(mask)
    for _ in range(2):
        t_out = (t + 2 - 3) // 2 + 1
        mask = mask[::2][:t_out]
        t = len(mask)
    return f"{self.boa_token}{self.audio_token * int(mask.sum())}{self.eoa_token}"
```

Two stride-2 convolutions, kernel 3, padding 1 — applied to the *mask*, not the features, purely to count. Why does the processor duplicate the encoder's arithmetic instead of asking it? Because the placeholders have to exist in `input_ids` **before** the model runs. The count must be predicted, and it must be exactly right. `_compute_audio_num_tokens` carries a docstring that says what happens when it is not:

> *Must match `audio_mask.sum()` from the audio tower or vLLM's `_merge_multimodal_embeddings` will raise on a length mismatch.*

That same docstring flags a genuine trap for anyone reading the config:

> *note: `config.conv_kernel_size=5` is the **conformer** depthwise conv, NOT this one*

The subsampling convolutions are kernel 3 / stride 2 / padding 1 and are **not exposed in `Gemma4AudioConfig` at all** — they are architectural constants hard-coded in both the encoder and this counting function. If you ever change the audio stack, you must change two places. It is the kind of coupling that only source reading reveals.

Video is the odd one out, and its handling is a small design lesson:

```python
timestamp_str = [f"{int(seconds // 60):02d}:{int(seconds % 60):02d}" for seconds in metadata.timestamps]
video_replacement = " ".join(
    [f"{t} {self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}" for t in timestamp_str]
)
```

Each sampled frame becomes `MM:SS` **as literal text**, followed by that frame's soft-token run. Temporal information is not encoded in an embedding or a position index — it is written into the prompt as a string and read by the language model like any other text. Compare that with Qwen-VL's M-RoPE, which encodes time as a position dimension (see [landscape](../landscape.md)). Note also that video runs are wrapped in `<boi>`/`<eoi>`, the *image* markers, with `<|video|>` inside.

### 3. The token map, and the `pad_token_id` shuffle

| Token | ID | Notes |
|---|---|---|
| `<|image|>` | 258880 | repeated `num_soft_tokens_per_image` times |
| `<|audio|>` | 258881 | repeated `ceil(duration_ms / 40)`, capped at 750 |
| `<|video|>` | 258884 | added at runtime by the processor — see below |
| `<boi>` / `<eoi>` | 255999 / 258882 | wrap image *and* video runs |
| `<boa>` / `<eoa>` | 256000 / 258883 | wrap audio runs |

A wonderfully honest line in the constructor:

```python
# FIXME: add the token to config and ask Ryan to re-upload
tokenizer.add_special_tokens({"additional_special_tokens": ["<|video|>"]})
```

The video token is not in the shipped tokenizer; the processor patches it in every time it is constructed. Worth knowing if you ever compare `len(tokenizer)` before and after building a processor.

The placeholder IDs sit **above** the model's usable embedding range in some configurations, which is why the model does this before the embedding lookup (chapter 08 in full):

```python
llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, llm_input_ids)
inputs_embeds = self.get_input_embeddings()(llm_input_ids)
```

Every placeholder is temporarily turned into a pad token, embedded (producing garbage that is about to be overwritten anyway), and then `masked_scatter` writes the real soft tokens over those positions. The alternative — indexing an embedding table with an out-of-range ID — is a crash.

### 4. `validate_inputs`: the error you will actually hit

```python
n_images_in_text = [sample.count(self.image_token) for sample in text]
n_images_in_images = [len(sublist) for sublist in images]
if n_images_in_text != n_images_in_images:
    raise ValueError("The total number of <|image|> tokens in the prompts should be the same as the number of images passed. ...")
```

Per-sample, not per-batch — a batch where sample 0 has two images and sample 1 has none must be passed as a nested list `[[img, img], []]`, not a flat list of two. `prepare_inputs_layout` calls `make_nested_list_of_images` to enforce that structure, and will also *invent* a text prompt if you pass images with no text at all:

```python
if images and not text:
    text = [" ".join([self.image_token] * len(image_list)) for image_list in images]
```

### 5. The turn grammar, thinking, and tools

Gemma 4's template is not ChatML. Turns are delimited by paired angle-brace tokens:

```
<bos><|turn>system
...system text and tool declarations...
<turn|>
<|turn>user
<|image|>What is shown in this image?<turn|>
<|turn>model
```

Three things are worth reading the Jinja for.

**The system turn is synthesised, not just copied.** It is emitted if *any* of three conditions hold — there is a system message, there are tools, or thinking is enabled:

```jinja
{%- if enable_thinking or tools or (messages and messages[0]['role'] in ['system', 'developer']) -%}
    {{- '<|turn>system\n' -}}
    {%- if enable_thinking -%}{{- '<|think|>\n' -}}{%- endif -%}
```

So `enable_thinking=True` is not a generation flag; it is a `<|think|>` token injected at the top of the first system turn. `developer` is accepted as an alias for `system`.

**Thinking is stripped from history.** Assistant content passes through a `strip_thinking` macro that removes everything between `<|channel>` and `<channel|>`, and reasoning is only re-rendered for messages *after* the last user message:

```jinja
{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls')) -%}
```

The model's reasoning from three turns ago is deliberately not fed back — it is scratch work, not context. `preserve_thinking=True` overrides this for tool-calling chains, where the reasoning that led to a call is worth keeping.

**Tools are declared in a compact DSL, not JSON.** A tool declaration renders as `<|tool>declaration:name{description:<|"|>...<|"|>,parameters:{...},type:<|"|>OBJECT<|"|>}<tool|>`, where `<|"|>` is a dedicated string-quote *token*. A call comes back as `<|tool_call>call:get_weather{location:<|"|>SF<|"|>}<tool_call|>`. Using single tokens for structural punctuation instead of literal `"` characters is only marginally cheaper than JSON — measured on the weather tool above it saves about 6% — but it is much harder for the model to break: `<|"|>` is one token, so there is no such thing as a half-open or unescaped quote, and a parser can split on token IDs rather than attempting JSON recovery on model output. The template also refuses to paper over a common client bug:

```jinja
{{- raise_exception("chat_template: tool_calls[].function.arguments must be a JSON object (mapping), not a string. Deserialize arguments before passing to the template.") -}}
```

### 6. `padding_side="left"`, and why it is not optional

Every example in the Gemma 4 docs constructs the processor the same way:

```python
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it", padding_side="left")
```

With right padding, a batch of prompts of different lengths puts pad tokens *after* the real content, so each sequence's last real token sits at a different index — and `generate` appends new tokens at the end of the tensor, i.e. after the padding. Left padding aligns every sequence's final token at the same position, which is what the decode loop assumes. Get this wrong and you do not get an exception; you get quietly degraded output, which is worse.

The matching habit on the way out:

```python
input_len = inputs["input_ids"].shape[-1]
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0][input_len:], skip_special_tokens=True))
```

`generate` returns prompt + completion. Slicing at `input_len` is how you get only what the model actually said.

## Design space

- **Placeholder-run expansion** (Gemma 4, LLaVA, Qwen-VL): reserve N text positions, overwrite their embeddings. The LLM's code path never changes — it only ever sees a sequence of embeddings. Costs N real context slots per image, which is why the size of N (chapter 03) is such a consequential knob.
- **Cross-attention injection** (Flamingo, Llama 3.2 Vision): visual features live outside the sequence and are attended to through dedicated layers. Context is not consumed, but the LLM needs new parameters and a modified forward pass.
- **Fixed query compression** (BLIP-2): a Q-Former squeezes any image into exactly 32 tokens. Cheap and constant, but the bottleneck is unforgiving for dense text.

Gemma 4 takes the first route and then spends its effort on making N controllable — the menu in chapter 03.

On timestamps, the split is just as clean: Gemma 4 writes `MM:SS` into the prompt as text; Qwen-VL encodes time as a RoPE dimension. Text costs tokens and is trivially interpretable; a position dimension is free but requires the whole stack to agree on it.

## Check yourself

1. Why can't you call `apply_chat_template(messages, tokenize=True)` on a message list containing an image without also passing the image itself?
2. A 3.4-second clip at 16kHz. Roughly how many `<|audio|>` placeholders, and what caps it?
3. Where does `<|video|>` come from, and why is it not in `tokenizer.json`?
4. You pass a batch of two prompts, the first with two images and the second with none, and get a `ValueError` about counts. What shape does `images` need to be?
5. `enable_thinking=True` changes the rendered prompt. Where exactly does the change appear, and what does it look like?
6. Your batched generations are subtly worse than your single generations. What is the first thing to check?

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_tokens_and_template.ipynb` | Expand a multimodal `messages` list one step at a time — raw template output, token IDs, decoded pieces — and locate every placeholder run by eye. Then a tool-calling round trip and a thinking-mode comparison | 🟢 CPU, tokenizer only |
| [`02_tokenize_everything.ipynb`](notebooks/02_tokenize_everything.ipynb) | The wider view: how text, images, audio and video each become tokens across the field, and what the compression ratios look like | 🟢 CPU |
