# 02 · Text I/O — From Messages to Model Input

This chapter can be read on its own. Its question is simple: **how does a Python chat conversation become the integer sequence that Gemma 4 receives?** With an image attached, two paths run side by side:

```text
messages ── chat template ──► prompt text containing one <|image|>
image    ── image processor ─► pixel_values + a count N
                                      │
prompt text + N ── expand marker ──► final prompt ── tokenizer ──► input_ids
```

This chapter follows the top path and explains the join. [Chapter 03](../03-image-processor/index.md) explains how the image processor obtains `pixel_values` and `N`; [chapter 08](../08-fusion-and-masks/index.md) explains how the model later replaces the `N` reserved positions with image features. You do not need either implementation yet.

## What you will learn

1. The separate jobs of a chat template, tokenizer, and multimodal processor
2. How to read Gemma 4's template from a one-turn conversation outward
3. Why one image marker becomes a run of `N` placeholder IDs
4. Why images are nested by sample when a batch is assembled
5. When left padding is appropriate, and what it costs

## Three objects, three jobs

The names are easy to blur together, so establish these boundaries first:

| Object | Input | Output | Job |
|---|---|---|---|
| chat template | structured `messages` | one prompt string | serialise roles and content in the syntax Gemma 4 was trained on |
| tokenizer | a string | integer `input_ids` | split text into vocabulary pieces and look up their IDs |
| `Gemma4Processor` | text plus optional images, audio, and video | one model-ready batch | coordinate the tokenizer and the three modality-specific preprocessors |

A chat template is not a model and it is not ChatML knowledge you are expected to have. It is a **serializer**. Different chat models were trained on different separators, so the same conversation has to be written in each model's dialect before tokenization.

## Start with one text turn

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
messages = [
    {"role": "user", "content": "Explain tokenization in one sentence."}
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(prompt)
```

The important structure of the result is:

```text
<bos><|turn>user
Explain tokenization in one sentence.<turn|>
<|turn>model
```

Read it literally:

- `<bos>` starts the sequence.
- `<|turn>user\n` opens a user turn; `<turn|>` closes it.
- `<|turn>model\n` is an empty model-turn header. `add_generation_prompt=True` adds this header so generation knows which role should speak next.

Only after that does the tokenizer convert the complete string—including the turn delimiters—into IDs:

```python
encoded = processor.tokenizer(prompt, return_tensors="pt")
print(encoded["input_ids"].shape)
```

That is the basic pipeline. Jinja is merely the implementation language used to loop over `messages` and emit this string.

## Read the template from the inside out

You do not need to understand all 386 lines at once. Start with the two decisions every template must make.

### 1. Write each role and close each turn

Conceptually, the message loop does this:

```jinja
<|turn>{{ message['role'] }}
{{ message['content'] }}<turn|>
```

Real Jinja adds validation and handles content lists, but the grammar is still “open role, write content, close turn.” An earlier assistant reply is not special; it becomes another closed `model` turn before the next `user` turn.

### 2. Content can be a string or a list

Plain text may be a string. Multimodal messages use a list so text and attachments have an explicit order:

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "path": "cat.jpg"},
        {"type": "text", "text": "What is the cat doing?"},
    ],
}]
```

For this list, the template's inner loop reduces each item to text:

```jinja
{%- if item.get('type') == 'text' -%}
    {{- item['text'] -}}
{%- elif item.get('type') in ['image', 'image_url'] -%}
    {{- '<|image|>' -}}
{%- endif -%}
```

At this stage the image becomes exactly **one** `<|image|>` marker. The template does not inspect pixels and cannot yet know how many model positions the image needs.

### 3. System messages are just an optional first turn

```python
messages = [
    {"role": "system", "content": "Answer as a patient tutor."},
    {"role": "user", "content": "What is a token?"},
]
```

Gemma 4 renders the first message as a `system` turn, then the `user` turn. It also accepts `developer` as an alias for that first role. This is model-specific grammar, not a universal chat standard.

These three rules are enough to read ordinary Gemma 4 prompts. Thinking and tools are optional extensions covered later in this chapter.

## From one image marker to N reserved positions

Now return to the image example. There are two distinct representations:

```text
after the template:   <|image|> What is the cat doing?
after the processor:  <|image> <|image|> × N <image|> What is the cat doing?
```

The names are annoyingly similar. The outside pair—`boi_token` and `eoi_token` in code—marks the boundary of the image block. The repeated `<|image|>` tokens are **placeholders**: empty seats that will later receive image vectors.

`N` is not always 280. Under the default image budget, 280 is the maximum; a small or unusually shaped image can use fewer. `Gemma4Processor` first runs the image processor, reads `num_soft_tokens_per_image`, and then performs the expansion:

```python
def replace_image_token(self, image_inputs, image_idx):
    n = image_inputs["num_soft_tokens_per_image"][image_idx]
    return f"{self.boi_token}{self.image_token * n}{self.eoi_token}"
```

This order explains an otherwise surprising API rule:

- `apply_chat_template(..., tokenize=False)` can show a string with one marker without loading the image.
- For a model-ready multimodal result, the processor needs the actual image (or its path/URL) so it can compute `N`, expand the marker, and tokenize the final string.

The complete convenience call is therefore:

```python
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
)
```

Here the image path lives in the message content. If you instead render the text first and call `processor(text=..., images=...)` yourself, you are taking the same steps manually.

## The token map: grammar tokens versus placeholders

The raw ID list only becomes useful after the tokens are grouped by function:

| Family | Examples | What the IDs mean to the model |
|---|---|---|
| turn grammar | `<bos>`, `<\|turn>`, `<turn\|>` | ordinary learned tokens that describe conversation structure |
| modality boundaries | `boi_token` / `eoi_token`, `boa_token` / `eoa_token` | ordinary learned tokens that mark where a modality block begins and ends |
| modality placeholders | `<\|image\|>`, `<\|audio\|>`, `<\|video\|>` | reserved positions whose text embeddings will be discarded and replaced by modality features |

The exact Gemma 4 interface IDs are:

| Config / processor field | ID | Use |
|---|---:|---|
| `boi_token_id` | 255999 | begin image or video block |
| `boa_token_id` | 256000 | begin audio block |
| `image_token_id` | 258880 | repeated once per image feature vector |
| `audio_token_id` | 258881 | repeated once per audio feature vector |
| `eoi_token_id` | 258882 | end image or video block |
| `eoa_token_index` | 258883 | end audio block; `_index` is the name used by the config |
| `video_token_id` | 258884 | repeated once per video feature vector |

### Why placeholders are rewritten before embedding lookup

First, a correction to an easy misreading: **the released Gemma 4 checkpoints do not currently have out-of-range placeholder IDs.** Their text `vocab_size` is 262,144, while the placeholder IDs are 258,880–258,884. `<|video|>` is added to the tokenizer at processor construction time, but its chosen ID still fits inside the model's embedding table.

The source comment says “replace image id with PAD *if* the image token is OOV,” but the implementation performs the replacement unconditionally. That makes the path safe for custom or resized configurations where tokenizer and model vocabulary sizes may diverge. It also avoids depending on text embeddings that have no lasting meaning: placeholders are **interface sentinels**, not words the decoder needs to preserve.

The model therefore follows this sequence:

1. Find every placeholder position by comparing IDs.
2. Temporarily replace those IDs with the valid `pad_token_id`.
3. Perform the normal text embedding lookup.
4. Overwrite the placeholder positions with real image/audio/video vectors.

[Chapter 08 shows these four operations and the `masked_scatter` that performs step 4](../08-fusion-and-masks/index.md). For now, the important point is that the pad embedding is disposable. In the official checkpoints the substitution is harmless hygiene; in a configuration where a sentinel really is out of range, it also prevents an index error.

The runtime addition of `<|video|>` explains why this can look suspicious:

```python
# FIXME: add the token to config and ask Ryan to re-upload
tokenizer.add_special_tokens({"additional_special_tokens": ["<|video|>"]})
```

Calling `add_special_tokens` does not in general resize a model's embedding matrix. In this checkpoint, however, the matrix already has enough reserved rows. So “video was added late, therefore its lookup is out of bounds” is **not** what happens here. The same safe substitution path handles all three placeholder types because none of their text embeddings are meant to survive.

## Why image batches are nested

This validation belongs here because placeholder expansion is a per-sample contract. For every sample, the number of image markers in its prompt must equal the number of images attached to that sample.

For a batch with two prompts:

| Batch sample | Markers in its prompt | Corresponding images |
|---|---:|---|
| sample 0 | 2 | `[img_a, img_b]` |
| sample 1 | 0 | `[]` |

the manual processor call is:

```python
texts = [prompt_with_two_image_markers, prompt_with_no_image_markers]
images = [[img_a, img_b], []]
inputs = processor(text=texts, images=images, return_tensors="pt")
```

The outer list is the batch; each inner list contains the images for one sample. It is **not** the number of repeated placeholder tokens—each image in an inner list will independently expand to its own run of `N` placeholders. `validate_inputs` checks the two per-sample counts and raises before image features can be attached to the wrong prompt.

When paths or URLs are embedded directly in `messages` and `apply_chat_template(..., tokenize=True)` is used, the processor builds this per-sample grouping for you.

## Audio and video use the same contract

Nothing new about templates is required for the other modalities:

| Template emits | Processor later reserves | Where `N` comes from |
|---|---|---|
| one `<\|image\|>` | `N` image slots | image size and selected token budget |
| one `<\|audio\|>` | `N` audio slots | valid output length of the audio front end |
| one `<\|video\|>` | `N` slots per sampled frame | frame sampling and image processing |

You only need the contract here: **the number of placeholders must equal the number of feature vectors the corresponding tower will return**. The audio count is approximately one token per 40 ms and capped at 750, but deriving that number requires the audio tower. It is explained, with the two stride-2 convolutions, in [chapter 05](../05-audio-and-video/index.md). The optional audio section in [the hands-on notebook](notebooks/01_tokens_and_template.ipynb) calls `_compute_audio_num_tokens` on real waveform lengths without making it prerequisite reading.

Video similarly adds readable `MM:SS` timestamps before sampled-frame blocks; [chapter 05](../05-audio-and-video/index.md) owns the sampling details.

## Optional template features: thinking and tools

Once ordinary turns make sense, the remaining Jinja is easier to place.

### Thinking changes the rendered prompt

`enable_thinking=True` is an argument to the chat template, not to `generate()`. The template creates a system turn if needed and inserts `<|think|>` there. It also removes old reasoning spans from conversation history by default, so scratch work from earlier turns does not grow the prompt forever. The notebook renders the same conversation with thinking off and on so the difference is visible.

### Tools add another serialisation format

When `tools=[...]` is passed, the system turn contains tool declarations. Gemma 4 serialises them in its own compact token-based DSL, and model tool calls use the matching syntax. You do not need to know ChatML or memorise this DSL: give `apply_chat_template` normal JSON-schema dictionaries and let the template serialise them. Inspect the rendered form only when debugging a tool call.

The template deliberately rejects `tool_calls[].function.arguments` when it is still a JSON string; callers must deserialize it to a dictionary first. [The notebook](notebooks/01_tokens_and_template.ipynb) contains one complete rendered example.

## Left padding: a generation convention, not a universal rule

For batched generation with a decoder-only model, use:

```python
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-E2B-it",
    padding_side="left",
)
```

Why it helps is visible in a two-row batch:

```text
left padding                 right padding
[PAD PAD A B C]              [A B C PAD PAD]
[D   E   F G H]              [D E F G   H  ]
             ↑ next-token logits are read from this final column
```

The standard decoder-only generation loop reads next-token logits from the last tensor position for every row. With left padding, that position is a real prompt token in every sample. With right padding, a shorter sample ends on a padding position; an attention mask prevents that position from attending *to padding*, but it does not turn its hidden state into the last real token's hidden state.

Left padding is common for **batched decoder-only inference**, not for every task:

- A single unpadded prompt has no left-versus-right issue.
- Encoder-only models and causal-LM training commonly use right padding; training masks padded labels and does not ask every row's last column for the next token.
- Left padding does not remove wasted padding compute or memory. Length bucketing and packed/continuous batching solve that problem.
- Custom position-ID code or attention kernels may assume valid tokens form a left-aligned prefix. Transformers handles Gemma 4's supported generation path, but custom pipelines must preserve the attention mask and correct position IDs.

So the practical rule is narrower and more useful: **left-pad mixed-length batches that you pass to Gemma 4 `generate()`; do not treat left padding as a universal tokenizer setting.** [Chapter 09](../09-generation-and-serving/index.md) returns to batching, cache behavior, and slicing completions from generated sequences.

## Source map

| File | Symbol | Why read it |
|---|---|---|
| `processing_utils.py` | [`ProcessorMixin.apply_chat_template`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/processing_utils.py#L1805), [`ProcessorMixin.__call__`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/processing_utils.py#L603) | the two-stage render/process path |
| `processing_gemma4.py` | [`Gemma4Processor.prepare_inputs_layout`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L100), [`validate_inputs`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L122) | per-sample image grouping and validation |
| `processing_gemma4.py` | [`replace_image_token`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L153) | one image marker → one boundary-wrapped placeholder run |
| `processing_gemma4.py` | [`replace_audio_token`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L173), [`replace_video_token`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L157) | optional details, after chapter 05 |
| `modeling_gemma4.py` | [`Gemma4Model.forward`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2098) | consumer side; read in chapter 08 |

## Check yourself

1. What information does a chat template add that is absent from the raw user text?
2. Why does the template emit one image marker while `input_ids` contains many image placeholder IDs?
3. In `images=[[img_a, img_b], []]`, what do the outer and inner lists each represent?
4. Does the runtime-added `<|video|>` exceed the released model's embedding table? Why does the model still replace it with `pad_token_id`?
5. What exactly does `add_generation_prompt=True` append?
6. When is left padding useful, and name one situation where right padding remains normal.

<details>
<summary>Show answers</summary>

<ol>
<li>The template adds role labels, turn boundaries, sequence/control tokens, modality markers, and—when requested—the empty model-turn header that starts generation.</li>
<li>The template knows only that an image occurs at that point, so it emits one marker. After inspecting the image, the processor computes <code>N</code> and expands that marker into exactly <code>N</code> reserved positions.</li>
<li>The outer list is the batch. Each inner list contains the images belonging to one sample: two for sample 0 and none for sample 1.</li>
<li>No. The released model has 262,144 embedding rows and <code>video_token_id=258884</code>, so the ID is in range. The model still substitutes <code>pad_token_id</code> because placeholder embeddings are disposable and because the same path protects custom configurations where an ID really is out of range.</li>
<li>It appends an empty <code>&lt;|turn&gt;model\n</code> header, indicating that the next generated content belongs to the model role.</li>
<li>Left padding is useful when mixed-length prompts are batched for decoder-only generation, because every row then ends on a real prompt token. Right padding remains normal for causal-LM training and encoder-only workloads; a single unpadded prompt needs neither.</li>
</ol>

</details>

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| [`01_tokens_and_template.ipynb`](notebooks/01_tokens_and_template.ipynb) | Build one text prompt, then one image prompt; inspect the rendered string, expanded placeholder run, and IDs. Optional extensions cover audio counts, thinking, and tools | 🟢 CPU, tokenizer/processor only |
| [`02_tokenize_everything.ipynb`](notebooks/02_tokenize_everything.ipynb) | A broader comparison of how text, images, and audio are discretised across model families; not required for the Gemma 4 path | 🟢 CPU |
