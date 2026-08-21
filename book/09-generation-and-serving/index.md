# 09 · Generation and Serving — From `forward` to a Running Service

**Position in the pipeline**: `hidden ──► lm_head ──► logits ──► generate() ──► text`, and then out of the notebook entirely.

Everything so far described a single forward pass. Generation is that forward pass in a loop, and almost all of the engineering is about not redoing work: the KV cache. Chapter 06 explained why Gemma 4's cache is shaped the way it is — sliding-window layers keep only a window, some layers keep nothing of their own at all. This chapter is where that pays off, and where you learn the `transformers` generation API well enough to debug it.

## What you will learn

1. What `Gemma4ForConditionalGeneration` adds on top of `Gemma4Model`, and why the docs' three examples reach for three different `Auto*` classes (`AutoModelForImageTextToText`, `AutoModelForCausalLM`, `AutoModelForMultimodalLM`)
2. The `Cache` classes: dynamic vs. `cache_implementation="static"`, what static buys (compile-friendly shapes) and what it costs
3. Batched multimodal generation: why `padding_side="left"` is required, and how to slice generated tokens off correctly (`input_len` and why it is not optional)
4. Streaming output, and controlling the thinking budget on a reasoning model
5. Quantisation options and what they do to a model with vision and audio towers attached
6. Serving it for real: an OpenAI-compatible endpoint via vLLM, and how to call it

## Source map

| Symbol | Role |
|---|---|
| [`Gemma4ForConditionalGeneration`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2507) | `Gemma4Model` + `lm_head` + `GenerationMixin` |
| [`Gemma4ForCausalLM`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1825) | The text-only path, for text-only workloads |
| [`Gemma4CausalLMOutputWithPast`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L109) | Logits, cache, and the multimodal extras |
| [`GenerationMixin.generate`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/generation/utils.py#L2274) | Not Gemma-specific — but read it once |

## Walkthrough

### 1. Three `Auto*` classes, one checkpoint

The Gemma 4 docs use a different class in each of their three examples, which looks arbitrary until you know what each maps to:

| Example | Class | What you get |
|---|---|---|
| image + text | `AutoModelForImageTextToText` | `Gemma4ForConditionalGeneration` |
| function calling (text only) | `AutoModelForCausalLM` | `Gemma4ForCausalLM` |
| audio | `AutoModelForMultimodalLM` | `Gemma4ForConditionalGeneration` |

The two multimodal aliases resolve to the same class; they exist so task-oriented code can express intent. `AutoModelForCausalLM` is the one that differs — it gives you `Gemma4ForCausalLM`, the **text-only** model with no vision or audio tower. If your workload is pure text, loading that class avoids materialising towers you will never call.

The two heads differ only in what they wrap:

```python
class Gemma4ForCausalLM(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: Gemma4TextConfig):
        self.model = Gemma4TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

class Gemma4ForConditionalGeneration(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
```

Note the tied-weights path: one level deeper for the multimodal model, because its `Gemma4Model` holds the language model as a submodule. `lm_head` is tied to the input embedding throughout — with a 262,144-token vocabulary at 1536 wide, that is ~400M parameters not spent twice.

### 2. Logits, softcapping, and `logits_to_keep`

```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
if self.config.final_logit_softcapping is not None:
    logits = torch.tanh(logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
```

`logits_to_keep` is the small optimisation that matters most during generation. Projecting a full 128K-token sequence to a 262,144-wide vocabulary would produce a tensor of tens of gigabytes; during decoding you need exactly one position. `generate` passes `logits_to_keep=1` for you. It is worth knowing about because if you call `forward` directly in a loop and wonder where your memory went, this is the answer.

The softcap (30.0 on all sizes) bounds every logit into ±30 smoothly, so no single token's score can run away. It slightly flattens the distribution, which interacts with sampling temperature — one reason Gemma's shipped `generation_config.json` looks the way it does:

```json
{"do_sample": true, "temperature": 1.0, "top_k": 64, "top_p": 0.95,
 "eos_token_id": [1, 106, 50], "pad_token_id": 0, "bos_token_id": 2}
```

**Sampling is on by default**, and there are **three** EOS IDs — end-of-sequence, end-of-turn, and one more. `generate` stops at any of them. If you are diffing outputs against a reference implementation and they diverge, check whether you accidentally suppressed one of these; and if you want determinism, pass `do_sample=False` explicitly rather than assuming greedy.

### 3. The cache, and why Gemma 4's is unusual

The library's `Cache` objects hold per-layer keys and values. Gemma 4 complicates that in two ways you already know from chapter 06:

- **Sliding layers do not need full-length cache.** A layer that only ever attends 512 tokens back can discard everything older. The cache implementation exploits this, which is what keeps memory sub-linear in context length for 5 of every 6 layers.
- **KV-shared layers hold nothing at all.** They read `shared_kv_states`, a dict passed through the forward pass, rather than a cache:

  ```python
  if past_key_values is not None and not self.is_kv_shared_layer:
      key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
  ```

  And `Gemma4CausalLMOutputWithPast` carries `shared_kv_states` alongside `past_key_values` for exactly this reason.

`cache_implementation="static"` — used in the docs' first example — preallocates a fixed-size cache instead of growing a dynamic one. Fixed shapes are what `torch.compile` needs to avoid recompiling on every new sequence length, so static cache plus compile is the standard fast-decoding recipe. The cost is that you must size it for your longest expected sequence and pay that memory from the start.

### 4. Batched multimodal generation

Chapter 02 covered `padding_side="left"`. Two more things bite in practice.

**Slice the prompt off yourself:**

```python
input_len = inputs["input_ids"].shape[-1]
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0][input_len:], skip_special_tokens=False))
```

`generate` returns prompt + completion, always.

**`skip_special_tokens` is a decision, not a default.** With `True` you get clean prose. With `False` you see `<|turn>`, `<|channel>thought`, `<|tool_call>` and friends — which is the only way to debug tool calling or inspect thinking output, since those are exactly the tokens you need. The docs' function-calling and audio examples both use `False` deliberately.

### 5. Thinking

Chapter 02 §5 showed that `enable_thinking` is a chat-template argument, not a generation flag:

```python
inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                       add_generation_prompt=True, enable_thinking=True)
```

It injects `<|think|>` into the system turn, and the model responds by emitting a `<|channel>thought … <channel|>` span before its answer. Consequences for serving:

- Thinking tokens are **generated tokens** — they cost latency and money, and they count against `max_new_tokens`. Set that budget with thinking in mind or you will get truncated reasoning and no answer.
- The template strips prior thinking from history (`strip_thinking`), so multi-turn conversations do not accumulate it. If you are managing history yourself, do the same.
- To display a thinking/answer split, generate with `skip_special_tokens=False` and split on the channel markers.

### 6. Serving with vLLM

vLLM ships Gemma 4 support — `Gemma4ForConditionalGeneration` is in its multimodal model registry. Standard invocation:

```bash
vllm serve google/gemma-4-E2B-it \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 4, "audio": 2}'
```

then call it like the OpenAI API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "google/gemma-4-E2B-it",
       "messages": [{"role": "user", "content": [
         {"type": "image_url", "image_url": {"url": "https://…/cat.jpg"}},
         {"type": "text", "text": "What is in this image?"}]}]}'
```

Two Gemma-4-specific notes. `--max-model-len` matters more than usual: the sliding/global mix makes long contexts cheaper than for a uniformly-global model, but the global layers still dominate, so raising it is not free. And `--limit-mm-per-prompt` is worth setting deliberately — chapter 03 told you each image costs 280 tokens and chapter 05 that a video costs 2,240, so the multimodal limits *are* context limits.

If you are on a Vast.ai vLLM image, the server is already wired up as a supervisor service; set `VLLM_MODEL` and `MODEL_NAME` and restart rather than launching a second copy.

## Design space

The interesting comparison here is not between inference engines but between **where multimodal preprocessing happens**.

- **`transformers` + `generate`**: the processor runs in your Python process, the towers run inside `forward`. Simple, debuggable, and the only place to do the kind of hooking this book relies on. Throughput is not the goal.
- **vLLM**: the processor still runs per request, but the towers run inside the engine and soft tokens are merged into the sequence by `_merge_multimodal_embeddings` — the function chapter 02's docstring warned must agree with the processor's token count. Continuous batching and paged attention make it the right answer for serving.
- **Precomputed embeddings**: run the towers once, cache the soft tokens, and feed `inputs_embeds`. Attractive for repeated queries over the same document — and, on Gemma 4 specifically, the case where you must also precompute `per_layer_inputs` (chapter 07 §4) or pay for an embedding inversion.

That last row is a good illustration of the theme of Part I: an architectural choice made for on-device efficiency (PLE) shows up two hundred pages later as a constraint on a serving optimisation. You can only see connections like that by having read the whole path.

## Check yourself

1. You are serving text-only traffic from a Gemma 4 checkpoint. Which `Auto*` class should you load and what do you save?
2. What does `logits_to_keep` prevent, and roughly how large would the tensor be without it at 32K context?
3. Why do sliding-window layers make long-context serving cheaper, and which layers still dominate the bill?
4. Your outputs are non-deterministic across runs despite passing no sampling arguments. Why?
5. `enable_thinking=True` and `max_new_tokens=100`. What is the likely failure mode?
6. You want to cache a long document's soft tokens across many questions. What must you cache besides `inputs_embeds`?

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_generate_and_cache.ipynb` | Measure prefill vs. decode, dynamic vs. static cache, batch size vs. throughput on real E2B weights; then a streaming example and a thinking-budget comparison | 🟡 24GB VRAM |

Deployment is covered as commands in this chapter's text rather than in a notebook — a server does not belong in a cell.
