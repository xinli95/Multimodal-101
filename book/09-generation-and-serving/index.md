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
| `Gemma4ForConditionalGeneration` | `Gemma4Model` + `lm_head` + `GenerationMixin` |
| `Gemma4ForCausalLM` | The text-only path, for text-only workloads |
| `Gemma4CausalLMOutputWithPast` | Logits, cache, and the multimodal extras |
| `GenerationMixin.generate` | Not Gemma-specific — but read it once |

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_generate_and_cache.ipynb` | Measure prefill vs. decode, dynamic vs. static cache, batch size vs. throughput on real E2B weights; then a streaming example and a thinking-budget comparison | 🟡 24GB VRAM |

Deployment is covered as commands in this chapter's text rather than in a notebook — a server does not belong in a cell.
