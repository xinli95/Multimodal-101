# Multimodal-101

A hands-on introduction to multimodal AI, organised around **one model taken apart layer by layer** rather than a catalogue of models described from the outside. Part of the 101 series alongside [AI-101](https://github.com/xinli95/AI-101) and [Speech-101](https://github.com/xinli95/Speech-101), built as a browsable book with [TeachBooks](https://teachbooks.io). A [Chinese edition](https://xinli95.github.io/Multimodal-101/zh/) is maintained in parallel.

Target readers: engineers and researchers with a basic Python / deep-learning background who want to know what is actually inside a multimodal model — and to learn `transformers` properly along the way, because that is where the answers live.

## How this book is organised

**Part I · Anatomy of a Multimodal Model** takes [Gemma 4](https://huggingface.co/docs/transformers/en/model_doc/gemma4) and follows one prompt all the way through it: config → chat template and tokenizer → image and audio front ends → vision and audio towers → the fusion point where four modalities become one sequence → the decoder → `generate()` → fine-tuning. Every chapter opens the real implementation in `transformers/models/gemma4/`, names the classes and functions, traces the tensors, and then rebuilds a piece of it by hand and asserts the result matches the library. Chapter 11 is the transfer test: apply that complete method to the compact GLM-OCR specialist and then widen the boundary from checkpoint to document system.

Gemma 4 earns this role because one open checkpoint family contains almost everything worth teaching: text, image, video and audio input; variable aspect ratios under a fixed token budget; 2D RoPE; Per-Layer Embeddings; sliding-window and global attention mixed together; cross-layer KV sharing; MoE; 128K–256K context. And it is small enough at E2B to run on one consumer GPU.

**Part II · The Generation Side** covers what Gemma 4 does not do: text in, pixels and sound out. Diffusion, DiT, flow matching, video, TTS, and unified models that do both directions. These chapters keep the survey format — theory, landscape, notebooks — because breadth is the right shape there.

## Design principles

1. **A spine, not a list.** Chapters 00–10 are one continuous path through one model. Chapter 11 checks that the method transfers instead of leaving you dependent on Gemma-specific names.
2. **Read the source.** Architecture claims in this book are checkable: they point at a file and a symbol in your own `transformers` install. Where a mechanism matters, a notebook reimplements it and `assert_close`s against the library.
3. **One model, with a map.** Every Part I chapter ends with a **Design space** section placing Gemma 4's choice against the alternatives (LLaVA, BLIP-2, Flamingo, Qwen-VL, InternVL, Whisper). The single model is the spine; those sections are the map.
4. **Fighting staleness.** This field reshuffles roughly every quarter. Time-sensitive claims carry an explicit **last-verified date** where they appear. If that date is more than 6 months old, re-check against official sources.

## Part I chapters

| Chapter | Subject | Source it opens |
|---|---|---|
| [00 · Orientation](00-orientation/index.md) | Why one model; the Gemma 4 family; the dataflow map; prehistory from CLIP to today | — |
| [01 · Config](01-config/index.md) | How a config becomes a model; nested text/vision/audio configs | `configuration_gemma4.py` |
| [02 · Text I/O](02-text-io/index.md) | Chat template, tokenizer, placeholder runs, function calling | `processing_gemma4.py` |
| [03 · Image Processor](03-image-processor/index.md) | Pixels under a token budget; aspect ratio; the 48-divisibility rule | `image_processing_gemma4.py` |
| [04 · Vision Tower](04-vision-tower/index.md) | Patch embedding, 2D RoPE, pooling, projection into text space | `modeling_gemma4.py` |
| [05 · Audio and Video](05-audio-and-video/index.md) | Mel features, convolutional subsampling, chunked attention; frame sampling | `feature_extraction_gemma4.py`, `video_processing_gemma4.py` |
| [06 · Text Decoder](06-text-decoder/index.md) | Sliding vs. global attention, QK-norm, KV sharing, K=V | `modeling_gemma4.py` |
| [07 · PLE and MoE](07-ple-and-moe/index.md) | Per-Layer Embeddings; routed experts | `modeling_gemma4.py` |
| [08 · Fusion and Masks](08-fusion-and-masks/index.md) | Where the modalities actually meet; bidirectional vision attention | `modeling_gemma4.py` |
| [09 · Generation and Serving](09-generation-and-serving/index.md) | `generate()`, cache implementations, batching, vLLM | `modeling_gemma4.py` |
| [10 · Fine-Tuning](10-finetuning/index.md) | What to freeze, LoRA, multimodal collators, memory arithmetic | `peft`, `Trainer` |
| [11 · GLM-OCR](11-glm-ocr/index.md) | Specialist OCR as a transfer test: base model, MTP serving, layout pipeline, contract-aware eval | `models/glm_ocr/`, GLM-OCR SDK |
| [Landscape](landscape.md) | Where Gemma 4 sits among everything else that reads | — |

## Part II chapters

| Chapter | Topic | Core theory | Practice models (open / closed) |
|---|---|---|---|
| [20 · Image Generation and Editing](20-image-generation/index.md) | Text → image, and instruction editing | Diffusion → DiT → Flow Matching; in-context conditioning | FLUX.2 [klein], Qwen-Image-Edit / GPT Image 2, Nano Banana |
| [21 · Video Generation](21-video-generation/overview.md) | Text/image → video | Video DiT, temporal attention, causal VAE | Wan 2.2, HunyuanVideo 1.5 / Veo 3.1 |
| [22 · Audio Generation](22-audio-generation/index.md) | Text → speech | Waveforms → codecs / flow matching → speech | Kokoro, F5-TTS, Higgs TTS 3 |
| [23 · Unified and Omni Models](23-unified-omni/overview.md) | Understanding *and* generation in one model | Architectures that unify both directions | BAGEL, InternVL-U, Qwen3.5-Omni |
| [24 · Applications and Evaluation](24-applications/overview.md) | Multimodal RAG, agents, benchmarks | Retrieval over documents, judging, routing | Everything above, combined |

## Suggested paths

- **Full path**: Part I 00 → 10 in order, then 11 as the transfer capstone — then Part II as interest dictates.
- **I only care how VLMs work**: Part I 00 → 04, then 08. Chapters 03, 04 and 08 are the load-bearing ones.
- **I only care about inference cost**: Part I 03 (token budget) → 06 (attention and KV cache) → 09 (generation and serving).
- **I only care about OCR and documents**: Part I 03 → 04 → 08 → 09, then 11 and chapter 24's document pipeline/RAG notebooks.
- **I only care about generation**: Part I 00, then Part II 20 → 23.

## Hardware tiers

Every notebook is tagged with a minimum hardware tier. Part I is deliberately built so that the *structural* notebooks need no GPU and no weight downloads — they use randomly-initialised miniature configs — while the *behavioural* ones use `google/gemma-4-E2B-it` (~10GB, not gated) or the compact `zai-org/GLM-OCR` checkpoint (~2.66GB BF16).

| Tier | Setup | What runs (examples) |
|---|---|---|
| 🟢 CPU / laptop | No GPU required | Config and mask anatomy, image processor reimplementation, tokenizer work, CLIP, Kokoro, Whisper, closed-API calls |
| 🟡 Consumer GPU | 12–24GB VRAM | Gemma 4 E2B end to end, LoRA SFT, Qwen3-VL small sizes, FLUX.2 [klein] |
| 🔴 Workstation / cloud | 40GB+ VRAM | Gemma 4 31B / 26B-A4B, Wan 2.2, HunyuanVideo 1.5, large InternVL3.5 |

## A note on licenses

The tutorial code itself is MIT. Model licenses vary widely (Apache 2.0 / Gemma terms / non-commercial / tiered commercial) — Gemma 4 is released under the Gemma terms, and Part II's models are marked individually in each chapter's `landscape.md`. Check before commercial use.

---

*Landscape data last verified: 2026-08-02. Part I is written against transformers 5.14.1.*
