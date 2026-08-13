# Multimodal-101

Multimodal AI, taught by taking **one model apart** instead of listing many from the outside.

**Part I** follows a single prompt all the way through [Gemma 4](https://huggingface.co/docs/transformers/en/model_doc/gemma4) — config → chat template → image/audio front ends → vision and audio towers → the fusion point where four modalities become one sequence → the decoder (PLE, MoE, sliding-window attention, KV sharing) → `generate()` → LoRA fine-tuning. Every chapter opens the real implementation in `transformers/models/gemma4/`, names the classes, traces the tensors, and rebuilds a piece of it by hand with an assertion that the result matches the library. You learn how a multimodal model works and how to use `transformers` at the same time, because they turn out to be the same lesson.

**Part II** covers what Gemma 4 does not do — text in, pixels and sound out: diffusion/DiT/flow matching, image editing, video, TTS, and unified models. These chapters keep the survey format (theory, landscape with a last-verified date, notebooks), because breadth is the right shape there.

📖 **Read online (English, canonical)**: https://xinli95.github.io/Multimodal-101

📖 **中文版**: https://xinli95.github.io/Multimodal-101/zh/ （理论/格局页面双语维护，notebook 为两版共用的英文版）

Built with the [TeachBooks](https://teachbooks.io) template, same as [AI-101](https://github.com/xinli95/AI-101) and [Speech-101](https://github.com/xinli95/Speech-101). Both editions deploy from one repo via [`deploy-books.yml`](.github/workflows/deploy-books.yml).

## Chapters

| Part I · Anatomy of a Multimodal Model | |
|---|---|
| [00 · Orientation](book/00-orientation/index.md) | Why one model; the family; the dataflow map; CLIP → today |
| [01 · Config](book/01-config/index.md) | How a config becomes a model |
| [02 · Text I/O](book/02-text-io/index.md) | Chat template, tokenizer, placeholder runs |
| [03 · Image Processor](book/03-image-processor/index.md) | Pixels under a fixed token budget |
| [04 · Vision Tower](book/04-vision-tower/index.md) | Patch embedding, 2D RoPE, pooling, projection |
| [05 · Audio and Video](book/05-audio-and-video/index.md) | Mel features, chunked attention, frame sampling |
| [06 · Text Decoder](book/06-text-decoder/index.md) | Sliding vs. global attention, KV sharing |
| [07 · PLE and MoE](book/07-ple-and-moe/index.md) | Per-Layer Embeddings; routed experts |
| [08 · Fusion and Masks](book/08-fusion-and-masks/index.md) | Where the modalities actually meet |
| [09 · Generation and Serving](book/09-generation-and-serving/index.md) | `generate()`, caches, batching, vLLM |
| [10 · Fine-Tuning](book/10-finetuning/index.md) | What to freeze, LoRA, collators, memory |
| [Landscape](book/landscape.md) | Where Gemma 4 sits among everything else that reads |

| Part II · The Generation Side | |
|---|---|
| [20 · Image Generation and Editing](book/20-image-generation/overview.md) | Diffusion → DiT → Flow Matching; instruction editing |
| [21 · Video Generation](book/21-video-generation/overview.md) | Video DiT, temporal attention, causal VAE |
| [22 · Audio Generation](book/22-audio-generation/overview.md) | Codec tokens, TTS as a language model |
| [23 · Unified and Omni Models](book/23-unified-omni/overview.md) | Understanding *and* generation in one model |
| [24 · Applications and Evaluation](book/24-applications/overview.md) | Multimodal RAG, agents, benchmarks |

## Build locally

```bash
pip install -r requirements.txt
teachbooks build book       # English edition
teachbooks build book-zh    # Chinese edition
```

Then open `book/_build/html/index.html` (or `book-zh/_build/html/index.html`).

Part I is written against **transformers 5.14.1**. Its behavioural notebooks use `google/gemma-4-E2B-it` (~10GB, not gated, fits a single 24GB GPU); its structural notebooks use randomly-initialised miniature configs and run on a CPU with no downloads.

## Contributing / translation policy

- **English (`book/`) is the source of truth** — content changes land there first.
- The Chinese edition (`book-zh/`) mirrors the same file layout; translate the changed pages after the English side merges.
- Notebooks (`book/*/notebooks/*.ipynb`) exist only in English and are shared by both editions; the Chinese notebook index pages link to them.

## License

Content CC BY 4.0, code MIT (see [LICENSE](LICENSE)). Model licenses vary — Gemma 4 is under the Gemma terms; Part II's models are marked in each chapter's `landscape.md`. Check before commercial use.
