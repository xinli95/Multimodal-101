# Multimodal-101

A hands-on introduction to multimodal AI that takes both **theory** and **practice** seriously. Part of the 101 series alongside [AI-101](https://github.com/xinli95/AI-101) and [Speech-101](https://github.com/xinli95/Speech-101), built as a browsable book with [TeachBooks](https://teachbooks.io). A [Chinese edition](https://xinli95.github.io/Multimodal-101/zh/) is maintained in parallel.

Target readers: engineers and researchers with basic Python / deep-learning background who want a systematic understanding of how multimodal models work — and to actually run the latest open-source models and frontier closed APIs, not just read about them.

## Design principles

1. **Theory and practice, side by side.** Every chapter ships `theory.md` (principles and key papers) plus runnable `notebooks/`. Theory explains *why the architecture looks the way it does*; practice guarantees *you can run it today*.
2. **Open source and closed frontier, in contrast.** Each topic offers two tracks: an open model you can run locally, and a frontier closed API — so you understand the internals *and* know where the capability ceiling is.
3. **Fighting staleness.** This field reshuffles roughly every quarter. Each chapter's `landscape.md` records the current state of play with an explicit **last-verified date**. If the date is more than 6 months old, re-check against official sources.
4. **Evolution over enumeration.** Instead of listing models, we trace the technical lineage (e.g. image generation: DDPM → LDM → DiT → Flow Matching), so when a new model drops you can place it on the map yourself.

## Chapters

| Chapter | Topic | Core theory | Practice models (open / closed) |
|---|---|---|---|
| [00-foundations](00-foundations/overview.md) | Multimodal foundations | Contrastive learning, three generative paradigms | CLIP / SigLIP |
| [01-vlm](01-vlm/overview.md) | Vision-language models | ViT + connector architecture, three-stage training | Qwen3-VL, InternVL3.5 / Gemini, GPT-5 |
| [02-doc-understanding](02-doc-understanding/overview.md) | Document understanding & OCR | Optical context compression | DeepSeek-OCR-2, olmOCR-2 / closed VLM APIs |
| [03-image-generation](03-image-generation/overview.md) | Image generation | Diffusion → DiT → Flow Matching | FLUX.2 [klein], Qwen-Image / GPT Image 2 |
| [04-image-editing](04-image-editing/overview.md) | Image editing | Instruction-based editing, in-context conditioning | Qwen-Image-Edit, FLUX Kontext / Nano Banana |
| [05-video-generation](05-video-generation/overview.md) | Video generation | Video DiT, temporal attention, causal VAE | Wan 2.2, HunyuanVideo 1.5 / Veo 3.1 |
| [06-audio](06-audio/overview.md) | Speech & audio | ASR architectures, codec-token LMs | Whisper, Kokoro, Higgs Audio v3 / closed TTS |
| [07-unified-omni](07-unified-omni/overview.md) | Unified multimodal models | Architectures unifying understanding + generation | BAGEL, InternVL-U, Qwen3.5-Omni |
| [08-applications](08-applications/overview.md) | Applications & evaluation | Multimodal RAG, agents, benchmarks | Everything above, combined |

## Suggested paths

- **Full path**: 00 → 01 → 03 → 05 → 06 → 07, inserting the remaining chapters as needed. Chapter 00 is the foundation and 07 is the capstone — don't skip around those two.
- **Understanding only (VLM)**: 00 → 01 → 02 → 08.
- **Generation only**: 00 → 03 → 04 → 05 → 06.

## Hardware tiers

Every notebook is tagged with a minimum hardware tier:

| Tier | Setup | What runs (examples) |
|---|---|---|
| 🟢 CPU / laptop | No GPU required | CLIP, Kokoro-82M, Whisper, closed-API calls |
| 🟡 Consumer GPU | 12–24GB VRAM | Qwen3-VL small sizes, FLUX.2 [klein] 4B, SD 3.5 |
| 🔴 Workstation / cloud | 40GB+ VRAM | Wan 2.2, HunyuanVideo 1.5, InternVL3.5 large sizes |

## A note on licenses

The tutorial code itself is MIT. The models used across chapters vary widely (Apache 2.0 / non-commercial / tiered commercial); each chapter's `landscape.md` marks them explicitly — check before commercial use.

---

*Landscape data last verified: 2026-08-02*
