# 00 · Orientation — One Model, All the Way Down

Most multimodal tutorials are catalogues: here is CLIP, here is LLaVA, here is Qwen-VL, here is a diffusion model. You finish knowing the names and still cannot answer the only question that matters — *what is actually inside one of these things, and where is that code?*

Part I of this book takes the opposite approach. We pick **one** model, [Gemma 4](https://huggingface.co/docs/transformers/en/model_doc/gemma4), and take it apart layer by layer, following its real implementation in `transformers`. Every chapter opens a specific file, names specific classes, traces specific tensors, and then rebuilds a piece of it by hand to prove we understood it.

## Why Gemma 4 is the right teaching model

| Property | Why it matters for learning |
|---|---|
| Text + **image + video + audio** input in one checkpoint | You see all four modality front-ends inside a single, consistent codebase — not four blog posts about four models |
| Variable aspect ratio under a **fixed token budget** | The modern answer to "how do images become tokens", and unlike fixed 224×224 squashing you can actually see the trade-off |
| **2D RoPE + learned 2D position table** in the vision tower | Position encoding for a grid, not a line — the concept that makes spatial reasoning work |
| **Per-Layer Embeddings (PLE)** | A genuinely unusual design you will not find in a textbook; forces you to read the code instead of pattern-matching |
| **Sliding-window + global** attention mix, cross-layer KV sharing, K=V projections | Every serious inference-cost trick in one decoder |
| **MoE** in the 26B-A4B size | Routing and sparse experts, same codebase |
| Sizes from **E2B to 31B**, none gated | The small one runs on a single consumer GPU; the big ones exist for the chapters that need them |
| Open implementation in `transformers` | The source *is* the reference. No guessing from a paper figure |

A fair objection: learning one model teaches you one model. That is why every chapter ends with a **Design space** section that places Gemma 4's choice against the alternatives (LLaVA's MLP projector, BLIP-2's Q-Former, Qwen-VL's M-RoPE, InternVL's tiling, Whisper's audio encoder). The single model is the spine; the design-space sections are the map.

## What you will learn in this chapter

1. What the Gemma 4 family actually contains — sizes, modalities, context lengths, and which capabilities are size-dependent
2. Where multimodal understanding came from, in one page (CLIP → BLIP-2 → LLaVA → today), so the rest of the book has a lineage to hang on
3. How to set up an environment that can run everything in Part I
4. Run Gemma 4 end to end once, print its module tree, and map every branch of that tree to a chapter of this book

## The map: the dataflow we are going to take apart

Everything in Part I is one stage of this pipeline. Keep this diagram; every chapter marks its position on it.

```
  messages ──► chat template ──► tokenizer ──► input_ids                       ch02
               (placeholder runs: <image>×280, <audio>×N, <video>×...)
                                   │
  image ──► Gemma4ImageProcessor ──┤► pixel_values + image_position_ids        ch03
                                   │
                                   ▼
                          Gemma4VisionModel                                    ch04
        patch embed ─► 2D RoPE ─► encoder ×16 ─► pooler ─► soft tokens
                                   │
  audio ──► Gemma4AudioFeatureExtractor ─► Gemma4AudioModel ─► soft tokens     ch05
  video ──► Gemma4VideoProcessor ─► (frames) ─► Gemma4VisionModel              ch05
                                   │
                                   ▼
                     Gemma4MultimodalEmbedder  (─► text embedding space)       ch04/05
                                   │
                                   ▼
        masked_scatter soft tokens into inputs_embeds  +  build the 4D mask    ch08
                                   │
                                   ▼
                           Gemma4TextModel                                     ch06
            PLE ─► [sliding | global] attention ─► MLP or MoE  ×N              ch07
                                   │
                                   ▼
                    lm_head ─► logits ─► generate() ─► text                    ch09
                                   │
                                   ▼
                        LoRA / Trainer: change the weights                     ch10
```

## The family, in one table

| Size | Params | Text | Image | Video | Audio | Context |
|---|---|---|---|---|---|---|
| **E2B** | ~2B effective | ✅ | ✅ | ✅ | ✅ | 128K |
| **E4B** | ~4B effective | ✅ | ✅ | ✅ | ✅ | 128K |
| **31B** | 31B dense | ✅ | ✅ | ✅ | ❌ | 256K |
| **26B-A4B** | 26B MoE, ~4B active | ✅ | ✅ | ✅ | ❌ | 256K |

All four are released as pretrained and instruction-tuned (`-it`) variants, all are reasoners with configurable thinking modes, all support a native `system` role and function calling. **Audio is only native on E2B and E4B** — that asymmetry is itself instructive, and chapter 05 explains what the audio tower costs.

Part I uses **`google/gemma-4-E2B-it`** as the default (single 24GB GPU, ~10GB of weights, not gated) and drops to randomly-initialised miniature configs whenever the point is structural rather than behavioural.

## Prehistory in one page

You do not need this to read the book, but it explains why Gemma 4 looks the way it does.

- **CLIP** (2021) — two towers, contrastive InfoNCE loss, image and text pushed into one embedding space. It gave the field a vision encoder whose features are already semantically aligned with language. Almost every VLM vision tower since is a CLIP-family descendant (SigLIP being the sigmoid-loss refinement that decoupled batch size from quality).
- **Flamingo** (2022) — inject visual features into a frozen LLM through gated cross-attention. Powerful, complicated.
- **BLIP-2** (2023) — a Q-Former compresses visual features into a small fixed set of query tokens. Elegant, and an extra module to train.
- **LLaVA** (2023) — a single MLP maps ViT patch features straight into the LLM's token embedding space, and you train on good instruction data. **This simplest route won.** Worth sitting with: data quality beat architectural cleverness.
- **Today** (Qwen3-VL, InternVL3.5, Gemma 4) — SigLIP-class tower, a projector, a strong LLM, everything trained jointly, plus the hard-won details: native/variable resolution, token-budget control, spatial position encoding, and long context.

Gemma 4 is squarely in the LLaVA lineage: soft tokens from an encoder, projected into the text embedding space, scattered into the sequence. What it adds is everything in the "hard-won details" column — which is exactly what chapters 03–08 are about.

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_hello_gemma4.ipynb` | Run Gemma 4 end to end (`pipeline` and `AutoModelForImageTextToText`), then `print(model)` and map every submodule to a chapter of this book | 🟡 24GB VRAM, ~10GB download |
| [`02_clip_zero_shot.ipynb`](notebooks/02_clip_zero_shot.ipynb) | The control experiment: CLIP does zero-shot classification but cannot count cats. Establishes what a *contrastive encoder* can and cannot do — the gap Gemma 4's LLM fills | 🟢 CPU |

## Environment

```bash
pip install "transformers>=5.14" accelerate torch pillow requests matplotlib
```

Part I is written against **transformers 5.14.1**, where Gemma 4 lives at `transformers/models/gemma4/`. Find your copy — you will be reading it constantly:

```python
import transformers, os
print(os.path.join(os.path.dirname(transformers.__file__), "models", "gemma4"))
```

```
configuration_gemma4.py      ch01    the four config classes
processing_gemma4.py         ch02    Gemma4Processor: the multimodal front door
image_processing_gemma4.py   ch03    pixels → patches under a token budget
video_processing_gemma4.py   ch05    frame sampling
feature_extraction_gemma4.py ch05    waveform → mel features
modeling_gemma4.py           ch04-09 everything else, ~126KB
modular_gemma4.py            ch01    the source modeling_gemma4.py is generated from
```
