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
4. Build a high-level mental model first; then run Gemma 4, print its module tree, and map implementation details onto that model

## The mental model: three learned blocks

Forget the class names for a moment. For an image, most VLMs can be understood as three learned blocks:

```text
                    What does each stage decide?

image
  │
  ├─ processor ───► How much visual compute should this image receive?
  │                 (resize, patch grid, token budget; no semantic understanding)
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. VISION TOWER                                                │
│     local patch embeddings ──► contextual visual features       │
│     "What is in the image, and how do its regions relate?"      │
└──────────────────────────────────────────────────────────────────┘
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. CONNECTOR / COMPRESSOR                                      │
│     many vision-width features ──► fewer LLM-width soft tokens   │
│     "How much visual information enters the language model?"    │
└──────────────────────────────────────────────────────────────────┘
  ▼
visual tokens ──────────────┐
                            ├─► one token sequence ─► 3. LLM ─► text
text ─► tokenizer ──────────┘                       reason + generate
```

That is the architecture-level model to keep in your head:

> **A modality tower understands its own input; a connector compresses and translates that representation into the LLM's token space; the LLM reasons over one mixed sequence.**

Audio follows the same pattern with an audio tower. Video is frames through the vision path plus temporal metadata. The processor and fusion code matter, but they are plumbing around these three learned blocks: the processor allocates visual compute before the tower, and fusion decides where the resulting soft tokens sit in the shared sequence.

### Follow one image: five transformations

The three-block picture becomes concrete if you track what changes, and what does *not* change, at each boundary. Let `B` be the requested final visual-token budget, `D_v` the vision width, and `D_text` the LLM width.

| Stage | Representation | The job | Where this book zooms in |
|---|---|---|---|
| 1. Allocate resolution | `H×W×3` pixels → resized pixels + patch coordinates | Spend the budget while preserving aspect ratio; in Gemma 4, allow up to roughly `9B` patches because 3×3 patches later become one token | ch03 |
| 2. Patch embed | local `16×16×3` pixels → `N×D_v` patch embeddings | Turn local pixel blocks into vectors and attach spatial position | ch04 |
| 3. Vision encode | `N×D_v` → `N×D_v` contextual features | Let every patch representation incorporate image context; the ViT is primarily an **encoder**, not the token compressor | ch04 |
| 4. Compress + align | `N×D_v` → about `B×D_text` soft tokens | 3×3 spatial pooling reduces sequence length; a projection changes representation width into the LLM embedding space | ch04 |
| 5. Fuse + reason | `B` visual tokens + `T` text tokens → one `(B+T)×D_text` sequence | Place both modalities in one sequence, build the mask, and let the text decoder produce logits | ch08 → ch06/07 → ch09 |

This separation prevents three common confusions:

- **Patch vector is not yet a contextual visual token.** It must be projected, positioned, and encoded.
- **The ViT does not mainly reduce token count.** In Gemma 4, the pooler performs that compression after the ViT.
- **Pooling and projection solve different problems.** Pooling changes sequence length; projection changes feature width.

### Eleven chapters, four questions

The chapter boundaries are implementation zoom levels, not eleven unrelated ideas. At the highest level, Part I asks only four questions:

| Big question | Chapters | What to retain on a first pass |
|---|---|---|
| What components should exist? | 01 | The config is the blueprint for towers, connector, and decoder |
| How does each input become an LLM-space token? | 02–05, 08 | Preprocess → encode → compress/align → place in one sequence |
| What happens once everything is one sequence? | 06–07 | Attention and feed-forward layers transform a shared residual stream |
| How do we use or change the model? | 09–11 | Generate, fine-tune, serve, and apply the same reading method to another system |

On a first read, learn the rows above. Use the individual chapters when you need to open a box and understand its mechanism.

### A five-question template for any VLM

When you meet LLaVA, Qwen-VL, InternVL, or another model, do not start by memorising its class tree. Ask:

1. **What is the vision tower family?** A CLIP/SigLIP-style ViT, InternViT, or a custom ViT?
2. **How was it initialised?** From an independently pretrained checkpoint, or trained for this multimodal model?
3. **Does multimodal training freeze or update it?** Pretrained and frozen are different claims.
4. **How is resolution allocated?** Fixed resize, tiling, native/dynamic resolution, or an explicit token budget?
5. **How do tower features become LLM tokens?** Pooling, pixel shuffle, learned merger, resampler, Q-Former, or a projector?

Those five answers explain most of a VLM's vision architecture before you read a single implementation detail. In particular:

> **ViT is the neural-network architecture; CLIP/SigLIP describes how a vision-language representation may be pretrained; VLM training describes how the vision system and LLM are connected and updated. These are three different layers of the story.**

## The implementation map: now zoom in

Only after the mental model is stable is the detailed class-level map useful. Everything in Part I is one stage of this pipeline; every chapter marks its position on it.

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

## How to read Part I

Each chapter has the same four parts:

- **Front matter** — where you are in the pipeline, what you will learn, and a source map naming every file and symbol the chapter opens.
- **Walkthrough** — the actual reading. Code is quoted from `transformers 5.14.1`; where a design choice is not obvious, the chapter argues for it rather than asserting it.
- **Design space** — how other models answered the same question, and what each answer optimises. This is the section that keeps one model from becoming a blind spot.
- **Check yourself** — questions with answers findable in the chapter or the source. If you can answer them you have read it; if not, you have skimmed it.

Two habits will make the difference:

1. **Keep the source open.** Every code block in Part I is quoted from a file on your disk. Reading the surrounding lines is where most of the learning actually happens — the book points, it does not substitute.
2. **Run the structural notebooks.** The ones tagged 🟢 need no GPU and no downloads. They exist because "I understood the explanation" and "I can reproduce the computation" are different states, and only the second one survives contact with your own project.

Chapters 03, 04 and 08 are load-bearing — the token budget, the vision tower and the fusion point are what "multimodal" actually means here. Chapters 06 and 07 are the ones that will change how you read *any* modern LLM, not just this one.

## Check yourself

1. Which Gemma 4 sizes can hear? Where in the released files is that fact represented?
2. An image, a video and an audio clip enter the model. At which point do the three stop being different kinds of thing?
3. Name the two files you would open to answer "what is new in Gemma 4 relative to Gemma 3", and say which one to read first.
4. Why does this book use a randomly-initialised miniature model for some notebooks and real E2B weights for others?
