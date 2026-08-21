# 11 · OCR as a Multimodal System — GLM-OCR

**Position in the book**: the transfer test. Chapters 00–10 taught one way to take a multimodal model apart; this chapter applies the same method to a specialist model whose output has to be not merely fluent, but *exact and structurally valid*.

GLM-OCR is a useful capstone because its 0.9B checkpoint is small enough to inspect end to end, while the product around it is larger than the checkpoint: a document parser first detects regions, sends text, table, and formula crops through the model in parallel, and then reconstructs reading order. The distinction between **base model** and **system** is the chapter's main argument.

```text
base model
image ──► CogViT ──► 2×2 downsample + merger ──► GLM decoder ──► text / HTML / LaTeX / JSON

document system
PDF ──► page images ──► PP-DocLayout-V3 ──► region crops ──► GLM-OCR × N ──► merge ──► Markdown + layout JSON
```

The model checkpoint is released under MIT; the SDK and PP-DocLayout-V3 integration are Apache-2.0. The technical report is CC BY 4.0.

## What you will learn

1. Why "OCR" now covers five different output contracts: transcription, formula recognition, table reconstruction, document parsing, and key-information extraction (KIE)
2. How to reconstruct the released checkpoint from `config.json`: a 24-layer, 1024-wide CogViT feeding a 16-layer, 1536-wide GLM decoder
3. Where visual-token compression actually happens: a strided 2×2 convolution followed by a gated merger, not a generic one-line projector
4. How task prompts turn one conditional generator into text, table, formula, and schema-constrained extraction models
5. What Multi-Token Prediction (MTP) changes — and why `transformers.generate()` is an autoregressive baseline, not the official MTP serving path
6. How the SDK composes layout detection, parallel region OCR, reading-order recovery, and output formatting
7. How to evaluate OCR by output contract instead of reporting one misleading aggregate number

## Source map

This chapter targets the released `zai-org/GLM-OCR` checkpoint and the implementation in `transformers` 5.14.x, the same version family as Part I.

| Source | Symbol / file | Role |
|---|---|---|
| Hugging Face checkpoint | [`config.json`](https://huggingface.co/zai-org/GLM-OCR/blob/main/config.json), [`preprocessor_config.json`](https://huggingface.co/zai-org/GLM-OCR/blob/main/preprocessor_config.json) | The facts about the released weights; prefer these over class defaults |
| `configuration_glm_ocr.py` | [`GlmOcrConfig`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/configuration_glm_ocr.py#L132), [`GlmOcrVisionConfig`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/configuration_glm_ocr.py#L30), [`GlmOcrTextConfig`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/configuration_glm_ocr.py#L72) | Nested architecture configs |
| `modeling_glm_ocr.py` | [`GlmOcrVisionPatchEmbed`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L517), [`GlmOcrVisionBlock`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L469) | Patchify and CogViT encoder |
| | [`GlmOcrVisionPatchMerger`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L500), [`GlmOcrVisionModel`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L537) | 2×2 spatial downsample and gated projection into text width |
| | [`GlmOcrModel.get_image_features`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L1008), [`get_placeholder_mask`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L1028) | Visual feature extraction and placeholder replacement |
| | [`GlmOcrTextModel`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L708), [`GlmOcrForConditionalGeneration`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L1202) | Decoder and autoregressive LM head |
| `processing_glm46v.py` / `image_processing_glm46v.py` | [`Glm46VProcessor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm46v/processing_glm46v.py#L43), [`Glm46VImageProcessor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm46v/image_processing_glm46v.py#L87) | Shared GLM-V chat and image preprocessing used by this checkpoint |
| GLM-OCR SDK | [`PageLoader`](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/dataloader/page_loader.py#L40), [`PPDocLayoutDetector`](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/layout/layout_detector.py#L27), [`OCRClient`](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/ocr_client.py#L24), [`ResultFormatter`](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/postprocess/result_formatter.py#L40) | The production document pipeline around the base model |

Two source-reading rules matter here.

First, **read the published checkpoint's JSON, not the Python defaults**. In `transformers` 5.14, `GlmOcrTextConfig.hidden_size` defaults to 1024; the checkpoint says 1536. The JSON describes the weights you downloaded.

Second, **do not infer the serving algorithm from the forward class name**. The `transformers` implementation exposes a conventional `lm_head`; the official vLLM and SGLang launch commands separately enable MTP/speculative decoding.

## 1. OCR is a family of output contracts

Classical OCR is often drawn as detection followed by recognition: find a text box, then emit characters. A modern document model may instead be asked for any of these:

| Task | Input | Output contract | A useful metric |
|---|---|---|---|
| Text recognition | crop or page | plain text | normalized edit distance / CER |
| Formula recognition | formula crop | LaTeX | CDM + parse validity |
| Table recognition | table crop | HTML or Markdown | TEDS + tag closure |
| Document parsing | one or more pages | ordered Markdown + layout metadata | element scores + reading-order edit |
| KIE | page + requested schema | JSON matching that schema | field-level F1 + JSON validity |

The word *recognition* is too small for the last three. A table model must reconstruct two-dimensional relationships that are not present in a plain text stream. A document parser must decide reading order. A KIE model must select values, associate them with fields, and obey a schema. These are all **conditional structured generation** tasks.

That framing explains several GLM-OCR choices:

- an autoregressive language decoder can emit text, HTML, LaTeX, Markdown, or JSON without separate task heads;
- task prompts select an output contract;
- structural validity can become a training reward;
- long deterministic outputs make token-by-token decoding an obvious optimisation target;
- a small model benefits from an explicit layout detector that breaks a page into easier subproblems.

## 2. Reconstruct the checkpoint from JSON

The released checkpoint is 2.66GB in BF16 and approximately 0.9B parameters. Its top-level config contains two sub-configs:

| Component | Released value | Consequence |
|---|---:|---|
| Vision depth / width | 24 / 1024 | roughly 0.4B CogViT encoder |
| Vision heads | 16 | 64 dimensions per head |
| Patch / temporal patch | 14 / 2 | a Conv3d consumes two identical image frames per patch |
| Spatial merge | 2 | every 2×2 patch group becomes one LLM visual token |
| Vision output width | 1536 | already matches the text decoder width |
| Text depth / width | 16 / 1536 | roughly 0.5B GLM decoder |
| Query / KV heads | 16 / 8 | grouped-query attention, two query heads per KV head |
| MLP width | 4608 | 3× the model width |
| Vocabulary | 59,392 | text and structural tokens share one LM head |
| Maximum positions | 131,072 | long enough for structured page outputs; not a promise that every page is useful at that length |

This is recognisably the same macro-pattern as Gemma 4:

```text
modality-specific tower
        ↓
features in the language-model width
        ↓
placeholder replacement
        ↓
decoder-only language model
```

But the compression policy is different. Gemma 4 chooses a fixed soft-token budget before the tower. GLM-OCR's shared GLM-V processor uses a pixel range, a 14×14 patch grid, and a 2×2 merge; larger document images therefore spend more visual tokens. For OCR, that is usually the right bias: small glyphs need pixels.

## 3. Pixels to visual prefix tokens

### 3.1 Patch embedding

`GlmOcrVisionPatchEmbed` is a Conv3d:

```python
kernel_size = [temporal_patch_size, patch_size, patch_size]  # [2, 14, 14]
self.proj = nn.Conv3d(3, 1024, kernel_size=kernel_size, stride=kernel_size)
```

For a still image, the processor supplies the temporal dimension expected by the shared image/video code path. The output is flattened to `[num_patches, 1024]` and passed through 24 vision blocks using two-dimensional rotary position information.

The visual attention is bidirectional within each packed image: every patch may use context from every other patch on that image. Packed cumulative sequence lengths keep two images in a batch from attending across the boundary.

### 3.2 The connector is inside the tower

After the last block and RMSNorm, the source performs the decisive compression:

```python
hidden_states = hidden_states.view(-1, 2, 2, 1024)
hidden_states = hidden_states.permute(0, 3, 1, 2)
hidden_states = self.downsample(hidden_states)  # Conv2d: 1024 → 1536, kernel=2, stride=2
hidden_states = hidden_states.view(-1, 1536)
merged = self.merger(hidden_states)
```

One 2×2 group becomes one 1536-wide token. The merger is not just a linear layer: projection → LayerNorm → GELU → gated MLP. The report calls this a lightweight cross-modal connector; in the `transformers` implementation it lives as the end of `GlmOcrVisionModel`.

The exact visual-token count for a processed grid is:

$$
N_{visual} = T \times \frac{H}{m} \times \frac{W}{m}, \qquad m=2.
$$

That equation is both compute accounting and an OCR-quality diagnostic. Halving both processed image dimensions quarters the visual sequence, but may erase the punctuation, decimal points, subscripts, and table rules that carry the document's meaning.

### 3.3 Fusion

`GlmOcrModel.get_image_features` returns the merger's `pooler_output` and splits a packed batch back into one sequence per image. `get_placeholder_mask` then verifies a strict invariant:

```text
number of <image> placeholders == number of visual features
```

The features replace the placeholder embeddings; the combined sequence enters `GlmOcrTextModel`. No cross-attention module is added. Chapter 08's lesson transfers exactly: after replacement, the decoder sees one embedding sequence and modality is carried by token type and multidimensional positions.

## 4. One generator, four task prompts

The three predefined recognition prompts are deliberately plain:

```text
Text Recognition:
Table Recognition:
Formula Recognition:
```

The input image comes before the prompt in the chat template. KIE is more explicit: the prompt should state a strict JSON schema, including nested fields and whether missing values are `null`, empty, or omitted.

This is not merely prompt engineering. During supervised fine-tuning the model sees those task strings paired with different target languages:

- natural-language characters for text;
- HTML/Markdown structure for tables;
- LaTeX for formulas;
- user-defined JSON for KIE.

The prompt is a low-cost task router over shared parameters. It also creates a failure mode: a vague KIE schema makes several outputs equally plausible, so even perfect visual perception cannot make the contract deterministic.

## 5. MTP: separate training, forward, and serving

OCR output is unusually friendly to multi-token prediction. It is long, mostly deterministic, locally constrained, and rich in repeated syntax. Once a decoder has emitted `<tr><td>`, the next several structural possibilities are much narrower than in open-ended prose.

The technical report says GLM-OCR was trained to predict ten future tokens per step. At inference it reports an average 5.2 generated tokens per decoding step and approximately 50% higher throughput. Shared parameters across draft heads control the memory overhead.

Three numbers in the released stack must not be conflated:

| Place | Number | Meaning |
|---|---:|---|
| Technical report | 10 | training prediction horizon |
| Checkpoint `text_config` | `num_nextn_predict_layers = 1` | metadata about the released next-token-prediction component |
| Official vLLM example | `num_speculative_tokens = 3` | serving-time proposal depth chosen for that command |

Most importantly, `transformers` 5.14's `GlmOcrForConditionalGeneration` contains the base model plus one conventional `lm_head`. Its normal `generate()` path is the **autoregressive correctness baseline**. To benchmark the advertised MTP path, launch a serving engine explicitly:

```bash
vllm serve zai-org/GLM-OCR \
  --port 8080 \
  --served-model-name glm-ocr \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The scientific comparison is not only tokens per second. Record:

- output tokens per second and pages per minute;
- peak memory and time to first token;
- exact output agreement with the autoregressive baseline;
- JSON parse rate, HTML tag closure, and LaTeX parse validity;
- acceptance length distribution and fallback frequency.

Speculation is a serving algorithm: proposed tokens are verified by the target model. A speed result without the acceptance rate hides the mechanism.

## 6. Training recipe: rewards can be executable

The report describes four stages:

| Stage | Data / objective | What it buys |
|---|---|---|
| 1 · Vision encoder | image-text, grounding/retrieval; MIM + CLIP; distillation | high-resolution visual features |
| 2.1 · VL pretraining | image-text, document parsing, grounding, VQA | align CogViT and GLM |
| 2.2 · Pretraining with MTP | document parsing, grounding, VQA | teach future-token prediction before specialisation |
| 3 · SFT with MTP | text, formula, table, KIE | learn the output contracts |
| 4 · GRPO | rollouts scored by task-aware validators | accuracy and structural reliability |

OCR makes reinforcement learning unusually concrete because much of the reward is executable:

| Task | Accuracy signal | Structural signal |
|---|---|---|
| Text | normalized edit distance | repetition penalty |
| Formula | CDM | structural validity |
| Table | TEDS | tag closure and structural parsing |
| KIE | field-level F1 | JSON parse, missing/duplicate field penalties |

Compare that with a general assistant's preference reward. Here the evaluator need not decide whether an answer is eloquent; it can parse it.

For domain adaptation, the official LLaMA-Factory recipe freezes the vision tower and multimodal projector for full SFT, or applies rank-8 LoRA to linear layers. The guide estimates LoRA can fit in 8GB VRAM and full fine-tuning needs roughly 24GB. The first intervention should still be data and resolution analysis: fine-tuning cannot recover characters destroyed before the vision tower.

## 7. Base model versus document system

The base model accepts an image plus a task prompt. The SDK adds four composable modules:

| Module | Responsibility | Failure surface |
|---|---|---|
| `PageLoader` | render PDF pages, resize/encode images | DPI too low; pages omitted or rotated |
| `PPDocLayoutDetector` | boxes/polygons and region labels | missed/merged/split regions; wrong class |
| `OCRClient` | send crops with text/table/formula prompts, in parallel | overloaded server; wrong prompt; crop loses context |
| `ResultFormatter` | reading order, merge, Markdown and JSON | column order, duplicated text, broken joins |

Document parsing uses the layout-first route. KIE can instead send the whole page directly to the base model with a schema prompt. These are different systems because the task requires different context:

- a table crop benefits from high effective resolution and a narrow output contract;
- KIE may need to connect a field name in the top-left with a value in the bottom-right, so cropping can remove the relationship.

The two-stage design improves a small model's stability and enables parallelism, but it introduces error propagation. If the detector misses a formula, no recogniser can recover it. If it labels a table as text, the wrong output language is requested. If reading order is wrong, every region may be transcribed perfectly and the document is still unusable.

That is why a production evaluation needs stage-level observability: save page render, boxes, labels, crops, per-region raw outputs, merge order, and final Markdown. A single final score cannot tell you where to fix the system.

## 8. Evaluation by contract

The report's headline 94.62 on OmniDocBench v1.5 is an **author-reported aggregate**. It is useful context, not a substitute for reproducing the slice that matches your documents. The same report shows that GLM-OCR's overall lead is driven especially by table structure; it does not win every text and formula submetric.

A minimal evaluation set should be small enough to inspect and diverse enough to break assumptions:

| Slice | What to vary | What to record |
|---|---|---|
| Printed text | font size, DPI, blur, rotation | normalized edit distance; digit recall |
| Tables | ruled/unruled, merged cells, dense numbers | TEDS; tag validity; cell alignment |
| Formulas | inline/display, matrices, nested scripts | CDM; LaTeX parse rate |
| Layout | one/two columns, footnotes, sidebars | region recall; reading-order edit |
| KIE | clear/ambiguous schemas, missing fields | field F1; JSON validity; invented fields |
| Language | represented and long-tail scripts | per-language CER, never only the mean |

Digits deserve a separate audit. A hallucinated adjective is annoying; a plausible but wrong invoice total is a financial error. Extract every number from prediction and reference, then report precision and recall alongside whole-text edit distance.

### An ablation ladder

Run the same pages through four conditions:

1. full page → base model;
2. oracle region crops → base model;
3. detected crops → base model, sequential;
4. detected crops → base model, parallel → formatter.

The gaps isolate different causes:

- 1 → 2 measures the value of decomposition and effective resolution;
- 2 → 3 measures layout-detection error;
- 3 → 4 measures concurrency stability and merge/postprocess behaviour.

Without this ladder, "the OCR model failed" is not a diagnosis.

## Design space

| System | Main idea | Strength | Cost / limitation |
|---|---|---|---|
| Classical OCR pipeline | detector + recogniser + rules | controllable, cheap, character-level confidence | brittle on tables/formulas and novel layouts |
| General VLM (Gemma 4 / Qwen3-VL) | one broad model, prompted for OCR | composes OCR with reasoning and other modalities | spends capacity on many tasks; may lack strict structure |
| End-to-end OCR VLM | whole page → structured sequence | simple interface; global context | long decoding; hallucination and reading-order pressure |
| **GLM-OCR SDK** | layout detector + specialist generator per region | effective resolution, parallelism, task-specific prompts | detector error propagates; merge logic is part of accuracy |
| Visual document retrieval | page image → retrieval embeddings, no parse | preserves layout; avoids lossy transcription before retrieval | cannot directly supply clean text or structured fields |

GLM-OCR lands in the productive middle. It uses a generative VLM where structure matters, while retaining explicit decomposition where a compact model benefits from it. "Old pipeline" and "new end to end" are not the only choices.

## Check yourself

1. A processed grid is `[T=1, H=80, W=60]`. How many visual tokens enter the LLM after a 2×2 merge?
2. Why is `vision_config.out_hidden_size=1536` coupled to `text_config.hidden_size=1536` in the released checkpoint?
3. Which outputs require global page context, and which benefit most from region crops?
4. Why is a successful `transformers.generate()` run not evidence that MTP was enabled?
5. The table text is perfect but TEDS is poor. Which part of the output contract failed?
6. The final Markdown omits a formula. Name the artifacts you inspect, in order, to locate the responsible stage.
7. Design a KIE schema for a receipt that makes missing discounts unambiguous.
8. When would a general VLM be the better engineering choice than GLM-OCR?

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| [`01_glm_ocr_anatomy.ipynb`](notebooks/01_glm_ocr_anatomy.ipynb) | Read the released JSON, build a miniature random GLM-OCR, hook patch/embed/downsample/merger shapes, derive the visual-token count, and verify the 2×2 downsample against `torch.nn.functional.conv2d` | 🟢 CPU; no weights |
| [`02_tasks_and_mtp.ipynb`](notebooks/02_tasks_and_mtp.ipynb) | Run text/table/formula/KIE with the real 0.9B checkpoint, validate output contracts, then benchmark separate AR and MTP server endpoints without confusing the two paths | 🟡 8–12GB+ or hosted servers |
| [`03_base_vs_pipeline.ipynb`](notebooks/03_base_vs_pipeline.ipynb) | Compare whole-page base-model OCR with the layout-aware SDK, preserve stage artifacts, inject crop/layout failures, and score digit recall plus structural validity | 🟡 self-hosted GPU or MaaS API |

## Primary sources

- [GLM-OCR checkpoint and model card](https://huggingface.co/zai-org/GLM-OCR)
- [GLM-OCR technical report](https://arxiv.org/abs/2603.10910)
- [Official GLM-OCR SDK](https://github.com/zai-org/GLM-OCR)
- [`transformers` GLM-OCR implementation, v5.14.1](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py)
- [Official fine-tuning guide](https://github.com/zai-org/GLM-OCR/tree/main/examples/finetune)
