# 10 · Fine-Tuning — Changing the Weights

**Position in the pipeline**: all of it, backwards.

The last chapter of Part I closes the loop. Once you know which module does what, "how do I fine-tune this" stops being a recipe you copy and becomes a decision you make: which of the four parameter groups — vision tower, audio tower, multimodal embedders, language model — should move for *your* task, and which should stay frozen.

## What you will learn

1. The four parameter groups of a Gemma 4 checkpoint, their sizes, and what training each one actually changes
2. Why the field converged on "freeze the towers, adapt the LLM" for most downstream tasks — and the cases where that is wrong
3. LoRA with PEFT: where to put the adapters, how rank and target modules interact with GQA and with the MoE path
4. Writing a multimodal data collator: the part every tutorial skips, where the placeholder counts from chapter 02 and the soft-token counts from chapters 03–05 must agree or nothing works
5. Label masking: training on the response only, and why a mistake here silently produces a model that predicts prompts
6. The memory arithmetic — weights, gradients, optimiser state, activations — worked out before you launch, not after it OOMs

## Source map

| Symbol | Role |
|---|---|
| `Gemma4PreTrainedModel` | `_init_weights`, `resize_token_embeddings`, `_resize_per_layer_embeddings` (adding tokens touches the PLE table too — chapter 07) |
| `Gemma4Model.vision_tower` / `.audio_tower` / `.embed_vision` / `.embed_audio` / `.language_model` | The four groups you freeze or unfreeze |
| `Trainer`, `TrainingArguments` | The `transformers` training loop |
| `peft.LoraConfig`, `get_peft_model` | The adapters |

## Walkthrough

### 1. The five groups, and what each one moves

Chapter 01 §4 built them; here is what training each actually changes:

| Group | Attribute | Training it changes… | Typical verdict |
|---|---|---|---|
| Vision tower | `model.vision_tower` | how pixels are *perceived* | freeze — needs image-scale data to move safely |
| Audio tower | `model.audio_tower` | how sound is perceived | freeze — same, more so |
| Vision embedder | `model.embed_vision` | how visual features *land* in text space | sometimes unfreeze |
| Audio embedder | `model.embed_audio` | ditto for audio | sometimes unfreeze |
| Language model | `model.language_model` | what the model *does* with what it perceives | the usual target |

The default recipe — freeze the towers, adapt the language model — works because most downstream tasks are not "see differently", they are "respond differently to what you see". Reformatting outputs, following a domain's conventions, answering in a house style: all language-side.

The exceptions are worth naming, because they are the cases where the default silently underperforms:

- **A genuinely novel visual domain** (medical imaging, satellite, industrial inspection). The tower may have no useful features for it. Unfreezing the tower — or at least `embed_vision` — is defensible, with a much lower learning rate than the LLM's.
- **Fine detail beyond the token budget.** If the model cannot read your small print, that is chapter 03's problem, not a training problem. Raise `max_soft_tokens` before you consider touching weights.
- **Audio on E2B/E4B.** The audio tower is the most numerically delicate part of the model (chapter 05 — clipped linears, gradient clipping, float32 attention). Unfreeze it last and watch for instability.

Freezing is unglamorous:

```python
for p in model.model.vision_tower.parameters(): p.requires_grad = False
for p in model.model.audio_tower.parameters():  p.requires_grad = False
```

### 2. LoRA target modules, with Gemma 4's complications

The usual PEFT starting point:

```python
from peft import LoraConfig, get_peft_model

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
```

Three Gemma-4-specific things will surprise you if you do not expect them.

**Many layers have no `k_proj` or `v_proj`.** On E2B, layers 15–34 are KV-shared (chapter 06 §4) and simply do not own those modules. A `target_modules` list matching by name will find them on the first 15 layers only. That is not a bug — you cannot adapt a projection that does not exist — but `print_trainable_parameters()` will show fewer adapters than you predicted, and now you know why.

**Under `attention_k_eq_v` (31B, 26B-A4B) global layers have no `v_proj` either.** Same reasoning.

**MLP width is not uniform.** Chapter 06 §5: on E2B the KV-shared layers carry a double-wide MLP. A rank-16 adapter on a 12288-wide projection is a proportionally *smaller* intervention than the same adapter on a 6144-wide one. If you want uniform adaptation pressure you have to think about it; most people should not bother, but it is worth knowing the layers are not interchangeable.

If you also want the connector to move, add it by name:

```python
target_modules=[..., "embedding_projection"]     # Gemma4MultimodalEmbedder's only Linear
```

That single bias-free linear (chapter 04 §5) is the cheapest possible way to let visual features re-land somewhere more useful, and it is often enough when a frozen tower is *almost* right.

**Adding tokens costs double.** If your task needs new special tokens, `resize_token_embeddings` must also grow `embed_tokens_per_layer` (chapter 07 §5) on the PLE sizes. `Gemma4PreTrainedModel._resize_per_layer_embeddings` handles it — just be aware the new rows are `num_layers × 256` wide and randomly initialised, so they need training.

### 3. The collator, where everything meets

This is the part tutorials skip, and it is where every earlier chapter shows up at once.

```python
def collate(examples):
    texts, images = [], []
    for ex in examples:
        texts.append(processor.apply_chat_template(ex["messages"], tokenize=False,
                                                   add_generation_prompt=False))
        images.append(ex["images"])            # nested: one list per sample

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for tok_id in (config.image_token_id, config.audio_token_id, config.video_token_id,
                   config.boi_token_id, config.eoi_token_id):
        labels[labels == tok_id] = -100
    batch["labels"] = labels
    return batch
```

Four requirements, each traceable to a specific chapter:

1. **`tokenize=False`, then let the processor tokenize.** The template emits one `<|image|>`; only the processor expands it to the right run length once it has seen the image (chapter 02 §2). Tokenizing in the template step gives you one placeholder where the model expects 280.
2. **Nested image lists.** `validate_inputs` compares per-sample counts (chapter 02 §4). A flat list across the batch raises.
3. **Mask every multimodal placeholder out of the labels.** Training the model to predict `<|image|>` tokens is meaningless and actively harmful — and it is silent, because the loss still goes down.
4. **`padding=True` with `padding_side` from the processor.** Left padding for generation; for *training* right padding is conventional and fine, since there is no decode loop. Be deliberate rather than accidental.

### 4. Train on the response only

The single most common mistake in instruction fine-tuning: masking pads but not prompts, so the model spends most of its gradient learning to generate *questions*.

```python
def mask_prompt(labels, input_ids, tokenizer):
    # find the last "<|turn>model\n" and mask everything before it
    start = tokenizer.encode("<|turn>model\n", add_special_tokens=False)
    ...
```

Gemma 4's turn grammar makes this tractable: the model's own content always begins after a `<|turn>model\n` marker (chapter 02 §5). Locate the final one and set everything up to it to `-100`.

Sanity check before you launch anything: decode the positions where `labels != -100` and read them. They should be exactly the assistant's reply and nothing else. This costs thirty seconds and catches most of the errors in this chapter.

### 5. The memory arithmetic

Do this before you launch, not after the OOM. For E2B in bf16:

| Item | Estimate |
|---|---|
| Weights | ~10GB (the checkpoint is 10.25GB) |
| LoRA adapters | tens of MB |
| Gradients | adapters only — tens of MB |
| Optimiser state (AdamW, 2 moments fp32) | ~8× adapter size, still small |
| **Activations** | **the actual variable** |

With LoRA the frozen weights dominate and everything trainable is noise by comparison — which is the whole point. Activations are what you control:

- **Sequence length is dominated by images.** One image is 280 tokens (chapter 03); four images and a paragraph is well over 1,000. Batch size × sequence length is the number that matters, not batch size.
- **`gradient_checkpointing=True`** trades roughly 30% more compute for a large activation saving. The decoder layers are `GradientCheckpointingLayer` subclasses already, so it is one flag.
- **The vision tower runs on every step even when frozen.** No gradients, but the forward pass and its activations are real. If your dataset reuses the same images, precomputing soft tokens is a genuine optimisation — subject to chapter 09 §"design space"'s warning about also precomputing `per_layer_inputs`.

A 2×24GB setup runs E2B LoRA comfortably at short sequence lengths. Full fine-tuning of the language model does not fit; that is a 40GB+ job.

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="out", per_device_train_batch_size=1, gradient_accumulation_steps=8,
    learning_rate=1e-4, num_train_epochs=1, bf16=True,
    gradient_checkpointing=True, logging_steps=10,
    remove_unused_columns=False,          # <- required: Trainer would strip pixel_values
)
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
```

`remove_unused_columns=False` is not optional. `Trainer` inspects `forward`'s signature and drops dataset columns it does not recognise; with a custom collator producing `pixel_values` and `image_position_ids`, the default silently removes them and you train a text-only model on prompts with 280 pad embeddings in them. It will not error.

## Design space

**What to adapt** has converged surprisingly hard across the field:

| Strategy | Cost | When |
|---|---|---|
| Prompting / few-shot | free | always try first |
| LoRA on the LLM | hours, one GPU | the default for behaviour change |
| LoRA + connector | same | frozen tower is close but features land badly |
| Full LLM fine-tune | 40GB+ | large domain shift, lots of data |
| Tower fine-tune | expensive, unstable | genuinely novel visual domain |
| Continued pretraining | industrial | you are not doing this |

**Where the field disagrees** is the multi-stage recipe. The classic LLaVA three-stage sequence — align the connector, then unfreeze everything for multi-task pretraining, then SFT — assumes you are *building* a VLM. Adapting one is a different problem, and the answer is usually "just the last stage". Gemma 4's own instruction-tuned checkpoints have already been through the full sequence; you are editing the end of it, not repeating it.

The one Gemma-4-specific caution: because so much of this model's parameter budget lives in unusual places — a 2.35B-parameter PLE table on E2B, KV-shared upper layers, non-uniform MLP widths — parameter counts from a LoRA config transfer poorly from other models. Print the trainable parameters and look at them rather than assuming your usual settings mean the same thing here.

## Check yourself

1. You LoRA E2B with `target_modules=["q_proj","k_proj","v_proj","o_proj"]` and get far fewer adapters than expected. Why?
2. Your fine-tuned model has started generating questions instead of answers. What did the collator do wrong?
3. Why must multimodal placeholder tokens be set to `-100` in the labels, and why is forgetting it hard to notice?
4. `remove_unused_columns` defaults to `True`. What exactly breaks, and why is there no exception?
5. You add three special tokens to E2B. Which tables grow, and what are the new rows' widths?
6. The vision tower is frozen. Why does it still affect your peak memory, and what could you do about it?
7. The model cannot read small text in your scanned documents. Give the two candidate fixes in the order you would try them.

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_lora_sft.ipynb` | A small image-instruction LoRA SFT on E2B end to end: dataset, collator, label masking, training run, and a before/after comparison on held-out samples | 🟡 24GB VRAM |
