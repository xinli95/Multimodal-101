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

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_lora_sft.ipynb` | A small image-instruction LoRA SFT on E2B end to end: dataset, collator, label masking, training run, and a before/after comparison on held-out samples | 🟡 24GB VRAM |
