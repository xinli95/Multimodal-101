# 01 · Config — How a `transformers` Model Is Assembled

**Position in the pipeline**: before it. A config is the blueprint every other chapter builds from.

A `transformers` model is a pure function of its config. `Gemma4Config` is a container holding three sub-configs — text, vision, audio — and `Gemma4Model.__init__` reads them to decide which towers to instantiate at all. Read the config and you have read the architecture, before touching a single weight.

## What you will learn

1. How `Gemma4Config` nests `Gemma4TextConfig` / `Gemma4VisionConfig` / `Gemma4AudioConfig`, and how a `None` sub-config makes an entire modality disappear
2. How the `Auto*` classes turn a config into a model (`AutoModel.from_config`) and a directory into a model (`from_pretrained`)
3. What the special-token IDs in the config are for — the hinge between chapter 02 (text) and chapter 08 (fusion)
4. What `modular_gemma4.py` is, why `modeling_gemma4.py` is generated from it, and what that means when you read the source
5. Build a miniature Gemma 4 from scratch with random weights, run a forward pass on a CPU, and account for every parameter

## Source map

| File | Symbol | Role |
|---|---|---|
| `configuration_gemma4.py` | `Gemma4Config` | Top level: `text_config`, `vision_config`, `audio_config` + the special-token IDs |
| | `Gemma4TextConfig` | The decoder: 262144 vocab, `layer_types`, `sliding_window`, PLE and MoE fields |
| | `Gemma4VisionConfig` | The vision tower: `patch_size=16`, `pooling_kernel_size`, `position_embedding_size=10240` |
| | `Gemma4AudioConfig` | The audio tower: `attention_chunk_size`, `subsampling_conv_channels`, `output_proj_dims` |
| `modeling_gemma4.py` | `Gemma4Model.__init__` | Where the config becomes modules: `vision_tower`, `audio_tower`, `language_model`, `embed_vision`, `embed_audio` |
| | `Gemma4PreTrainedModel` | The base class: weight init, embedding resizing, the `_can_*` capability flags |

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_config_to_model.ipynb` | Build a mini-Gemma 4 from four hand-written configs, forward a batch of random tokens through it, break the parameter count down by submodule, and round-trip it through `save_pretrained` / `from_pretrained` | 🟢 CPU, zero downloads |
