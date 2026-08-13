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

## Walkthrough

### 1. A config is a typed dataclass now

Open `configuration_gemma4.py`. If you last read `transformers` source a couple of years ago, the first surprise is that a config is no longer a bag of `**kwargs` assigned in `__init__`:

```python
@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4TextConfig(PreTrainedConfig):
    model_type = "gemma4_text"

    vocab_size: int = 262_144
    hidden_size: int = 2304
    num_hidden_layers: int = 30
    sliding_window: int = 512
    layer_types: list[str] | None = None
    ...
```

`@strict` (from `huggingface_hub.dataclasses`) makes it a validated dataclass: every field is annotated, every field has a default, and typos raise instead of silently becoming an unused attribute. Two consequences worth internalising:

- **The class body is the complete list of architectural knobs.** There is nowhere else for a hyperparameter to hide. Reading these four class bodies top to bottom is a legitimate way to learn the architecture.
- **Derived values live in `__post_init__`, not `__init__`.** That is where a config stops being a record and starts being a small program — see §3.

`model_type` is the registry key. `"gemma4_text"`, `"gemma4_vision"`, `"gemma4_audio"` and `"gemma4"` are each mapped to a model class in the `Auto*` tables, which is how §4 works.

### 2. Nesting, and how a modality disappears

`Gemma4Config` declares its children explicitly:

```python
class Gemma4Config(PreTrainedConfig):
    model_type = "gemma4"
    sub_configs = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4VisionConfig,
        "audio_config": Gemma4AudioConfig,
    }
```

`sub_configs` is what lets `from_pretrained` rebuild the nested objects out of nested JSON dicts, and what `__post_init__` uses to coerce whatever you passed:

```python
if self.text_config is None:
    self.text_config = Gemma4TextConfig()          # text is mandatory
elif isinstance(self.text_config, dict):
    self.text_config = Gemma4TextConfig(**self.text_config)

if self.vision_config is None:
    logger.info("vision_config is None. Gemma4Model.vision_tower will not be initialized.")
```

Note the asymmetry. A missing `text_config` is filled in with defaults, because a Gemma 4 without a language model is not a Gemma 4. A missing `vision_config` or `audio_config` is *honoured*: the tower is simply never built. This is not hypothetical — it is exactly how the released checkpoints differ:

```json
// google/gemma-4-31B-it/config.json
"audio_config": null,
```

That one `null` is the entire mechanism behind "audio is E2B/E4B only". There is no `if size == "31B"` anywhere in the modelling code.

The special-token IDs live on the **top-level** config, not on `text_config`:

```python
boi_token_id:    int | None = 255_999   # begin-of-image
eoi_token_id:    int | None = 258_882   # end-of-image
image_token_id:  int | None = 258_880
video_token_id:  int | None = 258_884
boa_token_id:    int | None = 256_000   # begin-of-audio
eoa_token_index: int | None = 258_883   # end-of-audio  (note: _index, not _id)
audio_token_id:  int | None = 258_881
```

They belong there because they are the **contract between the processor and the model**: chapter 02 writes these IDs into `input_ids`, chapter 08 finds them again and overwrites their embeddings. Neither the tokenizer alone nor the decoder alone owns that agreement. (The stray `eoa_token_index` naming is a real inconsistency in the codebase, not a typo in this book — the checkpoints ship both `eoa_token_id` and `eoa_token_index`.)

### 3. `__post_init__`: where the config computes

`Gemma4TextConfig.__post_init__` is short and does three things you will meet again in chapters 06 and 08.

**It generates the attention schedule.** If you do not supply `layer_types`, it is derived:

```python
if self.layer_types is None:
    sliding_window_pattern = 6  # by default 5:1
    self.layer_types = [
        "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
        for i in range(self.num_hidden_layers)
    ]

if self.layer_types and (last_layer_type := self.layer_types[-1]) != "full_attention":
    logger.warning(...)
    self.layer_types[-1] = "full_attention"
```

Five sliding layers, then one global, repeating — and the **last layer is forced global no matter what**. That override is a good thing to hold onto: whatever else happens, the final layer sees the whole sequence before the logits are computed.

**It picks a different RoPE per layer type.** This is the detail most people miss:

```python
default_rope_params = {
    "sliding_attention": {"rope_type": "default",      "rope_theta": 10_000.0},
    "full_attention":    {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                          "rope_theta": 1_000_000.0},
}
```

A layer that only ever sees 512 neighbouring tokens does not need a rotation frequency stretched for 128K context — so it keeps the classic θ=10,000. The global layers, which do see the whole 128K, get θ=1,000,000 *and* rotate only a quarter of each head's dimensions. Two attention regimes, two position-encoding regimes. Chapter 06 unpacks why.

**It has a side effect you would not predict.** Turning on fully-bidirectional attention silently rewrites the window:

```python
if self.use_bidirectional_attention == "all":
    self.is_causal = False
    self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds
```

A causal window of 512 looks backwards only; a bidirectional window of 512 would look 512 in *each* direction, i.e. twice the span. Halving keeps the total comparable. This is the kind of thing that is invisible in a paper and obvious in three lines of source — which is the whole argument for reading it.

### 4. From config to modules

`Gemma4Model.__init__` in `modeling_gemma4.py` is the payoff. It is almost entirely dispatch:

```python
self.vision_tower = AutoModel.from_config(config.vision_config) if config.vision_config is not None else None
self.language_model = AutoModel.from_config(config=config.text_config)
self.audio_tower  = AutoModel.from_config(config.audio_config)  if config.audio_config  is not None else None
self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config) if config.vision_config is not None else None
self.embed_audio  = Gemma4MultimodalEmbedder(config.audio_config,  config.text_config) if config.audio_config  is not None else None
```

`AutoModel.from_config` does not know anything about Gemma 4 specifically. It reads `config.model_type` — `"gemma4_vision"` — looks it up in the auto registry, finds `Gemma4VisionModel`, and instantiates it. That indirection is why you can swap a sub-config for a different architecture's config and have it work, and it is why every model in the library exposes `model_type`.

Also note there are **five** parameter groups here, not three. The two `Gemma4MultimodalEmbedder`s are separate from the towers they serve — the projection into text space is its own module, and in chapter 10 it is one of the things you can choose to unfreeze on its own.

The config also declares how the model shards, which is worth knowing exists even if you never touch it:

```python
base_model_tp_plan = {"layers.*.self_attn.q_proj": "colwise", ...}
base_model_ep_plan = {"layers.*.router": "ep_router",
                      "layers.*.experts.gate_up_proj": "grouped_gemm", ...}
base_model_pp_plan = {"embed_tokens": (["input_ids"], ["inputs_embeds"]), ...}
```

Tensor, expert and pipeline parallelism are declared as *data* on the config rather than written into the modelling code. The comment on `base_model_ep_plan` is a nice piece of engineering honesty: *"do not tp in attention (num_global_key_value_heads=2 too small to partition)"*.

### 5. What the four released sizes actually say

Reading one config teaches you the fields. Reading four side by side teaches you the design. Pull them yourself — they are small JSON files, no weights involved:

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
print(cfg.text_config.num_kv_shared_layers, cfg.text_config.hidden_size_per_layer_input)
```

| `text_config` field | E2B | 31B | 26B-A4B |
|---|---|---|---|
| `hidden_size` | 1536 | 5376 | 2816 |
| `num_hidden_layers` | 35 | 60 | 30 |
| `num_attention_heads` | 8 | 32 | 16 |
| `num_key_value_heads` | **1** (MQA) | 16 | 8 |
| `head_dim` / `global_head_dim` | 256 / 512 | 256 / 512 | 256 / 512 |
| `num_global_key_value_heads` | — | 4 | 2 |
| `sliding_window` | 512 | 1024 | 1024 |
| full-attention layers | every 5th | every 6th | every 6th |
| `num_kv_shared_layers` | **20 of 35** | 0 | 0 |
| `attention_k_eq_v` | `False` | **`True`** | **`True`** |
| `hidden_size_per_layer_input` (PLE) | **256** | **0** | **0** |
| `use_double_wide_mlp` | `True` | `False` | `False` |
| `use_bidirectional_attention` | **`None`** | **`"vision"`** | **`"vision"`** |
| `enable_moe_block` | `False` | `False` | **`True`** (128 experts, top-8, `moe_intermediate_size=704`) |
| `max_position_embeddings` | 131072 | 262144 | 262144 |
| `final_logit_softcapping` | 30.0 | 30.0 | 30.0 |
| vision tower | 768 wide, 16 layers | 1152 wide, 27 layers | 1152 wide, 27 layers |
| audio tower | 1024 wide, 12 layers | `null` | `null` |

Four things in that table are worth stopping on, because each contradicts a reasonable guess:

1. **PLE is a small-model technique.** `hidden_size_per_layer_input` is 256 on E2B and **0 on both large models** — the per-layer embedding path is switched off entirely up there. PLE buys capacity where parameters are scarce; at 31B you would rather just have more hidden size. Chapter 07 revisits this.
2. **Bidirectional vision attention is a large-model technique**, and it goes the other way: E2B leaves `use_bidirectional_attention` unset (everything causal, including image tokens), while 31B and 26B-A4B set `"vision"`. Chapter 08 shows both masks.
3. **E2B is extreme about KV cost** — a single KV head (MQA, not GQA) *and* 20 of its 35 layers holding no KV projections of their own. The large models pay for 16/8 KV heads but claw it back with `attention_k_eq_v`, reusing the key projection as the value projection. Two different answers to the same bill; chapter 06.
4. **The vision tower is not one tower.** E2B's is 768-wide/16-layer with clipped linears; the large models' is 1152-wide/27-layer with `standardize=True`. The *preprocessing* is identical across sizes (chapter 03), the encoder is not.

### 6. `modular_gemma4.py`, and how to read this codebase

There are two modelling files and only one of them is real code you should edit:

- **`modular_gemma4.py`** is the source. It is short-ish, and it works by *inheritance*: `from ..gemma3.modeling_gemma3 import Gemma3Attention, Gemma3DecoderLayer, Gemma3MLP, Gemma3RotaryEmbedding, ...`, then defines Gemma 4's classes as subclasses that override only what changed.
- **`modeling_gemma4.py`** is generated from it by the library's `modular` tooling: every inherited method is flattened in, so the file is self-contained and you never have to chase a base class across model directories.

The practical reading rule:

> **Read `modeling_gemma4.py` when you want to know what the model does.** It is the ground truth and it is complete.
> **Read `modular_gemma4.py` when you want to know what is *new*.** Its import list and its overrides are a precise diff against Gemma 3.

That import list is itself informative. `Gemma4MLP` and the decoder-layer skeleton come straight from Gemma 3; the vision tower, the audio tower, PLE, the MoE block and the mask construction do not. The genuinely new surface area of Gemma 4 is smaller than 126KB of generated code suggests.

## Design space

Every multimodal library has to solve "how do I describe a model made of several models". Three answers in circulation:

- **Nested configs with a registry** (Gemma 4, and `transformers` generally). Sub-configs carry their own `model_type`; `AutoModel.from_config` does the dispatch. Verbose JSON, but each tower is independently describable, loadable and swappable — and `null` cleanly means "this model has no such tower".
- **One flat config** (early LLaVA forks, many research repos). Simple until two towers want a field called `hidden_size`, at which point you get `vision_hidden_size`, `mm_hidden_size`, and an argument about naming.
- **Config-as-code** (Fairseq/Detectron-style builders, and `timm`'s creation functions). Maximum flexibility, but the architecture is no longer a serialisable artifact — you cannot diff two models by diffing two JSON files, which is exactly what §5 just did.

The nested-registry approach is why the table in §5 was possible at all, and it is a good argument for the design.

## Check yourself

1. You have a `Gemma4Config` with `audio_config=None`. Which attributes of `Gemma4Model` are `None`, and what happens if you pass `input_features` to `forward` anyway?
2. `num_hidden_layers=12` and no `layer_types`. Which layer indices end up as `full_attention`? (Careful — there are two rules.)
3. Why does the top-level config own `image_token_id` rather than `text_config`?
4. E2B sets `num_key_value_heads=1`. What is that called, and what does it do to KV-cache size relative to the 31B's 16?
5. You want to fine-tune only the projection from vision into text space. Which of the five submodules built in `Gemma4Model.__init__` do you unfreeze?

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_config_to_model.ipynb` | Build a mini-Gemma 4 from four hand-written configs, forward a batch of random tokens through it, break the parameter count down by submodule, and round-trip it through `save_pretrained` / `from_pretrained` | 🟢 CPU, zero downloads |
