# 01 · Config — 一个 `transformers` 模型是怎么被组装出来的

**在数据流中的位置**：在它之前。config 是后面每一章的施工图。

`transformers` 模型是其 config 的纯函数。`Gemma4Config` 是一个容器，装着文本、视觉、音频三个子 config；`Gemma4Model.__init__` 读它们来决定到底要不要实例化某个塔。读懂 config 就等于读懂了架构——一个权重都还没碰。

## 你会学到

1. `Gemma4Config` 如何嵌套 `Gemma4TextConfig` / `Gemma4VisionConfig` / `Gemma4AudioConfig`，以及一个 `None` 子 config 如何让整个模态凭空消失
2. `Auto*` 类如何把 config 变成模型（`AutoModel.from_config`）、把目录变成模型（`from_pretrained`）
3. config 里那些特殊 token id 是干什么的——02 章（文本）与 08 章（融合）之间的铰链
4. `modular_gemma4.py` 是什么，为什么 `modeling_gemma4.py` 是从它生成的，这对读源码意味着什么
5. 从零搭一个随机权重的迷你 Gemma 4，在 CPU 上跑通前向，并把每一个参数算清楚

## 源码地图

| 文件 | 符号 | 作用 |
|---|---|---|
| `configuration_gemma4.py` | `Gemma4Config` | 顶层：三个子 config + 特殊 token id |
| | `Gemma4TextConfig` | 解码器：262144 词表、`layer_types`、`sliding_window`、PLE 与 MoE 字段 |
| | `Gemma4VisionConfig` | 视觉塔：`patch_size=16`、`pooling_kernel_size`、`position_embedding_size=10240` |
| | `Gemma4AudioConfig` | 音频塔：`attention_chunk_size`、`subsampling_conv_channels`、`output_proj_dims` |
| `modeling_gemma4.py` | `Gemma4Model.__init__` | config 变成模块的地方 |
| | `Gemma4PreTrainedModel` | 基类：权重初始化、词表缩放、能力标志位 |

## 源码走读

### 1. config 现在是带类型的 dataclass

打开 `configuration_gemma4.py`。如果你上次读 `transformers` 源码还是两年前，第一个意外是：config 已经不是在 `__init__` 里赋一堆 `**kwargs` 的口袋了：

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

`@strict`（来自 `huggingface_hub.dataclasses`）把它变成一个带校验的 dataclass：每个字段都有类型标注、都有默认值，拼错字段名会直接报错而不是悄悄变成一个没人读的属性。两个值得记住的推论：

- **类的定义体就是架构旋钮的完整清单。** 超参数无处藏身。从头到尾读完这四个类的定义体，是学习架构的正当方式。
- **派生值住在 `__post_init__` 而不是 `__init__`。** 那里是 config 从"一条记录"变成"一小段程序"的地方 —— 见 §3。

`model_type` 是注册表的键。`"gemma4_text"`、`"gemma4_vision"`、`"gemma4_audio"`、`"gemma4"` 各自映射到 `Auto*` 表里的一个模型类，§4 就靠这个工作。

### 2. 嵌套，以及一个模态如何凭空消失

`Gemma4Config` 显式声明它的孩子：

```python
class Gemma4Config(PreTrainedConfig):
    model_type = "gemma4"
    sub_configs = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4VisionConfig,
        "audio_config": Gemma4AudioConfig,
    }
```

`sub_configs` 是 `from_pretrained` 从嵌套 JSON 重建嵌套对象的依据，也是 `__post_init__` 用来强制类型的依据：

```python
if self.text_config is None:
    self.text_config = Gemma4TextConfig()          # 文本是必须的
elif isinstance(self.text_config, dict):
    self.text_config = Gemma4TextConfig(**self.text_config)

if self.vision_config is None:
    logger.info("vision_config is None. Gemma4Model.vision_tower will not be initialized.")
```

注意这个不对称。缺失的 `text_config` 会用默认值补上，因为没有语言模型的 Gemma 4 不是 Gemma 4。缺失的 `vision_config` 或 `audio_config` 则被**尊重**：那座塔干脆不建。这不是假设，正是已发布 checkpoint 之间的真实差别：

```json
// google/gemma-4-31B-it/config.json
"audio_config": null,
```

这一个 `null` 就是"音频仅 E2B/E4B"的全部机制。建模代码里没有任何一处 `if size == "31B"`。

特殊 token id 挂在**顶层** config 上，而不是 `text_config`：

```python
boi_token_id:    int | None = 255_999   # 图像开始
eoi_token_id:    int | None = 258_882   # 图像结束
image_token_id:  int | None = 258_880
video_token_id:  int | None = 258_884
boa_token_id:    int | None = 256_000   # 音频开始
eoa_token_index: int | None = 258_883   # 音频结束（注意是 _index 不是 _id）
audio_token_id:  int | None = 258_881
```

它们属于那里，因为它们是 **processor 与模型之间的契约**：02 章把这些 id 写进 `input_ids`，08 章再把它们找回来并覆盖其 embedding。这个约定既不归 tokenizer 独有，也不归 decoder 独有。（`eoa_token_index` 这个别扭的命名是代码库里真实的不一致，不是本书的笔误 —— checkpoint 里 `eoa_token_id` 和 `eoa_token_index` 两个都有。）

### 3. `__post_init__`：config 开始计算的地方

`Gemma4TextConfig.__post_init__` 很短，但做了三件你在 06、08 章还会再见到的事。

**它生成注意力排班表。** 如果你不提供 `layer_types`，它会被推导出来：

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

五个滑窗层，然后一个全局层，如此循环 —— 而且**最后一层无论如何被强制为全局**。这个覆盖值得记住：不管前面发生了什么，算 logits 之前的最后一层一定看得见整条序列。

**它为每种层型挑不同的 RoPE。** 这是最多人忽略的细节：

```python
default_rope_params = {
    "sliding_attention": {"rope_type": "default",      "rope_theta": 10_000.0},
    "full_attention":    {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                          "rope_theta": 1_000_000.0},
}
```

一个只看得见 512 个邻居的层，不需要为 128K 上下文拉伸过的旋转频率 —— 所以它保留经典的 θ=10,000。而全局层确实要看完整个 128K，于是拿到 θ=1,000,000，**并且**只旋转每个 head 四分之一的维度。两套注意力机制，两套位置编码机制。06 章展开讲为什么。

**它有一个你猜不到的副作用。** 打开全双向注意力会悄悄改写窗口：

```python
if self.use_bidirectional_attention == "all":
    self.is_causal = False
    self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds
```

512 的因果窗口只向后看；512 的双向窗口会向**两个**方向各看 512，跨度翻倍。折半让总跨度保持可比。这种东西在论文里根本看不见，在三行源码里一目了然 —— 这就是读源码的全部理由。

### 4. 从 config 到模块

`modeling_gemma4.py` 里的 `Gemma4Model.__init__` 是回报所在。它几乎全是分派：

```python
self.vision_tower = AutoModel.from_config(config.vision_config) if config.vision_config is not None else None
self.language_model = AutoModel.from_config(config=config.text_config)
self.audio_tower  = AutoModel.from_config(config.audio_config)  if config.audio_config  is not None else None
self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config) if config.vision_config is not None else None
self.embed_audio  = Gemma4MultimodalEmbedder(config.audio_config,  config.text_config) if config.audio_config  is not None else None
```

`AutoModel.from_config` 对 Gemma 4 一无所知。它读 `config.model_type` —— `"gemma4_vision"` —— 在 auto 注册表里查到 `Gemma4VisionModel`，然后实例化。正是这层间接，让你能把某个子 config 换成另一个架构的 config 并且照样工作；也正是因此，库里每个模型都暴露 `model_type`。

另外注意这里有**五**组参数，不是三组。两个 `Gemma4MultimodalEmbedder` 与它们服务的塔是分开的 —— 投影进文本空间是独立模块，而在 10 章里它是可以被单独解冻的东西之一。

config 还声明模型如何切分，这一点即使你永远不碰也值得知道它存在：

```python
base_model_tp_plan = {"layers.*.self_attn.q_proj": "colwise", ...}
base_model_ep_plan = {"layers.*.router": "ep_router",
                      "layers.*.experts.gate_up_proj": "grouped_gemm", ...}
base_model_pp_plan = {"embed_tokens": (["input_ids"], ["inputs_embeds"]), ...}
```

张量并行、专家并行、流水并行都被声明为 config 上的**数据**，而不是写死在建模代码里。`base_model_ep_plan` 上的注释是一段可爱的工程诚实：*"do not tp in attention (num_global_key_value_heads=2 too small to partition)"*。

### 5. 四个已发布尺寸到底说了什么

读一个 config 学会字段。把四个并排读，才学会设计。自己拉下来 —— 都是小 JSON 文件，不涉及权重：

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
print(cfg.text_config.num_kv_shared_layers, cfg.text_config.hidden_size_per_layer_input)
```

| `text_config` 字段 | E2B | 31B | 26B-A4B |
|---|---|---|---|
| `hidden_size` | 1536 | 5376 | 2816 |
| `num_hidden_layers` | 35 | 60 | 30 |
| `num_attention_heads` | 8 | 32 | 16 |
| `num_key_value_heads` | **1**（MQA） | 16 | 8 |
| `head_dim` / `global_head_dim` | 256 / 512 | 256 / 512 | 256 / 512 |
| `num_global_key_value_heads` | — | 4 | 2 |
| `sliding_window` | 512 | 1024 | 1024 |
| 全局注意力层 | 每 5 层 | 每 6 层 | 每 6 层 |
| `num_kv_shared_layers` | **35 层里的 20** | 0 | 0 |
| `attention_k_eq_v` | `False` | **`True`** | **`True`** |
| `hidden_size_per_layer_input`（PLE） | **256** | **0** | **0** |
| `use_double_wide_mlp` | `True` | `False` | `False` |
| `use_bidirectional_attention` | **`None`** | **`"vision"`** | **`"vision"`** |
| `enable_moe_block` | `False` | `False` | **`True`**（128 专家，top-8，`moe_intermediate_size=704`） |
| `max_position_embeddings` | 131072 | 262144 | 262144 |
| `final_logit_softcapping` | 30.0 | 30.0 | 30.0 |
| 视觉塔 | 768 宽，16 层 | 1152 宽，27 层 | 1152 宽，27 层 |
| 音频塔 | 1024 宽，12 层 | `null` | `null` |

这张表里有四件事值得停下来看，因为每一件都推翻了一个合理的猜测：

1. **PLE 是小模型技术。** `hidden_size_per_layer_input` 在 E2B 上是 256，在**两个大模型上都是 0** —— 逐层 embedding 通路在上面被整个关掉。PLE 在参数稀缺的地方买容量；到了 31B，你宁可直接要更大的 hidden size。07 章会再回到这里。
2. **双向视觉注意力是大模型技术**，而且方向恰好相反：E2B 不设 `use_bidirectional_attention`（包括图像 token 在内一切都是因果的），而 31B 和 26B-A4B 设成 `"vision"`。08 章会把两种 mask 都画出来。
3. **E2B 对 KV 成本近乎偏执** —— 只有一个 KV head（MQA，不是 GQA），**并且** 35 层里有 20 层不持有自己的 KV 投影。大模型付得起 16/8 个 KV head，但用 `attention_k_eq_v` 把 key 投影复用为 value 投影扳回来。同一张账单的两种答法；06 章详述。
4. **视觉塔不是同一座塔。** E2B 的是 768 宽 / 16 层且启用 clipped linear；大模型的是 1152 宽 / 27 层且 `standardize=True`。跨尺寸**预处理**完全一致（03 章），编码器则不然。

### 6. `modular_gemma4.py`，以及这份代码该怎么读

有两个建模文件，但只有一个是你该改的真实代码：

- **`modular_gemma4.py`** 是源头。它比较短，靠**继承**工作：`from ..gemma3.modeling_gemma3 import Gemma3Attention, Gemma3DecoderLayer, Gemma3MLP, Gemma3RotaryEmbedding, ...`，然后把 Gemma 4 的类定义成只覆写变化部分的子类。
- **`modeling_gemma4.py`** 由它经库的 `modular` 工具生成：所有继承来的方法都被展平进去，文件自包含，你永远不用跨模型目录去追基类。

实用的阅读规则：

> **想知道模型做什么，读 `modeling_gemma4.py`。** 它是事实来源，而且是完整的。
> **想知道什么是*新的*，读 `modular_gemma4.py`。** 它的 import 列表和覆写就是相对 Gemma 3 的精确 diff。

那份 import 列表本身就有信息量。`Gemma4MLP` 和解码层骨架直接来自 Gemma 3；视觉塔、音频塔、PLE、MoE 块和 mask 构造则不是。Gemma 4 真正新增的表面积，比 126KB 的生成代码看上去要小得多。

## 设计空间

每个多模态库都要解决"如何描述一个由若干模型组成的模型"。流通的三种答案：

- **嵌套 config + 注册表**（Gemma 4，以及整个 `transformers`）。子 config 自带 `model_type`，`AutoModel.from_config` 负责分派。JSON 冗长，但每座塔都可独立描述、独立加载、独立替换 —— 而且 `null` 干净地表示"这个模型没有这座塔"。
- **一个扁平 config**（早期 LLaVA 分支、许多研究仓库）。简单，直到两座塔都想要一个叫 `hidden_size` 的字段，于是你得到 `vision_hidden_size`、`mm_hidden_size`，以及一场关于命名的争论。
- **config 即代码**（Fairseq/Detectron 风格的 builder，以及 `timm` 的构造函数）。灵活性最大，但架构不再是可序列化的产物 —— 你没法通过 diff 两个 JSON 来 diff 两个模型，而 §5 干的正是这件事。

嵌套注册表方案正是 §5 那张表得以存在的原因，这本身就是对该设计的有力论证。

## 自测

1. 你有一个 `audio_config=None` 的 `Gemma4Config`。`Gemma4Model` 的哪些属性是 `None`？如果你仍然给 `forward` 传 `input_features` 会怎样？
2. `num_hidden_layers=12` 且不提供 `layer_types`。哪些层下标会成为 `full_attention`？（小心 —— 有两条规则。）
3. 为什么 `image_token_id` 归顶层 config 而不是 `text_config`？
4. E2B 设 `num_key_value_heads=1`。这叫什么？相对 31B 的 16，它对 KV cache 大小意味着什么？
5. 你只想微调"视觉进入文本空间"的那个投影。`Gemma4Model.__init__` 建的五个子模块里，你该解冻哪一个？

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_config_to_model.ipynb` | 手写四个 config 搭出迷你 Gemma 4，前向一批随机 token，按子模块拆解参数量，再走一遍 `save_pretrained` / `from_pretrained` 往返 | 🟢 CPU，零下载 |
