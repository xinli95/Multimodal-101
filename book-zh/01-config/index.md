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

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_config_to_model.ipynb` | 手写四个 config 搭出迷你 Gemma 4，前向一批随机 token，按子模块拆解参数量，再走一遍 `save_pretrained` / `from_pretrained` 往返 | 🟢 CPU，零下载 |

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/01-config/index.html)。
