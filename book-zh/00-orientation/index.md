# 00 · Orientation — 一个模型，从头拆到尾

大多数多模态教程是目录：这是 CLIP，这是 LLaVA，这是 Qwen-VL，这是扩散模型。读完你记住了一堆名字，却仍然答不上唯一重要的问题——**这些东西内部到底长什么样，那些代码在哪。**

本书 Part I 反过来做：只挑**一个**模型 [Gemma 4](https://huggingface.co/docs/transformers/en/model_doc/gemma4)，顺着它在 `transformers` 里的真实实现逐层拆开。每一章都打开一个具体文件、点名具体的类、追踪具体的张量，然后亲手复现其中一块，用断言证明我们真的看懂了。

## 为什么 Gemma 4 是理想教具

| 特性 | 对学习的意义 |
|---|---|
| 文本 + **图像 + 视频 + 音频** 输入同在一个 checkpoint | 四个模态前端在同一份一致的代码里，而不是四篇讲四个模型的博客 |
| **固定 token 预算**下的可变长宽比 | "图像怎么变成 token"的现代答案，而且取舍看得见 |
| 视觉塔的 **2D RoPE + 学习式 2D 位置表** | 给网格而不是给直线做位置编码——空间推理的根基 |
| **Per-Layer Embeddings (PLE)** | 一个教科书里找不到的设计，逼你去读代码而不是套模板 |
| **滑窗 + 全局**混合注意力、跨层 KV 共享、K=V 投影 | 现代推理省钱的招数在一个 decoder 里齐了 |
| 26B-A4B 尺寸里的 **MoE** | 路由与稀疏专家，同一套代码 |
| 从 **E2B 到 31B**，且都不 gated | 小的单卡能跑，大的留给需要它的章节 |
| `transformers` 里的开源实现 | 源码即事实，不用从论文插图里猜 |

一个合理的质疑：只学一个模型，就只懂一个模型。所以每章末尾都有**「设计空间」**小节，把 Gemma 4 的选择和替代方案摆在一起（LLaVA 的 MLP projector、BLIP-2 的 Q-Former、Qwen-VL 的 M-RoPE、InternVL 的 tiling、Whisper 的音频编码器）。单个模型是脊椎，设计空间小节是地图。

## 本章你会学到

1. Gemma 4 家族到底有什么——尺寸、模态、上下文长度，以及哪些能力和尺寸相关
2. 多模态理解从哪来，一页讲完（CLIP → BLIP-2 → LLaVA → 今天）
3. 配好能跑通 Part I 的环境
4. 端到端跑一次 Gemma 4，打印它的模块树，把树上每个分支对应到本书的某一章

## 全书要拆的那条数据流

```
  messages ──► chat template ──► tokenizer ──► input_ids                       ch02
               （占位符段：<image>×280、<audio>×N、<video>×...）
                                   │
  image ──► Gemma4ImageProcessor ──┤► pixel_values + image_position_ids        ch03
                                   │
                                   ▼
                          Gemma4VisionModel                                    ch04
        patch embed ─► 2D RoPE ─► encoder ×16 ─► pooler ─► soft tokens
                                   │
  audio ──► Gemma4AudioFeatureExtractor ─► Gemma4AudioModel ─► soft tokens     ch05
  video ──► Gemma4VideoProcessor ─► （抽帧）─► Gemma4VisionModel                ch05
                                   │
                                   ▼
                     Gemma4MultimodalEmbedder （─► 文本嵌入空间）                ch04/05
                                   │
                                   ▼
        masked_scatter 把 soft token 写进 inputs_embeds + 构造 4D mask          ch08
                                   │
                                   ▼
                           Gemma4TextModel                                     ch06
            PLE ─► [滑窗 | 全局] 注意力 ─► MLP 或 MoE  ×N                       ch07
                                   │
                                   ▼
                    lm_head ─► logits ─► generate() ─► 文本                     ch09
                                   │
                                   ▼
                        LoRA / Trainer：改变权重                                 ch10
```

## 家族一览

| 尺寸 | 参数 | 文本 | 图像 | 视频 | 音频 | 上下文 |
|---|---|---|---|---|---|---|
| **E2B** | ~2B 有效 | ✅ | ✅ | ✅ | ✅ | 128K |
| **E4B** | ~4B 有效 | ✅ | ✅ | ✅ | ✅ | 128K |
| **31B** | 31B 稠密 | ✅ | ✅ | ✅ | ❌ | 256K |
| **26B-A4B** | 26B MoE，~4B 激活 | ✅ | ✅ | ✅ | ❌ | 256K |

四个尺寸都有预训练版和指令微调版（`-it`），都是可配置 thinking 模式的推理模型，都原生支持 `system` 角色与 function calling。**音频只在 E2B / E4B 上原生支持**——这个不对称本身就值得琢磨，05 章会讲音频塔的代价。

Part I 默认使用 **`google/gemma-4-E2B-it`**（单卡 24GB、权重约 10GB、未 gated）；当重点是结构而非行为时，改用随机初始化的迷你 config。

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_hello_gemma4.ipynb` | 端到端跑通 Gemma 4，然后 `print(model)` 把每个子模块对应到本书章节 | 🟡 24GB 显存，约 10GB 下载 |
| <a href="../../00-orientation/notebooks/02_clip_zero_shot.html"><code>02_clip_zero_shot.ipynb</code></a> | 对照实验：CLIP 能做零样本分类，但数不清有几只猫。划定对比式编码器的能力边界——正是 Gemma 4 用 LLM 填上的那道缝 | 🟢 CPU |

## 环境

```bash
pip install "transformers>=5.14" accelerate torch pillow requests matplotlib
```

Part I 基于 **transformers 5.14.1** 写成，Gemma 4 位于 `transformers/models/gemma4/`。找到你自己那份，接下来会一直读它：

```python
import transformers, os
print(os.path.join(os.path.dirname(transformers.__file__), "models", "gemma4"))
```

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/00-orientation/index.html)。
