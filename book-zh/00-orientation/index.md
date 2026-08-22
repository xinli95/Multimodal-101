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
4. 先建立高层 mental model，再端到端跑一次 Gemma 4，把模块树的实现细节挂回这张总图

## Mental model：三个学习出来的模块

先把 class 名字都放下。对一张图像，大多数 VLM 都可以压缩成三个学习出来的模块：

```text
                         每一步在决定什么？

图像
  │
  ├─ processor ───► 这张图值得花多少视觉计算？
  │                 （resize、patch 网格、token 预算；这里还不理解语义）
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. VISION TOWER                                                │
│     局部 patch embeddings ──► 带上下文的视觉 features            │
│     「图里有什么，各区域之间是什么关系？」                        │
└──────────────────────────────────────────────────────────────────┘
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. CONNECTOR / COMPRESSOR                                      │
│     大量 vision-width features ──► 较少的 LLM-width soft tokens  │
│     「有多少视觉信息、以什么表示进入语言模型？」                   │
└──────────────────────────────────────────────────────────────────┘
  ▼
视觉 tokens ────────────────┐
                            ├─► 一条 token 序列 ─► 3. LLM ─► 文本
文本 ─► tokenizer ──────────┘                      推理 + 生成
```

脑子里先只留这一句：

> **模态 tower 理解自己的输入；connector 把表示压缩并翻译到 LLM 的 token 空间；LLM 在一条混合序列上推理。**

音频也是同一结构，只是换成 audio tower。视频则是帧走视觉路径，再加时间信息。Processor 与 fusion 很重要，但它们是三个学习模块外围的 plumbing：processor 在 tower 之前分配视觉计算，fusion 决定 soft tokens 在共享序列里放在哪里。

### 跟着一张图走：五次表示变化

只要追踪每个边界上「什么变了、什么没变」，三个模块就会变得具体。令 `B` 为最终视觉 token 预算，`D_v` 为视觉宽度，`D_text` 为 LLM 宽度。

| 阶段 | 表示 | 这一阶段的工作 | 本书在哪放大 |
|---|---|---|---|
| 1. 分配分辨率 | `H×W×3` pixels → resize 后的 pixels + patch 坐标 | 尽量保留长宽比，同时把预算花完；Gemma 4 因为后面 3×3 patches 合成一个 token，所以最多允许约 `9B` 个 patches | ch03 |
| 2. Patch embed | 局部 `16×16×3` pixels → `N×D_v` patch embeddings | 把局部像素块变成向量，并加入空间位置 | ch04 |
| 3. Vision encode | `N×D_v` → `N×D_v` contextual features | 让每个 patch representation 看见整张图的上下文；ViT 首先是**编码器**，不是负责减 token 的模块 | ch04 |
| 4. 压缩 + 对齐 | `N×D_v` → 约 `B×D_text` soft tokens | 3×3 spatial pooling 减少序列长度；projection 把视觉表示宽度改成 LLM embedding 宽度 | ch04 |
| 5. 融合 + 推理 | `B` 个视觉 tokens + `T` 个文本 tokens → 一条 `(B+T)×D_text` 序列 | 把两个模态放进同一序列、构造 mask，再由文本 decoder 产生 logits | ch08 → ch06/07 → ch09 |

这个拆法能避免三个最常见的混淆：

- **Patch vector 还不是带上下文的 visual token。** 它还要经过 projection、位置编码和 vision encoder。
- **ViT 的主要工作不是减少 token 数。** Gemma 4 是在 ViT 之后由 pooler 做压缩。
- **Pooling 和 projection 解决不同问题。** Pooling 改变序列长度；projection 改变 feature width。

### 十一章，其实只有四个问题

章节边界只是实现层面的放大镜，不是十一件互不相关的知识。站在最高层，Part I 只问四个问题：

| 大问题 | 章节 | 第一遍应该留下什么 |
|---|---|---|
| 模型里应该有哪些组件？ | 01 | Config 是 towers、connector 与 decoder 的蓝图 |
| 各种输入怎样变成 LLM 空间里的 token？ | 02–05、08 | preprocess → encode → compress/align → 放进同一序列 |
| 所有东西成为一条序列后发生什么？ | 06–07 | Attention 与 feed-forward layers 改写同一条 residual stream |
| 怎样使用或改变这个模型？ | 09–11 | 生成、微调、部署，并把同一套读法迁移到另一个系统 |

第一遍先学会上面四行；需要打开某个方框时，再进入对应细章。

### 阅读任何 VLM 都能复用的五个问题

以后碰到 LLaVA、Qwen-VL、InternVL 或别的模型，不要先背 class tree。连续问：

1. **Vision tower 属于什么 family？** CLIP/SigLIP 风格 ViT、InternViT，还是 custom ViT？
2. **它怎样初始化？** 来自独立预训练 checkpoint，还是专门为这个多模态模型训练？
3. **多模态训练时 frozen 还是继续更新？** Pretrained 与 frozen 是两件事。
4. **怎样分配 resolution？** 固定 resize、tiling、native/dynamic resolution，还是明确的 token budget？
5. **Tower features 怎样变成 LLM tokens？** Pooling、pixel shuffle、learned merger、resampler、Q-Former，还是 projector？

这五个答案在你读实现之前，就已经解释了一个 VLM 大半的视觉架构。尤其要分清：

> **ViT 回答 neural-network architecture 是什么；CLIP/SigLIP 回答视觉—语言表示可能怎样预训练；VLM training 回答视觉系统和 LLM 怎样连接、哪些部分继续更新。这是三层不同的问题。**

## 实现地图：现在再放大

只有 mental model 稳定以后，下面这张 class-level 地图才真正有用。Part I 的每一章都是这条流水线中的一个局部：

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

## 怎么读 Part I

每一章都是同样的四个部分：

- **导读** —— 你在流水线的哪个位置、会学到什么，以及一张点名本章打开的每个文件与符号的源码地图。
- **源码走读** —— 真正的阅读。代码引自 `transformers 5.14.1`；凡是设计选择不显然的地方，本章论证它而不是断言它。
- **设计空间** —— 别的模型如何回答同一个问题，各自优化什么。正是这一节让"只讲一个模型"不至于变成盲区。
- **自测** —— 答案都能在本章或源码里找到。答得上来说明你读了；答不上来说明你只是扫了一遍。

两个习惯会决定差别：

1. **把源码开着。** Part I 里每个代码块都引自你磁盘上的某个文件。真正的学习多半发生在读它周围那些行的时候 —— 本书负责指路，不负责替代。
2. **把结构类 notebook 跑一遍。** 标 🟢 的那些不需要 GPU、不需要下载。它们存在，是因为"我看懂了解释"和"我能复现这个计算"是两种不同的状态，而只有后者能在你自己的项目里存活。

03、04、08 三章是承重的 —— token 预算、视觉塔、融合点，就是这里"多模态"的实际含义。06 和 07 章则会改变你读**任何**现代 LLM 的方式，不止这一个。

## 自测

1. 哪些 Gemma 4 尺寸能"听"？这个事实在已发布的文件里体现在哪？
2. 一张图、一段视频、一段音频进入模型。它们在哪个点上不再是三种不同的东西？
3. 要回答"Gemma 4 相对 Gemma 3 新在哪"，你会打开哪两个文件？先读哪一个？
4. 本书为什么有些 notebook 用随机初始化的迷你模型，另一些用真实的 E2B 权重？

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
