# Multimodal-101

一个兼具**理论**与**实践**的多模态 AI 教程，组织方式是**把一个模型从头拆到尾**，而不是从外部罗列一堆模型。与 [AI-101](https://github.com/xinli95/AI-101)、[Speech-101](https://github.com/xinli95/Speech-101) 同属 101 系列，用 [TeachBooks](https://teachbooks.io) 构建成可直接浏览的在线书。

目标读者：有基本 Python / 深度学习基础，想知道一个多模态模型内部到底是什么——并顺便把 `transformers` 学扎实，因为答案就写在那里。

## 本书结构

**Part I · 一个多模态模型的解剖** 拿 [Gemma 4](https://huggingface.co/docs/transformers/en/model_doc/gemma4) 开刀，跟着一条 prompt 走完全程：config → chat template 与 tokenizer → 图像与音频前端 → 视觉塔与音频塔 → 四个模态汇成一条序列的融合点 → 解码器 → `generate()` → 微调。每一章都打开 `transformers/models/gemma4/` 里的真实实现，点名具体的类与函数，追踪张量，然后亲手复现其中一块，并断言结果与库一致。

Gemma 4 配得上这个位置，是因为一个开源 checkpoint 家族里几乎装下了所有值得讲的东西：文本、图像、视频、音频输入；固定 token 预算下的可变长宽比；2D RoPE；Per-Layer Embeddings；滑窗与全局混合注意力；跨层 KV 共享；MoE；128K–256K 上下文。而 E2B 尺寸小到单张消费级显卡就能跑。

**Part II · 生成侧** 讲 Gemma 4 不做的事：文本进，像素与声音出。扩散、DiT、flow matching、视频、TTS，以及双向都做的统一模型。这几章保持 survey 形式——理论、格局、notebook——因为在那里"广度"才是正确的形状。

## 设计理念

1. **是脊椎，不是清单。** Part I 是穿过一个模型的一条连续路径。你永远知道自己在哪，因为全书只有一张数据流图，每章都在图上标出自己的位置。
2. **读源码。** 本书里关于架构的说法都是可核对的：它们指向你自己 `transformers` 安装里的某个文件和符号。凡是重要的机制，都有 notebook 手写复现并与库 `assert_close`。
3. **一个模型，外加一张地图。** Part I 每章末尾都有**「设计空间」**小节，把 Gemma 4 的选择与替代方案（LLaVA、BLIP-2、Flamingo、Qwen-VL、InternVL、Whisper）摆在一起。单个模型是脊椎，这些小节是地图。
4. **对抗过时。** 这个领域约每季度洗牌一次。[`landscape.md`](landscape.md) 与 Part II 的格局页记录当前状态，并**标注 last-verified 日期**。超过 6 个月请以官方渠道为准。

## Part I 章节

| 章节 | 主题 | 打开的源码 |
|---|---|---|
| [00 · Orientation](00-orientation/index.md) | 为什么只讲一个模型；家族全景；数据流地图；从 CLIP 到今天 | — |
| [01 · Config](01-config/index.md) | config 如何变成模型；文本/视觉/音频嵌套配置 | `configuration_gemma4.py` |
| [02 · Text I/O](02-text-io/index.md) | chat template、tokenizer、占位符段、function calling | `processing_gemma4.py` |
| [03 · Image Processor](03-image-processor/index.md) | token 预算下的像素；长宽比；48 整除规则 | `image_processing_gemma4.py` |
| [04 · Vision Tower](04-vision-tower/index.md) | patch 嵌入、2D RoPE、池化、投影进文本空间 | `modeling_gemma4.py` |
| [05 · Audio and Video](05-audio-and-video/index.md) | mel 特征、卷积下采样、分块注意力；抽帧 | `feature_extraction_gemma4.py` 等 |
| [06 · Text Decoder](06-text-decoder/index.md) | 滑窗 vs 全局注意力、QK-norm、KV 共享、K=V | `modeling_gemma4.py` |
| [07 · PLE and MoE](07-ple-and-moe/index.md) | Per-Layer Embeddings；路由专家 | `modeling_gemma4.py` |
| [08 · Fusion and Masks](08-fusion-and-masks/index.md) | 模态真正相遇的地方；双向视觉注意力 | `modeling_gemma4.py` |
| [09 · Generation and Serving](09-generation-and-serving/index.md) | `generate()`、cache、批量、vLLM | `modeling_gemma4.py` |
| [10 · Fine-Tuning](10-finetuning/index.md) | 冻什么、LoRA、多模态 collator、显存账 | `peft`、`Trainer` |
| [Landscape](landscape.md) | Gemma 4 在"会读"的模型里处于什么位置 | — |

## Part II 章节

| 章节 | 主题 | 理论核心 | 实践模型（开源 / 闭源） |
|---|---|---|---|
| [20 · 图像生成与编辑](20-image-generation/overview.md) | 文生图与指令编辑 | Diffusion → DiT → Flow Matching；in-context 条件 | FLUX.2 [klein]、Qwen-Image-Edit / GPT Image 2、Nano Banana |
| [21 · 视频生成](21-video-generation/overview.md) | 文/图生视频 | Video DiT、时序注意力、Causal VAE | Wan 2.2、HunyuanVideo 1.5 / Veo 3.1 |
| [22 · 语音与音频生成](22-audio-generation/overview.md) | 文本转语音 | Codec token、TTS 即语言模型 | Kokoro、Higgs Audio v3 / 闭源 TTS |
| [23 · 统一与 Omni 模型](23-unified-omni/overview.md) | 理解**与**生成同在一个模型 | 统一双向的架构 | BAGEL、InternVL-U、Qwen3.5-Omni |
| [24 · 应用与评测](24-applications/overview.md) | 多模态 RAG、Agent、Benchmark | 文档检索、评判、路由 | 综合运用前面各章 |

## 学习路径建议

- **完整路径**：Part I 00 → 10 顺序读——它是一条完整的论证，跳着读会断——之后按兴趣挑 Part II。
- **只关心 VLM 怎么工作**：Part I 00 → 04，再加 08。承重的是 03、04、08 三章。
- **只关心推理成本**：Part I 03（token 预算）→ 06（注意力与 KV cache）→ 09（生成与部署）。
- **只关心生成**：Part I 00，然后 Part II 20 → 23。

## 硬件分级

每个 notebook 都标了最低硬件档。Part I 刻意做成：讲**结构**的 notebook 完全不需要 GPU、不需要下载（用随机初始化的迷你 config），讲**行为**的才用 `google/gemma-4-E2B-it`（约 10GB，未 gated）。

| 档位 | 配置 | 能跑什么（举例） |
|---|---|---|
| 🟢 CPU / 笔记本 | 无需 GPU | config 与 mask 解剖、图像处理器复现、tokenizer 实验、CLIP、Kokoro、Whisper、闭源 API |
| 🟡 消费级 GPU | 12–24GB 显存 | Gemma 4 E2B 端到端、LoRA SFT、Qwen3-VL 小尺寸、FLUX.2 [klein] |
| 🔴 工作站 / 云 | 40GB+ 显存 | Gemma 4 31B / 26B-A4B、Wan 2.2、HunyuanVideo 1.5、大尺寸 InternVL3.5 |

## 关于 License

教程代码为 MIT。模型 license 差异很大（Apache 2.0 / Gemma 条款 / 非商用 / 分级商用）——Gemma 4 采用 Gemma 条款，Part II 的模型在各章 `landscape.md` 中逐一标注。商用前请自行核对。

> 📝 Notebook 为中英文版共用（英文），点击章节内的链接跳转到英文版页面阅读/下载。

---

*格局数据最后核实：2026-08-02。Part I 基于 transformers 5.14.1 写成。*
