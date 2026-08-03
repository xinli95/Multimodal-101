# Multimodal-101

一个兼具**理论**与**实践**的多模态 AI 入门教程，与 [AI-101](https://github.com/xinli95/AI-101)、[Speech-101](https://github.com/xinli95/Speech-101) 同属 101 系列，用 [TeachBooks](https://teachbooks.io) 构建成可直接浏览的在线书。目标读者：有基本 Python / 深度学习基础，想系统理解多模态模型的原理，并能亲手跑通最新开源模型、调用闭源前沿 API 的工程师和研究者。

## 设计理念

1. **理论与实践双轨**：每章都包含 `theory.md`（原理与关键论文）和 `notebooks/`（可运行代码）。原理讲清楚"为什么是这个架构"，实践保证"今天就能跑起来"。
2. **开源 + 闭源对照**:每个方向都提供"本地可跑的开源模型"和"闭源前沿 API"两条路径，让你既理解内部机制，又了解能力天花板在哪。
3. **对抗过时**：多模态领域约每季度洗牌一次。每章的 `landscape.md` 记录当前格局，并**标注 last-verified 日期**。看到超过 6 个月的日期，请以官方渠道为准。
4. **演进脉络优先**：不罗列模型，而是讲清楚技术演进线（如图像生成的 DDPM → LDM → DiT → Flow Matching），新模型出来时你能自己定位它在脉络中的位置。

## 章节结构

| 章节 | 主题 | 理论核心 | 实践模型（开源 / 闭源） |
|---|---|---|---|
| [00-foundations](00-foundations/overview.md) | 多模态基础 | 对比学习、三大生成范式 | CLIP / SigLIP |
| [01-vlm](01-vlm/overview.md) | 视觉语言模型 | ViT + Connector 架构、训练三阶段 | Qwen3-VL、InternVL3.5 / Gemini、GPT-5 |
| [02-doc-understanding](02-doc-understanding/overview.md) | 文档理解与 OCR | 光学上下文压缩 | DeepSeek-OCR-2、olmOCR-2 / 闭源 VLM API |
| [03-image-generation](03-image-generation/overview.md) | 图像生成 | Diffusion → DiT → Flow Matching | FLUX.2 [klein]、Qwen-Image / GPT Image 2 |
| [04-image-editing](04-image-editing/overview.md) | 图像编辑 | 指令式编辑、in-context 条件 | Qwen-Image-Edit、FLUX Kontext / Nano Banana |
| [05-video-generation](05-video-generation/overview.md) | 视频生成 | Video DiT、时序注意力、Causal VAE | Wan 2.2、HunyuanVideo 1.5 / Veo 3.1 |
| [06-audio](06-audio/overview.md) | 语音与音频 | ASR 架构、Codec Token LM | Whisper、Kokoro、Higgs Audio v3 / 闭源 TTS |
| [07-unified-omni](07-unified-omni/overview.md) | 统一多模态模型 | 理解+生成统一架构演进 | BAGEL、InternVL-U、Qwen3.5-Omni |
| [08-applications](08-applications/overview.md) | 应用与评测 | 多模态 RAG、Agent、Benchmark | 综合运用前面各章 |

## 学习路径建议

- **完整路径**：00 → 01 → 03 → 05 → 06 → 07，其余章节按需插入。00 章是地基，07 章是集大成，顺序不建议跳。
- **只关心理解（VLM）**：00 → 01 → 02 → 08。
- **只关心生成**：00 → 03 → 04 → 05 → 06。

## 硬件分层

每个 notebook 标注最低配置档位：

| 档位 | 配置 | 能跑什么（示例） |
|---|---|---|
| 🟢 CPU / 笔记本 | 无 GPU 亦可 | CLIP、Kokoro-82M、Whisper、闭源 API 调用 |
| 🟡 消费级 GPU | 12–24GB VRAM | Qwen3-VL 小尺寸、FLUX.2 [klein] 4B、SD 3.5 |
| 🔴 工作站 / 云 | 40GB+ VRAM | Wan 2.2、HunyuanVideo 1.5、InternVL3.5 大尺寸 |

## License 提醒

教程代码本身采用 MIT。但各章使用的模型 license 差异很大（Apache 2.0 / 非商用 / 分级商用），每章 `landscape.md` 中有明确标注，商用前务必自查。

---

*Landscape 数据 last-verified: 2026-08-02*
