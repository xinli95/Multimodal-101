# Landscape · 多模态理解

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

Part I 深挖一个模型，这一页是它的配重：Gemma 4 在所有"会读图、读文档、读视频、读音频"的模型里处于什么位置。生成侧有各自的格局页（[20](20-image-generation/landscape.md)、[21](21-video-generation/landscape.md)、[22](22-audio-generation/landscape.md)、[23](23-unified-omni/landscape.md)）。

## 几乎所有模型底下的那几个视觉编码器

| 模型 | 机构 | License | 说明 |
|---|---|---|---|
| SigLIP 2 | Google | Apache 2.0 | 当前开源 VLM 里最常见的视觉塔 |
| CLIP (ViT-L/14) | OpenAI | MIT | 经典基线，教学首选 |
| DINOv2/v3 | Meta | Apache 2.0 | 自监督一系，稠密预测强 |

Gemma 4 的视觉塔是自家训练的，而不是外挂一个已发布的 SigLIP checkpoint——这也是它的预处理（不做 ImageNet 归一化、学习式 2D 位置表）与 LLaVA 一系默认做法不同的部分原因。见 [03 章](03-image-processor/index.md)与 [04 章](04-vision-tower/index.md)。

## 开源 VLM

| 模型 | 机构 | 规模 | License | 亮点 |
|---|---|---|---|---|
| **Gemma 4** | Google | E2B / E4B / 26B-A4B / 31B | Gemma 条款 | **Part I 的主角。** 全系文本+图像+视频，E2B/E4B 带音频；128K–256K 上下文；PLE、MoE、可变长宽比视觉 |
| **Qwen3-VL** | 阿里 | 2B → 235B-A22B (MoE) | Apache 2.0 | 开源旗舰，对标 Gemini 2.5 Pro / GPT-5；OCR、grounding、视频、agent 全面 |
| **InternVL3.5** | 上海 AI Lab | 1B → 241B-A28B | MIT/Apache（按尺寸） | 开源 SOTA 竞争者，推理与效率并重 |
| Llama 4 multimodal | Meta | 多尺寸 MoE | Llama 协议 | 生态大，英文强 |
| Molmo | AI2 | 1B–72B | Apache 2.0 | 数据完全开放；独特的指点能力 |
| Pixtral | Mistral | 12B/124B | Apache 2.0 | 欧洲代表 |
| Phi-4 multimodal | 微软 | ~5.6B | MIT | 端侧小模型代表 |

读 Part I 时值得对照的三条设计轴：

| 维度 | Gemma 4 | Qwen3-VL | InternVL3.5 | LLaVA (2023) |
|---|---|---|---|---|
| 图像 → token | 固定预算、原生长宽比、3×3 池化 | 原生分辨率、2×2 merge、序列随图增长 | 固定 tile + 缩略图 | 一张 224×224 |
| 空间位置 | 学习式 2D 表 + 2D RoPE | M-RoPE（时/高/宽） | tile 上的标准 1D | 576 个 patch 上的 1D |
| Connector | Pooler + 线性 embedder | MLP merger | MLP + pixel shuffle | 单个 MLP |

## 闭源前沿

| 模型 | 机构 | 多模态特点 |
|---|---|---|
| Gemini 3.x Pro | Google | 从第一天起就是统一多模态——文本/图像/音频/视频原生；公认多模态理解最强。Gemma 4 是这条线的开源同胞，这也是它值得研究的原因之一 |
| GPT-5.x | OpenAI | 原生多模态输入，推理链强 |
| Claude (Opus/Sonnet/Fable) | Anthropic | 图像输入 + 文档理解，工程任务强 |

## 文档与 OCR

| 模型 | 机构 | 规模 | License | 亮点 |
|---|---|---|---|---|
| **DeepSeek-OCR-2** | DeepSeek | ~3B | MIT | 2026-01 发布；grounded Markdown、高吞吐——"PDF → Markdown"的开源首选之一 |
| **olmOCR-2** | AI2 | 8B | Apache 2.0 | 完全开放（数据/训练/评测）；OmniDocBench 均分 83+ |
| PaddleOCR-VL-1.5 | 百度 | ~0.9B | Apache 2.0 | 极小参数量下的 SOTA 级效果；端侧/批处理友好 |
| Chandra | Datalab | 9B | - | 多语种强 |
| MinerU / Marker | 社区 | pipeline | 注意 AGPL/GPL | 经典流水线派的实用工具 |

通用 VLM 的 OCR 已经足够好，轻量文档任务未必需要专用模型——而 Gemma 4 的 soft token 菜单（[03 章](03-image-processor/index.md)）正是决定它能不能读清小字的那个旋钮。闭源 API 在复杂表格与手写上仍是稳健性天花板，但更贵，且没有坐标级 grounding。

| 场景 | 选择 |
|---|---|
| 批量处理 PDF（自有 GPU） | DeepSeek-OCR-2 / olmOCR-2 |
| 端侧或纯 CPU 环境 | PaddleOCR-VL-1.5 |
| 少量、高价值、极复杂文档 | 闭源 VLM API |
| 需要可溯源（bbox 出处） | DeepSeek-OCR-2（grounded 输出） |

评测：OmniDocBench（最全面）、OCRBench v2、olmOCR-Bench。

## 语音理解

| 模型 | 机构 | 说明 |
|---|---|---|
| Whisper (large-v3 / turbo) | OpenAI | 一切都拿它作基线的 encoder-decoder ASR |
| Gemma 4 E2B/E4B 音频塔 | Google | 通用多模态模型内部的分块注意力编码器——[05 章](05-audio-and-video/index.md) |
| Parakeet / Canary | NVIDIA | CTC/transducer 流式一系 |

## 趋势判断

1. 开源与闭源在标准基准上大体持平；闭源的优势集中在长尾稳健性与原生音视频融合。
2. "会思考的 VLM"（多模态 CoT + RLVR）已是新模型的默认配置——Gemma 4 四个尺寸全都是。
3. 音频作为一等输入模态（而非外挂 ASR 管线）正在从前沿模型向下扩散；Gemma 4 E2B/E4B 是够得着的开源样本。
4. GUI / computer-use agent 是 VLM 增长最快的落地场景。
5. 关注 Qwen3.8（>1T 参数多模态，2026-07 WAIC 预览；承诺开放权重但尚未发布）。
