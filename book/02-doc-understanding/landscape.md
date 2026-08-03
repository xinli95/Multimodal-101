# Landscape · 文档理解与 OCR

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

## 开源专用模型

| 模型 | 机构 | 规模 | License | 亮点 |
|---|---|---|---|---|
| **DeepSeek-OCR-2** | DeepSeek | ~3B | MIT | 2026-01 发布；grounded Markdown、高吞吐，"PDF→Markdown" 开源首选之一 |
| **olmOCR-2** | AI2 | 8B | Apache 2.0 | 全开放（数据/训练/评测），OmniDocBench 均分 83+ |
| PaddleOCR-VL-1.5 | 百度 | ~0.9B | Apache 2.0 | 极小参数量拿 SOTA 级效果，端侧/大批量友好 |
| Chandra | Datalab | 9B | - | 多语言强 |
| MinerU / Marker | 开源社区 | 流水线 | AGPL/GPL 注意 | 传统流水线派实用工具 |

## 通用 VLM 直接做文档

- Qwen3-VL / InternVL3.5 的 OCR 能力已很强，少量文档场景无需专用模型。
- 闭源 API（Gemini、GPT-5、Claude）对复杂表格/手写体的鲁棒性仍是天花板，但成本高、无坐标 grounding（易幻觉且难溯源）。

## 选型速查

| 场景 | 推荐 |
|---|---|
| 海量 PDF 批处理（自建 GPU） | DeepSeek-OCR-2 / olmOCR-2 |
| 端侧或 CPU 环境 | PaddleOCR-VL-1.5 |
| 少量高价值文档、格式极复杂 | 闭源 VLM API |
| 需要可溯源（bbox 定位） | DeepSeek-OCR-2（grounded 输出） |

## 评测基准

- OmniDocBench（最全面）、OCRBench v2、olmOCR-Bench
