# Landscape · 应用与评测

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

## 多模态 RAG 组件

| 组件 | 开源选择 | 备注 |
|---|---|---|
| 文档解析 | DeepSeek-OCR-2、olmOCR-2、MinerU | 见 landscape 的 OCR 部分 |
| 视觉检索嵌入 | ColQwen 系、Jina-CLIP、BGE-VL | late-interaction 效果最好但索引大 |
| 向量库 | Qdrant、LanceDB、Milvus | 均已支持多向量/late-interaction |
| 生成端 | Qwen3-VL / 闭源 VLM API | 见 Part I |

## Agent 框架与模型

- GUI grounding 专用模型：UI-TARS 系（字节）、CogAgent；通用 VLM（Qwen3-VL、Claude computer use、Gemini）也已内建 GUI 能力
- 框架：browser-use、Agent SDK 类（各家 API 均有 computer-use 工具）

## 评测基础设施

| 工具/基准 | 用途 |
|---|---|
| VLMEvalKit、lmms-eval | 开源 VLM 批量评测框架 |
| MMMU、OCRBench、Video-MME | 理解侧标准基准 |
| GenEval、VBench、OmniDocBench | 生成/文档侧基准 |
| LMArena（Vision/Image/Video） | 人类偏好盲测，看趋势最直观 |

## 趋势判断

1. 多模态 RAG 中"解析路线 vs 视觉检索路线"尚未分胜负，混合检索是务实选择。
2. GUI agent 是 2026 年 VLM 商业化最快的赛道。
3. 评测正在从"一个分数"走向"维度化 + 特定任务自建评测集"；教程所有 notebook 遵循这一实践。
