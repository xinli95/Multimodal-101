# 02 · Document Understanding — 文档理解与 OCR

VLM 最实用的落地场景：PDF/扫描件/表格/图表 → 结构化文本。也是多模态 RAG 的入口。本章覆盖 OCR 专用 VLM 的设计思路（尤其是"光学上下文压缩"这一巧妙想法），并搭一条完整的 PDF → Markdown 流水线。

## 学习目标

- 理解通用 VLM 与 OCR 专用模型的差异（分辨率策略、输出格式、grounding）
- 理解 DeepSeek-OCR 的核心洞察：用视觉 token 压缩长文本上下文
- 跑通 PDF → Markdown 流水线，并用 OmniDocBench 思路评估质量
- 知道什么时候该用专用 OCR 模型、什么时候直接用通用 VLM API

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks/](notebooks/) — 实践代码
