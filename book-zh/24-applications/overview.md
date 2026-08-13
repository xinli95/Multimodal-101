# 08 · Applications — 应用与评测

把前面各章组装成真实系统：多模态 RAG、多模态 Agent、以及"怎么科学地评估多模态系统"。这一章偏工程，是把教程知识转化为生产力的出口。

## 学习目标

- 搭建多模态 RAG：文档解析（见 [landscape](../landscape.md) 的 OCR 部分）+ 多模态嵌入检索 + VLM 生成
- 理解多模态 Agent 的核心循环：截图理解 → 决策 → 操作（GUI agent）
- 建立评测思维：自动指标、LLM-as-judge、arena 盲测各自的适用与陷阱
- 了解生产工程问题：视觉 token 成本、延迟、缓存、模型路由

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks](notebooks/README.md) — 实践代码
