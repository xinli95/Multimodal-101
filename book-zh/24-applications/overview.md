# 24 · Applications — 应用与评测

**与 Part I 的关系。** 本章每一个生产关切，Part I 都已经给过你一个数。视觉 token 成本就是 [03 章](../03-image-processor/index.md)的 soft token 菜单 —— 每张图 280、每个视频帧 70，以及决定你的文档能不能被读清的那道 [OCR 悬崖](../04-vision-tower/index.md)（实测：70 token 把 `QX-7741-ZB` 读成 `QQ-7414-2B`，560 才完全正确）。延迟与吞吐是 [09 章](../09-generation-and-serving/index.md)的 cache 曲线和 15 倍批量放大。"要不要微调"是 [10 章](../10-finetuning/index.md)那个冻结还是适配的决定。

本章的要点在于：这些不是彼此独立的话题。一个检索十张页面图的多模态 RAG 系统，在问题还没问出口之前就已经花掉 2,800 个上下文 token；这笔账划不划得来，你现在可以从第一性原理算，而不必盲目跑基准。

## 学习目标

- 搭建多模态 RAG：文档解析 + 多模态嵌入检索 + VLM 生成，且 token 预算事先算清
- 理解多模态 Agent 的核心循环：截图理解 → 决策 → 动作（GUI agent）
- 建立评测直觉：自动指标、LLM-as-judge、arena 盲测 —— 各自适用在哪、各自的陷阱在哪
- 掌握生产关切：视觉 token 成本、延迟、缓存、模型路由

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks](notebooks/README.md) — 实践代码
