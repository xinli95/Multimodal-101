# 07 · Unified & Omni — 统一多模态模型

集大成的一章：一个模型同时做理解与生成、跨所有模态。这是闭源前沿（GPT-5、Gemini 3）的真实形态，也是开源最活跃的研究方向。理论讲清楚"统一"的几种架构路线，实践跑 BAGEL / InternVL-U / Qwen 3.5-Omni。

## 学习目标

- 理解"理解用连续特征、生成用离散/扩散"这一根本矛盾的四种解法
- 读懂三个代表架构：Janus（解耦编码）、BAGEL（MoT + 混合 AR/diffusion）、Qwen-Omni（Thinker-Talker）
- 理解统一训练带来的涌现能力（世界知识编辑、跨模态推理）
- 跑通一个开源统一模型的理解+生成+编辑全流程

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks/](notebooks/) — 实践代码
