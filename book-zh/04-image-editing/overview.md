# 04 · Image Editing — 图像编辑

2025 年后独立成型的赛道："对着一张图说人话，模型精确改图"。本章讲指令式编辑的原理（in-context 条件、参考图注入），实践对比开源（Qwen-Image-Edit、FLUX Kontext）与闭源（Nano Banana、GPT Image 2）。

## 学习目标

- 理解从 inpainting（画 mask）到 instruction editing（纯语言）的范式转变
- 理解 in-context conditioning：参考图作为序列条件与目标图联合注意力
- 掌握编辑的核心难题：改动局部性（不该变的别变）与身份一致性
- 跑通开源编辑模型，与闭源 API 做同题对比

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks](notebooks/README.md) — 实践代码
