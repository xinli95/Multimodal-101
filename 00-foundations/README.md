# 00 · Foundations — 多模态基础

后续所有章节的地基。讲两件事：**不同模态如何进入同一个表示空间**（对齐），以及**生成模型的三大范式**（自回归、扩散、流匹配）。这两条线贯穿全书。

## 学习目标

- 理解对比学习如何把图像和文本拉进同一嵌入空间（CLIP 的核心思想）
- 理解各模态的"token 化"：文本 BPE、图像 patch/VAE latent、音频 codec token
- 能够区分三大生成范式的建模目标、采样方式与各自适用场景
- 跑通 CLIP/SigLIP 的零样本分类与图文检索

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks/](notebooks/) — 实践代码
