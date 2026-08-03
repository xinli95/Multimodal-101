# 理论大纲 · 图像编辑

## 1. 编辑范式演进

- **Inpainting 时代**：用户画 mask + 局部重绘（SD inpainting）
- **InstructPix2Pix (2022)**：首次纯指令编辑——用 GPT-3 + P2P 合成 (原图, 指令, 结果) 三元组训练
- **In-context 条件时代**（FLUX Kontext、Qwen-Image-Edit、GPT Image、Nano Banana）：参考图编码为 token 序列，与目标图在 DiT 中联合注意力，天然支持多参考图

## 2. 核心技术问题

- **改动局部性**：怎么保证"只改要求改的"——训练数据构造（编辑对的质量）+ 结构上的条件注入方式
- **身份一致性**：人物/IP 跨编辑保持（角色一致性是商业需求第一位）
- **语义级 vs 外观级编辑**：换风格/转视角（语义）与增删物体（外观）对模型能力要求不同
- **文字编辑**：图中文字的精确替换（Qwen-Image-Edit 强项，源于 VLM 文本编码器 + 双路条件设计）

## 3. 与统一模型的关系

- 编辑是"理解 + 生成"的交叉点：模型必须先看懂原图（理解），再重建大部分内容（生成）
- 因此统一模型（GPT Image、Nano Banana、BAGEL、InternVL-U）在编辑上有天然优势——这是 07 章的伏笔

## 4. 评测

- GEdit-Bench、ImgEdit、arena 类盲测
- 人工评估三维度：指令遵循 / 非编辑区保真 / 整体质量

## 关键论文清单

| 论文 | 年份 | 为什么读 |
|---|---|---|
| InstructPix2Pix | 2022 | 指令编辑开山，数据合成思路影响至今 |
| FLUX.1 Kontext | 2025 | in-context 编辑代表作 |
| Qwen-Image-Edit 技术报告 | 2025 | 双路条件 + 文字编辑 |
| Nano Banana / GPT Image 系博客与 system card | 2025–2026 | 闭源能力边界参考 |
