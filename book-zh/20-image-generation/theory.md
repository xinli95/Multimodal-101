# 理论大纲 · 图像生成

## 1. 现代文生图的标准三件套

1. **VAE**：像素 ↔ 潜空间（8x/16x 下采样），生成发生在潜空间
2. **文本编码器**：CLIP/T5 → 新趋势是直接用 LLM/VLM（FLUX.2 用 Mistral-3 VLM、Qwen-Image 用 Qwen2.5-VL），世界知识直接注入条件
3. **骨干网络**：U-Net（SD1.5/SDXL 时代）→ **MMDiT**（SD3/FLUX：文本与图像 token 双流联合注意力）

## 2. 训练目标演进

- DDPM 噪声预测 → v-prediction → **Rectified Flow / Flow Matching**（SD3、FLUX、Qwen-Image 全用它）
- 直观理解：把"弯曲的去噪轨迹"拉直，训练更稳、少步数采样质量更高

## 3. 推理与加速

- 采样器族谱：DDIM → DPM-Solver → Flow Euler
- **步数蒸馏**：LCM、Turbo/Lightning、对抗蒸馏 → FLUX.2 [klein] 亚秒级出图的原理
- CFG 与 guidance 蒸馏（FLUX dev 系把 CFG 蒸进模型，推理省一半算力）

## 4. 可控性与微调

- **LoRA**：低秩适配，风格/人物定制的事实标准
- ControlNet / IP-Adapter：结构控制与图像参考（在 DiT 时代逐渐被上下文条件替代，见下方 §6）
- 文字渲染为什么难：tokenizer 粒度 + 训练数据；GPT Image 2 / Qwen-Image 在此的突破

## 5. 评测

- 自动指标的局限：FID 已基本失效；GenEval / DPG-Bench（组合性）、arena 盲测（人类偏好）成主流
- 关键维度：prompt 遵循、文字渲染、多主体一致性、美学

## 6. 编辑：范式演进

- **Inpainting 时代**：用户画 mask + 局部重绘（SD inpainting）
- **InstructPix2Pix (2022)**：首次纯指令编辑——用 GPT-3 + P2P 合成 (原图, 指令, 结果) 三元组训练
- **In-context 条件时代**（FLUX Kontext、Qwen-Image-Edit、GPT Image、Nano Banana）：参考图编码为 token 序列，与目标图在 DiT 中联合注意力，天然支持多参考图

## 7. 编辑：核心技术问题

- **改动局部性**：怎么保证"只改要求改的"——训练数据构造（编辑对的质量）+ 结构上的条件注入方式
- **身份一致性**：人物/IP 跨编辑保持（角色一致性是商业需求第一位）
- **语义级 vs 外观级编辑**：换风格/转视角（语义）与增删物体（外观）对模型能力要求不同
- **文字编辑**：图中文字的精确替换（Qwen-Image-Edit 强项，源于 VLM 文本编码器 + 双路条件设计）

## 8. 编辑与统一模型

- 编辑是"理解 + 生成"的交叉点：模型必须先看懂原图（理解），再重建大部分内容（生成）
- 因此统一模型（GPT Image、Nano Banana、BAGEL、InternVL-U）在编辑上有天然优势——这是 23 章的伏笔

## 9. 编辑：评测

- GEdit-Bench、ImgEdit、arena 类盲测
- 人工评估三维度：指令遵循 / 非编辑区保真 / 整体质量

## 关键论文清单

| 论文 | 年份 | 为什么读 |
|---|---|---|
| DDPM | 2020 | 起点 |
| Latent Diffusion | 2022 | SD 的由来 |
| DiT | 2022 | Transformer 骨干 |
| SD3 (MMDiT + RF) | 2024 | 现代架构定型之作 |
| FLUX.1/.2 技术资料 | 2024–2026 | 当前开源最强实践 |
| Qwen-Image 技术报告 | 2025 | 中文文字渲染 + VLM 做文本编码器 |
| InstructPix2Pix | 2022 | 指令编辑开山，数据合成思路影响至今 |
| FLUX.1 Kontext | 2025 | in-context 编辑代表作 |
| Qwen-Image-Edit 技术报告 | 2025 | 双路条件 + 文字编辑 |
| Nano Banana / GPT Image 系博客与 system card | 2025–2026 | 闭源能力边界参考 |
