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
- ControlNet / IP-Adapter：结构控制与图像参考（在 DiT 时代逐渐被上下文条件替代，见 04 章）
- 文字渲染为什么难：tokenizer 粒度 + 训练数据；GPT Image 2 / Qwen-Image 在此的突破

## 5. 评测

- 自动指标的局限：FID 已基本失效；GenEval / DPG-Bench（组合性）、arena 盲测（人类偏好）成主流
- 关键维度：prompt 遵循、文字渲染、多主体一致性、美学

## 关键论文清单

| 论文 | 年份 | 为什么读 |
|---|---|---|
| DDPM | 2020 | 起点 |
| Latent Diffusion | 2022 | SD 的由来 |
| DiT | 2022 | Transformer 骨干 |
| SD3 (MMDiT + RF) | 2024 | 现代架构定型之作 |
| FLUX.1/.2 技术资料 | 2024–2026 | 当前开源最强实践 |
| Qwen-Image 技术报告 | 2025 | 中文文字渲染 + VLM 做文本编码器 |
