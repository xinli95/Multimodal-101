# 03 · Image Generation — 图像生成

生成侧的第一章。理论上走完 DDPM → Latent Diffusion → DiT → Flow Matching 这条演进线（00 章已铺垫），实践上本地跑通 FLUX.2 [klein]，并与 GPT Image 2 等闭源 API 对比。

## 学习目标

- 完整理解现代文生图模型的三件套：VAE、文本编码器、DiT 骨干
- 理解 Classifier-Free Guidance、采样器与步数蒸馏（为什么 klein 能 0.5 秒出图）
- 本地跑通 FLUX.2 [klein] / SD 3.5，体验 prompt 工程与参数调节
- 了解 LoRA 微调原理并训练一个自己的风格 LoRA

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks](notebooks/README.md) — 实践代码
