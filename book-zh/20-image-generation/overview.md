# 20 · Image Generation — 图像生成与编辑

**与 Part I 的关系。** Gemma 4 是像素进、文本出；本章把箭头反过来：文本进、像素出。术语的迁移比想象中多——DiT 对 latent 做 patchify，和 Gemma 4 视觉塔对图像做 patchify 是同一件事，然后同样在 token 网格上跑 Transformer。差别在于目标函数（去噪/传输，而非预测下一个 token），以及最后必须把 token 网格还原成像素。

生成与编辑原本是两章，但 FLUX.2 和 Qwen-Image-Edit 之后它们已经是同一个模型，所以这里合成一章。理论上走完 DDPM → Latent Diffusion → DiT → Flow Matching，再讲指令式编辑与 in-context 条件；实践上本地跑通 FLUX.2 [klein]，并与闭源 API 对比。

## 学习目标

- 完整理解现代文生图模型的三件套：VAE、文本编码器、DiT 骨干
- 理解 Classifier-Free Guidance、采样器与步数蒸馏（为什么 klein 能 0.5 秒出图）
- 理解从 inpainting（画 mask）到指令编辑（纯语言）的范式转变，以及 in-context 条件：参考图作为序列条件与目标图联合注意力
- 抓住编辑的两个硬问题：改动局部性（不该动的别动）与身份一致性
- 本地跑通 FLUX.2 [klein] / SD 3.5，体验 prompt 工程与参数调节，并训练一个自己的风格 LoRA

## 内容

- [theory.md](theory.md) — 原理与关键论文（§1–5 生成，§6–9 编辑）
- [flux2-klein-deep-dive.md](flux2-klein-deep-dive.md) — 沿 Diffusers 源码逐层拆解 FLUX.2 [klein] 4B
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks](notebooks/README.md) — 实践代码
