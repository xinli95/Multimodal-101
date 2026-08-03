# 理论大纲 · 多模态基础

## 1. 模态与表示

- 什么是模态：文本 / 图像 / 视频 / 音频的原始数据形态与信息密度差异
- 万物皆 token：
  - 文本 → BPE/SentencePiece 子词
  - 图像 → ViT patch embedding（理解侧）；VAE latent + patchify（生成侧）
  - 视频 → 时空 patch（spacetime patches）
  - 音频 → 梅尔频谱帧（理解侧）；神经 codec 离散 token（生成侧，如 EnCodec / SoundStream）
- 关键论文：ViT (Dosovitskiy 2020)、EnCodec (Défossez 2022)

## 2. 跨模态对齐：对比学习

- CLIP (Radford 2021)：双塔结构 + InfoNCE，batch 内负样本
- SigLIP (Zhai 2023)：sigmoid loss 替代 softmax，解耦 batch size 与效果
- 为什么 CLIP 系编码器成为几乎所有 VLM 和文生图模型的视觉底座
- 局限：细粒度感知弱、文本长度受限 → 引出后续章节的解决方案

## 3. 三大生成范式

### 3.1 自回归（Autoregressive）
- 建模 p(x_t | x_<t)，逐 token 采样
- 优势：与 LLM 同构、天然支持交错多模态序列；劣势：图像/音频 token 序列长、误差累积
- 代表：GPT 系、图像侧 VQ-GAN + AR、音频侧 codec LM

### 3.2 扩散（Diffusion）
- 前向加噪 / 反向去噪；DDPM (Ho 2020) → DDIM 加速采样
- Latent Diffusion (Rombach 2022)：在 VAE 潜空间做扩散，算力骤降，Stable Diffusion 由此而来
- Classifier-Free Guidance：无需分类器的条件控制

### 3.3 流匹配（Flow Matching）
- Rectified Flow / Flow Matching (Lipman 2022, Liu 2022)：学习从噪声到数据的直线化速度场
- 为什么 2024 年后新模型（SD3、FLUX、Wan、LTX）几乎全部转向 flow matching：训练更稳、采样步数更少
- 与 diffusion 的统一视角：都是学习一个从先验到数据分布的连续变换

## 4. 架构收敛：Transformer 一统

- U-Net → DiT (Peebles & Xie 2022)：扩散模型的骨干也变成 Transformer，scaling law 得以适用
- 结论先行：后续章节所有模型（VLM、图像、视频、音频、Omni）本质都是 "Transformer + 模态特定的 tokenizer/detokenizer"

## 推荐阅读顺序

1. CLIP 论文（精读）
2. DDPM → LDM（理解损失函数即可，推导可跳过）
3. DiT + Flow Matching（重点：为什么业界收敛于此）
