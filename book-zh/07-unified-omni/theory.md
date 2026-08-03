# 理论大纲 · 统一多模态模型

## 1. 为什么统一难

- 理解偏好**连续、语义化**的视觉特征（CLIP 系编码器）
- 生成偏好**细节保真**的表示（VQ token 或 VAE latent + diffusion）
- 一套表示既要"懂"又要"画"，天然冲突 → 所有架构差异都是对这个矛盾的不同回答

## 2. 四条架构路线

1. **纯自回归 + 离散视觉 token**（Chameleon、Emu3）：最优雅，图像质量长期吃亏
2. **解耦编码器**（Janus/Janus-Pro）：理解走 SigLIP、生成走 VQ，共享一个 LLM——简单有效的折中
3. **AR + Diffusion 混合**（BAGEL、GPT Image 推测同族）：LLM 负责理解与规划，diffusion expert 负责像素；BAGEL 的 MoT（Mixture-of-Transformer-Experts）让两者共享注意力上下文
4. **离散流匹配统一**（NExT-OMNI 等）：研究前沿,试图用一个训练目标覆盖所有模态

## 3. 语音实时交互的特殊架构

- **Thinker-Talker**（Qwen3-Omni/3.5-Omni）：Thinker 产生文本/推理，Talker 并行流式产出语音 token——解决"边想边说"
- 全双工对话：打断处理、双向流
- 与 06 章 codec LM 的衔接：Talker 生成的正是 codec token

## 4. 统一带来了什么（为什么值得做）

- **世界知识注入生成**：改图时"知道"东西该长什么样（Nano Banana 演示的核心）
- **跨模态推理链**：先想清楚再画（thinking-before-generation）
- **交错生成**:图文混排输出、视觉思维链
- 开放问题：统一是否真的互相增益？（ROVER 等 benchmark 显示互益尚不稳定）

## 5. 评测

- 理解侧沿用 MMMU 系；生成侧 GenEval/GEdit；统一专属：ROVER、Unison

## 关键论文清单

| 论文 | 年份 | 为什么读 |
|---|---|---|
| Chameleon | 2024 | 纯 AR 路线代表 |
| Janus / Janus-Pro | 2024–2025 | 解耦编码，教学最清晰 |
| BAGEL (Emerging Properties in Unified Multimodal Pretraining) | 2025 | MoT + 涌现能力分析，本章核心精读 |
| Qwen3-Omni / Qwen3.5-Omni 技术报告 | 2025–2026 | Thinker-Talker 实时全模态 |
| InternVL-U | 2026 | 4B 小体量统一模型，可本地复现 |
