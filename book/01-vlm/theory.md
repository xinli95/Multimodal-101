# 理论大纲 · VLM

## 1. 架构范式演进

- **双塔对齐**（CLIP，2021）：只能检索/分类，不能对话
- **冻结 LLM + 桥接**：
  - Flamingo (2022)：gated cross-attention 注入视觉特征
  - BLIP-2 (2023)：Q-Former 把视觉特征压缩成固定数量 query token
  - LLaVA (2023)：极简路线——一个 MLP projector 直接把 ViT 特征投进 LLM 词嵌入空间。**这条最简单的路线最终胜出**，值得思考为什么（数据质量 > 结构精巧）
- **现代标准架构**（Qwen-VL 系、InternVL 系）：SigLIP 类视觉塔 + MLP connector + 强 LLM，全参数联合训练

## 2. 关键技术点

- **动态/原生分辨率**：从固定 224px 到任意分辨率切块（Qwen2-VL 的 naive dynamic resolution、InternVL 的 tile 化），直接决定 OCR 与文档理解上限
- **视觉 token 压缩**：一张高清图动辄上千 token，压缩策略（pooling、pixel-shuffle、Q-Former）影响成本与细粒度感知的平衡
- **视频输入**：帧采样 + 时间戳编码（如 M-RoPE / 时间感知位置编码），长视频理解依赖长上下文
- **多模态位置编码**：Qwen 系的 M-RoPE——把位置编码拆成时间/高/宽三维

## 3. 训练流程（三阶段范式）

1. **对齐预训练**：冻结 LLM 与 ViT，只训 connector（图文对数据）
2. **多任务预训练/持续预训练**：全参放开，混合 OCR、grounding、VQA、interleaved 数据
3. **SFT + RL**：指令跟随、偏好对齐；2025 年后普遍加入 **多模态 RLVR**（可验证奖励），推理型 VLM（思维链看图）成为标配

## 4. 能力边界与评测

- 基准：MMMU（大学级多学科）、MMBench、OCRBench、DocVQA、Video-MME
- 已知短板：精细空间关系、计数、幻觉（描述不存在的物体）
- 前沿方向：GUI Agent（屏幕理解 + 操作）、3D grounding、超长视频

## 关键论文清单

| 论文 | 年份 | 为什么读 |
|---|---|---|
| CLIP | 2021 | 一切的起点 |
| Flamingo | 2022 | cross-attention 桥接路线 |
| BLIP-2 | 2023 | Q-Former 路线 |
| LLaVA / LLaVA-1.5 | 2023 | 极简路线胜出的证明 |
| Qwen2-VL → Qwen3-VL 技术报告 | 2024–2025 | 现代 SOTA 架构全貌 |
| InternVL3.5 技术报告 | 2025 | 开源另一极，级联 RL 训练 |
