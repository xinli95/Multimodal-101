# Landscape · 基础组件

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

## 视觉编码器（VLM / 生成模型的底座）

| 模型 | 机构 | License | 备注 |
|---|---|---|---|
| SigLIP 2 | Google | Apache 2.0 | 当前开源 VLM 最常用的视觉塔 |
| CLIP (ViT-L/14) | OpenAI | MIT | 经典基线，教学首选 |
| DINOv2/v3 | Meta | Apache 2.0 | 自监督路线，密集预测任务强 |

## 文本编码器（生成模型条件侧）

- T5/Flan-T5 系仍广泛用于文生图条件编码；新一代模型（FLUX.2、Qwen-Image）趋势是直接用 LLM/VLM 作为文本编码器。

## 范式现状小结

- **理解侧**：自回归 LLM + 视觉编码器是绝对主流（见 01 章）。
- **图像/视频生成**：Flow Matching + DiT 已成标配（见 03/05 章）。
- **音频生成**：codec token 自回归 LM 为主流（见 06 章）。
- **统一模型**：AR + diffusion 混合（BAGEL）与离散流匹配（NExT-OMNI）均在探索（见 07 章）。
