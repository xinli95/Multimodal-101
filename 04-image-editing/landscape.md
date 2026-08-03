# Landscape · 图像编辑

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

## 开源 / 开放权重

| 模型 | 机构 | License | 亮点 |
|---|---|---|---|
| **Qwen-Image-Edit**（最新 2511+ 版本） | 阿里 | Apache 2.0 | 文字精确编辑第一；外观+语义编辑全能，开源首选 |
| **FLUX.2 系（编辑内建）** | BFL | klein: Apache 2.0 | FLUX.2 起生成/编辑一体，多参考图；Kontext [dev] 为上一代 |
| Step1X-Edit、SeedEdit（开源版） | 阶跃/字节 | 各异 | 备选 |

## 闭源前沿

| 模型 | 机构 | 亮点 |
|---|---|---|
| **Nano Banana Pro** | Google (Gemini image) | 指令编辑综合最强之一；角色一致性、世界知识 |
| **GPT Image 2** | OpenAI | 精确局部编辑 + 最多 10 张参考图合成 |
| SeedEdit / Seedream | 字节 | 国内闭源代表 |

## 选型速查

| 场景 | 推荐 |
|---|---|
| 图中文字替换（尤其中文） | Qwen-Image-Edit |
| 本地私有化 + 可微调 | Qwen-Image-Edit（可 LoRA 定制编辑行为） |
| 角色一致性要求极高的商业内容 | Nano Banana Pro / GPT Image 2 |
| 亚秒级交互式编辑 | FLUX.2 [klein] |

## 趋势判断

1. "编辑"正在被吸收进"生成"：单独的编辑模型会消失，全部并入统一多模态模型。
2. 微调开源编辑模型（如 Qwen-Image-Edit + LoRA）在垂直场景可超过闭源通用模型——教程实验点。
