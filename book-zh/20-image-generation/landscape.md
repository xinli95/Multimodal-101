# Landscape · 图像生成

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

## 生成 · 开源 / 开放权重

| 模型 | 机构 | 规模 | License | 亮点 |
|---|---|---|---|---|
| **FLUX.2 [dev]** | Black Forest Labs | 32B | 非商用（商用需授权） | 开放权重最强画质档 |
| **FLUX.2 [klein]** | Black Forest Labs | 4B/9B | Apache 2.0 | 2026-01 发布；消费级 GPU（~13GB VRAM）0.5 秒内出图，**教学首选** |
| **Qwen-Image-2512** | 阿里 | 20B | Apache 2.0 | 盲测最强开源；中英文字渲染突出 |
| SD 3.5 | Stability AI | 2.5B–8B | 社区 license | 生态最广的基线 |
| HiDream-I1 | HiDream | 17B | MIT | 备选 |

## 生成 · 闭源前沿

| 模型 | 机构 | 亮点 |
|---|---|---|
| **GPT Image 2** | OpenAI | arena 第一；密集文字渲染、精确指令编辑、10 张参考图合成 |
| Nano Banana Pro (Gemini image) | Google | 指令编辑与世界知识结合 |
| Midjourney V8 | Midjourney | 美学天花板 |
| MAI-Image-2.5 | 微软 | arena 前列 |

⚠️ 已退役：Imagen 4 API 2026-08-17 关停；DALL-E 3 已被 GPT Image 系替代。

## 生成 · 趋势判断

1. 开源与闭源差距已缩小到"leaderboard 位次"而非"能用不能用"；开源赢在可控、私有化、成本。
2. 文本编码器 LLM 化（世界知识注入）是当前一代的共同选择。
3. 生成与编辑边界消失——新模型默认同时支持 T2I 和 I2I（详见本页编辑部分）。

---
## 编辑 · 开源 / 开放权重

| 模型 | 机构 | License | 亮点 |
|---|---|---|---|
| **Qwen-Image-Edit**（最新 2511+ 版本） | 阿里 | Apache 2.0 | 文字精确编辑第一；外观+语义编辑全能，开源首选 |
| **FLUX.2 系（编辑内建）** | BFL | klein: Apache 2.0 | FLUX.2 起生成/编辑一体，多参考图；Kontext [dev] 为上一代 |
| Step1X-Edit、SeedEdit（开源版） | 阶跃/字节 | 各异 | 备选 |

## 编辑 · 闭源前沿

| 模型 | 机构 | 亮点 |
|---|---|---|
| **Nano Banana Pro** | Google (Gemini image) | 指令编辑综合最强之一；角色一致性、世界知识 |
| **GPT Image 2** | OpenAI | 精确局部编辑 + 最多 10 张参考图合成 |
| SeedEdit / Seedream | 字节 | 国内闭源代表 |

## 编辑 · 选型速查

| 场景 | 推荐 |
|---|---|
| 图中文字替换（尤其中文） | Qwen-Image-Edit |
| 本地私有化 + 可微调 | Qwen-Image-Edit（可 LoRA 定制编辑行为） |
| 角色一致性要求极高的商业内容 | Nano Banana Pro / GPT Image 2 |
| 亚秒级交互式编辑 | FLUX.2 [klein] |

## 编辑 · 趋势判断

1. "编辑"正在被吸收进"生成"：单独的编辑模型会消失，全部并入统一多模态模型。
2. 微调开源编辑模型（如 Qwen-Image-Edit + LoRA）在垂直场景可超过闭源通用模型——教程实验点。
