# Landscape · 图像生成

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

## 开源 / 开放权重

| 模型 | 机构 | 规模 | License | 亮点 |
|---|---|---|---|---|
| **FLUX.2 [dev]** | Black Forest Labs | 32B | 非商用（商用需授权） | 开放权重最强画质档 |
| **FLUX.2 [klein]** | Black Forest Labs | 4B/9B | Apache 2.0 | 2026-01 发布；消费级 GPU（~13GB VRAM）0.5 秒内出图，**教学首选** |
| **Qwen-Image-2512** | 阿里 | 20B | Apache 2.0 | 盲测最强开源；中英文字渲染突出 |
| SD 3.5 | Stability AI | 2.5B–8B | 社区 license | 生态最广的基线 |
| HiDream-I1 | HiDream | 17B | MIT | 备选 |

## 闭源前沿

| 模型 | 机构 | 亮点 |
|---|---|---|
| **GPT Image 2** | OpenAI | arena 第一；密集文字渲染、精确指令编辑、10 张参考图合成 |
| Nano Banana Pro (Gemini image) | Google | 指令编辑与世界知识结合 |
| Midjourney V8 | Midjourney | 美学天花板 |
| MAI-Image-2.5 | 微软 | arena 前列 |

⚠️ 已退役：Imagen 4 API 2026-08-17 关停；DALL-E 3 已被 GPT Image 系替代。

## 趋势判断

1. 开源与闭源差距已缩小到"leaderboard 位次"而非"能用不能用"；开源赢在可控、私有化、成本。
2. 文本编码器 LLM 化（世界知识注入）是当前一代的共同选择。
3. 生成与编辑边界消失——新模型默认同时支持 T2I 和 I2I（详见 04 章）。
