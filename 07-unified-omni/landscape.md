# Landscape · 统一多模态模型

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

## 开源 / 开放权重

| 模型 | 机构 | 规模 | License | 覆盖 |
|---|---|---|---|---|
| **Qwen3.5-Omni** | 阿里 | 30B MoE 起 | Apache 2.0（开源版） | 输入文/图/音/视频，输出文+语音；256k 上下文、113 语言 ASR；音频任务超 Gemini 3.1 Pro |
| **BAGEL** | 字节 Seed | 7B 激活/14B | Apache 2.0 | 理解+生成+编辑；MoT 架构；配套 Hyper-BAGEL 加速（无 v2，仍是 2025 原版） |
| **InternVL-U** | 上海AI Lab | 4B | MIT | 2026-03 发布；理解/推理/生成/编辑一体，**本地实践首选**（体量小） |
| Janus-Pro | DeepSeek | 1B/7B | MIT | 2025 年初模型，作架构教学用，不作 SOTA 用 |
| NExT-OMNI | 研究项目 | - | 开源 | 离散流匹配 any-to-any，前沿参考 |

## 闭源前沿

| 模型 | 机构 | 形态 |
|---|---|---|
| GPT-5.x + GPT Image 2 | OpenAI | 原生多模态输入 + 统一图像生成/编辑 |
| Gemini 3.x + Nano Banana Pro | Google | 出生即统一架构，全模态输入输出 |
| Qwen3.8-Max (preview) | 阿里 | >1T 参数多模态，2026-07 预览，承诺开源未兑现 |

## 趋势判断

1. 闭源前沿已经全部是统一模型；开源在 2026 年密集跟进（InternVL-U、Qwen-Omni 系）。
2. "先推理后生成"（thinking-before-generation）正在成为生成质量的新杠杆。
3. 实时全双工语音交互是 omni 模型的主要落地形态（语音助手、陪伴、客服）。
4. 理解与生成互益尚未完全证实——留意 ROVER/Unison 类 benchmark 的后续结论。
