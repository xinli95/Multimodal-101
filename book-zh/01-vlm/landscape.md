# Landscape · VLM

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

## 开源阵营

| 模型 | 机构 | 规模 | License | 亮点 |
|---|---|---|---|---|
| **Qwen3-VL** | 阿里 | 2B → 235B-A22B (MoE) | Apache 2.0 | 开源旗舰，benchmark 对标 Gemini 2.5 Pro / GPT-5；OCR、grounding、视频、agent 全面 |
| **InternVL3.5** | 上海AI Lab | 1B → 241B-A28B | MIT/Apache（分尺寸） | 开源 SOTA 之一，推理与效率并重 |
| Llama 4 multimodal | Meta | 多尺寸 MoE | Llama license | 生态大，英文强 |
| Molmo | AI2 | 1B–72B | Apache 2.0 | 数据全开放，pointing 能力独特 |
| Pixtral | Mistral | 12B/124B | Apache 2.0 | 欧洲阵营代表 |
| Phi-4 multimodal | 微软 | ~5.6B | MIT | 端侧小模型代表 |

**教学推荐**：本地实践用 Qwen3-VL 的小尺寸（🟡 消费级 GPU）；架构学习读 Qwen3-VL + InternVL3.5 两份技术报告即可覆盖主流设计。

## 闭源前沿

| 模型 | 机构 | 多模态特点 |
|---|---|---|
| Gemini 3.x Pro | Google | 统一多模态架构，原生处理文本/图像/音频/视频，公认多模态理解最强 |
| GPT-5.x | OpenAI | 原生多模态输入，推理链强 |
| Claude (Opus/Sonnet/Fable) | Anthropic | 图像输入 + 文档理解，工程任务强 |

## 趋势判断

1. 开源与闭源在标准 benchmark 上已基本拉平，闭源优势集中在长尾鲁棒性与视频/音频原生融合。
2. "思考型 VLM"（多模态 CoT + RLVR）已是新模型默认配置。
3. GUI/Computer-use agent 是 VLM 落地增长最快的场景。
4. 关注 Qwen3.8（>1T 参数多模态，2026-07 WAIC 预览，承诺开源未放权重）。
