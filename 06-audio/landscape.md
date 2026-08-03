# Landscape · 语音与音频

> **Last verified: 2026-08-02** — 超过 6 个月请重新核实。

## ASR（开源）

| 模型 | 机构 | License | 适用 |
|---|---|---|---|
| Whisper Large V3 / Turbo | OpenAI | MIT | 多语言默认选择 |
| Parakeet TDT | NVIDIA | CC-BY-4.0 | 低延迟流式 |
| Distil-Whisper | HF | MIT | 速度优先 |
| Moonshine | Useful Sensors | MIT | 端侧最小 |

## TTS（开源 / 开放权重）

| 模型 | 机构 | 规模 | License | 亮点 |
|---|---|---|---|---|
| **Kokoro-82M** | 社区 | 82M | Apache 2.0 | CPU 可跑、TTS Arena 顶级，**教学首选** |
| **Higgs Audio v3** | Boson AI | 4B | ⚠️ 研究/非商用（商用另签） | 2026-06 发布；chat-native、流式（句中即出声）、100+ 语言克隆 |
| **Fish Audio S2 Pro** | Fish Audio | - | 开放权重（查最新条款） | EmergentTTS-Eval 第一，超 ElevenLabs |
| Chatterbox-Turbo | Resemble | - | MIT | 生产级低延迟 |
| Orpheus / Dia2 | Canopy / Nari | 3B / - | Apache 2.0 | 情感控制 / 对话体 |

## 闭源

- ElevenLabs（商用生态最全）、OpenAI TTS、Google/Gemini TTS、MiniMax-Speech、Seed-TTS（字节）

## 音乐生成

- 闭源：Suno、Udio；开源：YuE（歌声+伴奏）、ACE-Step

## 选型速查

| 场景 | 推荐 |
|---|---|
| 本地免费快速上手 | Whisper + Kokoro（全 🟢 CPU 可跑） |
| 语音 agent（流式、打断） | Higgs Audio v3（注意商用 license）/ Chatterbox-Turbo |
| 最高音质多语言克隆 | Fish Audio S2 Pro / ElevenLabs |

## 趋势判断

1. 开源 TTS 已在盲测中超越部分闭源大厂——音频是开源追平最彻底的模态。
2. "TTS 模型"正演变为"会说话的 LLM"（chat-native，如 Higgs v3 支持副语言控制、上下文感知语气）。
3. 独立 ASR/TTS 终将被 omni 模型吸收，但生产中级联方案因可控性仍占主流。
