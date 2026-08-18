# 理论大纲 · 音频生成

## 1. 音频表示

- 波形 → 梅尔频谱（理解侧标准输入）
- **神经 codec**：EnCodec / SoundStream / DAC —— RVQ（残差向量量化）把音频变成多层离散 token，~1.5kbps 仍保真
- codec 是音频生成的"VAE"：整个生成侧建立在它之上

## 2. 理解侧，一段话讲完

语音**识别**在 [05 章](../05-audio-and-video/index.md)，那里 Gemma 4 自己的分块注意力音频编码器是主角、Whisper 是设计空间对照组；模型清单在[理解侧 landscape](../landscape.md)。其中有两个事实与本章相关。Whisper 的鲁棒性来自 68 万小时弱监督数据而非架构 —— 是数据，不是设计。以及 ASR 正在被通用模型吸收：Gemma 4 E2B 原生能听，Qwen3.5-Omni 支持 113 种语言识别，所以"跑一个 ASR 模型"越来越是个延迟决策而非能力决策。

**不能**迁移的是表示。为理解而建的编码器会扔掉一切与辨认词语无关的东西；为生成而建的 codec 必须保留足够重建波形的信息。同一份输入，相反的目标 —— 这正是 §1 的 codec 与 05 章的 mel 前端长得毫无关系的原因。

## 3. TTS（语音合成）：范式演进

- 老范式：声学模型 + 声码器（Tacotron + HiFi-GAN），按说话人训练
- **现代范式：codec LM**（VALL-E 2023 开端）：
  - 文本 token + 参考音频 token → 自回归生成 codec token → codec 解码成波形
  - 零样本克隆 = 参考音频作为 prompt 的 in-context learning
- 变体：flow matching 声学路线（非自回归，快）、混合式
- 关键能力分化：流式低延迟（对话 agent）vs 表现力（情感/韵律控制，inline token 标记）

## 4. 语音对话与全双工

- 级联（ASR→LLM→TTS）vs 端到端语音对话（GPT-4o 式）
- 全双工：打断、backchannel——为什么这很难（23 章 Thinker-Talker 架构呼应）

## 5. 音乐与音效

- 音乐生成：Suno/Udio（闭源）、YuE、ACE-Step（开源）
- 音效/Foley：视频配音效（V2A），与 21 章音视频联合生成呼应

## 6. 评测

- ASR：WER/CER（LibriSpeech、Common Voice、多语言 FLEURS）
- TTS：EmergentTTS-Eval、Audio Turing Test、说话人相似度 + 自然度 MOS

## 关键论文清单

| 论文 | 年份 | 为什么读 |
|---|---|---|
| Whisper | 2022 | 数据驱动鲁棒 ASR |
| EnCodec | 2022 | 音频离散化基石 |
| VALL-E | 2023 | codec LM 范式开端 |
| Higgs Audio v3 技术资料 | 2026 | chat-native、流式 TTS 最新实践 |
