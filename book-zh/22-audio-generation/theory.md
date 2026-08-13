# 理论大纲 · 语音与音频

## 1. 音频表示

- 波形 → 梅尔频谱（理解侧标准输入）
- **神经 codec**：EnCodec / SoundStream / DAC —— RVQ（残差向量量化）把音频变成多层离散 token，~1.5kbps 仍保真
- codec 是音频生成的"VAE"：整个生成侧建立在它之上

## 2. ASR（语音识别）

- **Whisper (2022)**：encoder-decoder + 68 万小时弱监督数据，鲁棒性来自数据而非结构
- 流式路线：CTC / RNN-T / TDT（NVIDIA Parakeet），低延迟场景必需
- 蒸馏与加速：Distil-Whisper、faster-whisper（CTranslate2）
- 新趋势：ASR 被吸收进 omni 模型（Qwen3.5-Omni 支持 113 种语言识别）

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
