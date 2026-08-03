# 06 · Audio — 语音与音频

理解侧讲 ASR（Whisper 架构为主线），生成侧讲 TTS 的 codec token LM 范式——这也是理解 07 章 Omni 模型"如何说话"的前置知识。实践覆盖 Whisper、Kokoro、Higgs Audio v3。

## 学习目标

- 理解 ASR 两大路线：encoder-decoder（Whisper）vs CTC/transducer（流式）
- 理解神经音频 codec（EnCodec 系）：音频如何变成离散 token
- 理解现代 TTS = "codec token 上的语言模型"，以及零样本声音克隆的原理
- 跑通本地 STT + TTS 闭环，构建一个语音对话 pipeline

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks](notebooks/README.md) — 实践代码
