# 22 · Audio Generation — 语音与音频生成

**与 Part I 的关系。** 05 章讲的是音频"进"：波形 → mel 特征 → soft token → 进入 prompt。本章把箭头反过来——文本进、波形出——而且这里的对称性比图像那边更明显。现代 TTS 本质上是一个跑在 **codec token** 上的语言模型：先用神经音频编解码器把声音离散化，再像 LLM 预测下一个词那样预测下一个音频 token。Gemma 4 的音频塔和 codec 是同一个问题（怎么把连续声音变成有限字母表）的两种答案，只是优化目标相反：一个为了理解，一个为了重建。

ASR 本身已经移到 [05 章](../05-audio-and-video/index.md)，在那里 Whisper 作为 Gemma 4 音频编码器的对照组出现。本章只讲生成。

## 学习目标

- 理解神经音频 codec（EnCodec 一系）：音频如何变成离散 token，RVQ 在做什么
- 理解现代 TTS = "codec token 上的语言模型"，以及零样本声音克隆为什么自然而然
- 理解为"重建"优化的 codec 保留了什么，而为"理解"优化的编码器又丢掉了什么
- 串起本地 STT + TTS 回环成为可用的语音对话管线，并测清楚延迟到底花在哪

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks](notebooks/README.md) — 实践代码
