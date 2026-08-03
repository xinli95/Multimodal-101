# 01 · VLM — 视觉语言模型

多模态理解的主战场：让 LLM "看懂"图像和视频。本章讲清楚主流 VLM 的架构范式（视觉编码器 + Connector + LLM）、训练流程，并实际跑通开源旗舰模型与闭源 API 的对比。

## 学习目标

- 画出一个现代 VLM 的完整数据流：图像 → ViT → connector → LLM 上下文
- 理解三种 connector 设计（Linear/MLP、Q-Former、Cross-Attention）的取舍
- 理解动态分辨率（native resolution）为什么是 OCR/文档能力的关键
- 本地跑通 Qwen3-VL 小尺寸模型，与 Gemini / GPT-5 API 在同一组任务上对比

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks](notebooks/README.md) — 实践代码
