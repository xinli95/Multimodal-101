# 05 · Video Generation — 视频生成

图像生成向时间维的扩展：Video DiT、3D Causal VAE、时空注意力。实践本地跑 Wan 2.2 / HunyuanVideo 1.5，闭源侧调 Veo 3.1 API，并理解为什么视频是当下算力最烧钱的模态。

## 学习目标

- 理解视频生成相对图像新增的三个问题：时间一致性、算力爆炸、音视频同步
- 掌握 3D Causal VAE 与时空注意力（full 3D vs 分解式）的设计取舍
- 理解 I2V（图生视频）为何是生产中的主力路径
- 本地跑通一个开源视频模型（低分辨率短片），调用 Veo API 对比

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks/](notebooks/) — 实践代码
