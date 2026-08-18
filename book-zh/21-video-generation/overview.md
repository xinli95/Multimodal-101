# 21 · Video Generation — 视频生成

**与 Part I 的关系。** [05 章](../05-audio-and-video/index.md)讲的是视频"进"，而最惊人的地方是它几乎不需要新机件：抽 32 帧、每帧过一遍图像塔、把 `MM:SS` 时间戳当文本写进 prompt。没有时序注意力，没有 3D 卷积。看懂一段视频，说到底大半是按顺序看懂几张图。

生成一段视频则不然。当你必须**产出**第 17 帧而不只是读它，时间一致性就不再免费 —— 本章的一切都是为了把它买回来。这个不对称是从 Part I 带过来最有用的一条：会读视频的模型可以靠抽帧蒙混过关，会写视频的模型不行。

术语仍然可以迁移。Video DiT 对 latent 做 patchify，和 [04 章](../04-vision-tower/index.md)视觉塔对图像做 patchify 是同一件事，然后同样在得到的网格上跑 Transformer —— 只是现在网格多了一根时间轴，注意力的账也随之增长。

## 学习目标

- 理解视频相对图像新增的三个问题：时间一致性、算力爆炸、音画同步
- 掌握 3D Causal VAE 与时空注意力的设计取舍（完整 3D vs 分解式）
- 理解为什么 I2V（图生视频）是生产环境的主力路径
- 本地跑通开源视频模型（短片、低分辨率），并与 Veo API 对比

## 内容

- [theory.md](theory.md) — 原理与关键论文
- [landscape.md](landscape.md) — 当前格局（含 last-verified 日期）
- [notebooks](notebooks/README.md) — 实践代码
