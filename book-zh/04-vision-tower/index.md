# 04 · Vision Tower — 从 patch 到 soft token

**在数据流中的位置**：`pixel_values ──► Gemma4VisionModel ──► pooler ──► Gemma4MultimodalEmbedder ──► soft tokens`

图像在这一章真正变成语言模型读得懂的东西。四个阶段，每个都有值得琢磨的设计决策：

1. **Patch embedder** — 每个 16×16 patch 做线性投影，再加上按 `(x, y)` 坐标查表得到的**学习式** 2D 位置嵌入（每轴 10,240 个槽位）
2. **Encoder** — 16 层 Transformer，用 **2D RoPE**：一半 head 维度按 x 坐标旋转，另一半按 y。"猫在狗左边"能被表示，靠的就是它
3. **Pooler** — **按位置**对 3×3 邻域做平均池化，把 2,520 个 patch 压成 280 个 soft token
4. **Multimodal embedder** — 从视觉隐藏维度投影到文本嵌入空间，于是结果可以像普通 embedding 一样塞进 token 序列

## 你会学到

1. 网格为什么需要两套位置信号（学习式绝对表 + 旋转式相对编码），各自买到了什么
2. 2D RoPE 如何靠切分 head 维度实现——以及怎么用十五行代码自己写一个
3. 按位置而非按序列下标池化，为什么能扛住可变长宽比与 padding
4. 投影进文本空间发生在哪个模块，以及它为什么是 LLaVA MLP projector 的直系后代
5. **设计空间**：LLaVA 的 MLP vs BLIP-2 的 Q-Former vs Flamingo 的 cross-attention vs Qwen-VL 的 M-RoPE + 原生分辨率 vs InternVL 的 tiling

## 源码地图

| `modeling_gemma4.py` 中的符号 | 作用 |
|---|---|
| `Gemma4VisionPatchEmbedder` | patch 投影 + 学习式 2D 位置表（`_position_embeddings`） |
| `Gemma4VisionRotaryEmbedding`、`apply_multidimensional_rope` | 2D RoPE：按轴旋转一半 head 维度 |
| `Gemma4VisionAttention` / `...EncoderLayer` / `...Encoder` | 主干堆叠 |
| `Gemma4VisionPooler`、`_avg_pool_by_positions` | 由 patch 坐标驱动的 3×3 池化 |
| `Gemma4MultimodalEmbedder` | 视觉维度 → 文本嵌入空间 |
| `Gemma4Model.get_image_features` | 一次调用跑完上面全部 |

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_vision_tower_anatomy.ipynb` | hook 住每个阶段打印形状；手写 `apply_multidimensional_rope` 并 `assert_close`；可视化哪些 patch 池化成了哪个 soft token | 🟢 CPU（迷你 config）/ 🟡 真权重 |
| `02_image_understanding.ipynb` | 真权重 E2B 下这座塔换来了什么：VQA、OCR、文档结构化抽取、grounding 画框 | 🟡 24GB 显存 |
| <a href="../../04-vision-tower/notebooks/03_compare_qwen3vl.html"><code>03_compare_qwen3vl.ipynb</code></a> | 设计空间对照组：Qwen3-VL 的原生分辨率 + 2×2 merge 在同一批任务上的表现，视觉 token 数并排对比 | 🟡 12GB+ 显存 |

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/04-vision-tower/index.html)。
