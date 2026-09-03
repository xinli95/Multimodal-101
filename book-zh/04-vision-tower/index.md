# 04 · Vision Tower — 从 patch 到 soft token

03 章停在把图像变成 patch 向量和坐标的位置。这些数值仍然只描述一块块局部颜色。本章继续问：**Gemma 4 如何把局部 patch 向量变成带有上下文的视觉特征，再让它们拥有语言模型所需的 embedding width？**

```text
03 章：image processor
pixel_values + image_position_ids
              │
              ▼
┌──────────────────────────────────────────────┐
│ Gemma4VisionModel                           │
│                                              │
│ patch embedder → vision encoder → 3×3 pooler │
└──────────────────────────────────────────────┘
              │ 带上下文的视觉特征
              ▼
Gemma4MultimodalEmbedder
RMSNorm → 线性投影到 text width
              │
              ▼
语言模型可以接收的 visual embeddings
              │
              ▼
08 章：把它们放入 prompt 中预留的位置
```

**Vision tower** 是 `Gemma4VisionModel`：patch embedder、Transformer encoder 和 pooler。**Connector** 是紧随其后的 `Gemma4MultimodalEmbedder`：一个 RMSNorm 加一个 learned linear projector。先把这条边界划清，整个架构会好懂很多。

## 你会学到

1. `pixel_values` 的一行在进入模型前究竟是什么
2. patch embedding 和 vision encoder 如何把局部像素变成带上下文的特征
3. 图像 patch 为什么需要位置信息——用「绝对地址」与「相对几何」理解
4. Gemma 4 为什么等 encoder 做完以后，才对 3×3 patch 分组做 pooling
5. `Gemma4MultimodalEmbedder` 如何把视觉特征投影到 text embedding space
6. 这套 connector 与 LLaVA projector、BLIP-2 Q-Former、Flamingo cross-attention 有什么区别

## 先跟一张图走完整条路径

先用一个贯穿全章的例子建立直觉，再读实现。在默认的 280-token 档位，03 章会把一张 640×480 图像 resize 成 912×672。它包含 57×42 的 patch 网格，也就是 2,394 个真实 patch。Processor 再把它 pad 到 2,520 行，让这个档位的所有图像拥有相同的输入形状。

对 E2B 模型，各阶段形状如下：

| 阶段 | 这张图的形状 | 发生了什么变化？ |
|---|---:|---|
| image processor 输出 | `[1, 2520, 768]` | 2,394 个真实 16×16 RGB patch，加上 126 个全零 padding 行 |
| patch embedder | `[1, 2520, 768]` | 每个 raw patch 变成 learned vision feature；在 E2B 上宽度恰好仍是 768 |
| 16 层 vision encoder | `[1, 2520, 768]` | 形状不变，但每个真实 patch 已经包含来自其他 patch 的上下文 |
| 3×3 pooler | `[1, 280, 768]` | 每 9 个相邻 patch feature 平均成一个候选 soft token |
| 移除池化后的 padding | `[266, 768]` | 只保留真实的 19×14 pooled grid |
| multimodal embedder | `[266, 1536]` | token 数仍为 266；feature width 从 vision width 变成 text width |

这张表就是全章的缩影：encoder 改变每一行的**含义**，pooler 改变**行数**，connector 改变每一行的**宽度**。

### 2,520 行到底是什么？

`2,520 = 280 × 3²`，是默认档位在 pooling 之前允许的最大 patch 数，并不是每张图都有 2,520 个真实 patch。在这个例子里：

- 第 0–2,393 行包含真实 patch 像素；
- 第 2,394–2,519 行是全零 padding；
- 对应的 `image_position_ids` 给第一组写入真实 `(x, y)` 坐标，给 padding 组写入 `(-1, -1)`。

Vision tower 会把「坐标是否等于 `(-1, -1)`」转换成 Boolean padding mask。Encoder 用它避免真实 patch attend 到 padding，pooler 用它避免全零行变成 visual token。所以 position tensor 在这里提供的是两个朴素的信息：真实 patch 来自哪里，以及某一行是不是 padding。

## 按顺序理解架构

### 1. Patch embedder：像素变成模型特征

Processor 输出的每一行都是一个摊平的 16×16 RGB 方块：

```text
16 × 16 × 3 = 768 个像素值
```

Patch embedder 做三件事：

```python
pixel_values = 2 * (pixel_values - 0.5)       # [0, 1] → [-1, 1]
hidden_states = self.input_proj(pixel_values) # raw patch → vision hidden width
hidden_states = hidden_states + position_embeddings
```

#### `[0, 1] → [-1, 1]` 算不算 normalization？

按普通数学语言来说，算：它是一次固定的 affine rescaling 和 centring。容易混淆的是 Transformers API 的命名。03 章说 `do_normalize=False`，意思是 **image processor** 没有做常见的逐 channel `(x - image_mean) / image_std`；它只把字节值除以 255。随后模型对所有 channel 统一执行 `2x - 1`。

所以这两个说法同时成立：

- processor 没有执行 ImageNet 风格的逐 channel mean/std normalization；
- learned projection 之前，模型仍把 `[0, 1]` 的值重新居中到 `[-1, 1]`。

`input_proj` 是一个没有 bias 的线性层。它接收一个 raw patch 中的 768 个数，为它产出一个 vision hidden width 的向量：E2B 为 768，大模型为 1,152。每个 patch 独立执行这一步。

原文里的「vision stem」只是指 **learned vision network 的入口**。ResNet 风格的 stem 会先运行多层卷积，并逐渐降低空间分辨率。Gemma 4 不这样做：03 章已经把图像切成 patch，因此 learned entrance 只有一次线性投影。「没有 convolution、ResNet、patch-merging pyramid」只是描述这项架构选择，并不是说这座 tower 不提取特征；接下来的 Transformer encoder 才负责主要的特征提取。

### 2. 位置：先给地址，再让 attention 理解几何

一个 patch 的 768 个颜色值不会说明它来自哪里。同一块蓝色可能是图像顶部的天空，也可能是底部的水面。二维网格被摊平成一维序列后，模型因此需要显式位置信息。

Gemma 4 用两种方式提供位置。第一次阅读时，可以这样理解：

| 位置信号 | 它帮助回答的直觉问题 | 在哪里进入计算？ |
|---|---|---|
| learned 2D position embedding | 「这个 patch 在图像的哪里？」 | 一次性加到 patch feature 上 |
| 2D RoPE | 「两个 patch 在横向和纵向上是什么关系？」 | 每一层 attention 都作用于 query 和 key |

Learned embedding 为 x 和 y 分别准备一张 lookup table。对坐标 `(x, y)`，Gemma 4 查出 `x_embedding[x]` 与 `y_embedding[y]`，相加后再加到 patch feature。Padding 坐标是 `(-1, -1)`，其 position embedding 会被 mask 成零。

在 self-attention 里，2D RoPE 用另一种方式利用相同的 x、y 坐标：每个 attention head 的一半维度携带横向位置，另一半携带纵向位置。这样，attention 会对空间位移敏感：两对 patch 如果横向间距相同，即便整体移动到图像的其他地方，也会得到相同的 x-relative rotation。

「绝对地址 versus 相对几何」是帮助第一次阅读的直觉，不是说训练完成后两个机制的职责还能被完美隔离。初读只需要抓住：learned embedding 让 patch 的初始表示知道位置；2D RoPE 让 attention 计算本身知道这是二维网格。精确的 frequency construction 和 `clamp-then-mask` 实现放在 [`01_vision_tower_anatomy.ipynb`](../../book/04-vision-tower/notebooks/01_vision_tower_anatomy.ipynb)；理解主架构不以它们为前提。

### 3. Vision encoder：让 patch 获得上下文

完成 patch embedding 后，每个 patch 知道自己的像素和位置，却仍不知道整张图里还有什么。Vision encoder 为它补上上下文。

E2B 使用 16 个 Transformer block；更大的 Gemma 4 版本使用 27 个。每个 block 都是熟悉的两段结构：

```text
patch features
    │
    ├─ self-attention ──► 在所有真实 patch 之间交换信息
    │
    └─ gated MLP ───────► 变换每个 patch feature
         两段外围都有 normalization 和 residual connection
```

Vision attention 不是 causal 的。同一层里，左上角的 patch 可以 attend 到右下角的 patch。重复这个过程，会把局部证据变成带上下文的证据：一个最初只包含棕色像素的 patch，在结合附近纹理、轮廓、面部和场景信息后，可以成为「狗耳朵的一部分」。

序列形状经过 encoder 时不会变化。E2B 上 `[1, 2520, 768]` 进入，`[1, 2520, 768]` 离开。这**不代表什么都没发生**：每个向量都经过 16 层 attention 和 MLP 重写。Padding 行为了保持规则的 batch shape 仍然存在，但 attention mask 不允许它们作为图像内容参与计算。

### 4. Pooler：九个带上下文的 patch feature 变成一个 soft token

如果把例子里的 2,394 个真实 patch feature 全部送进语言模型，会占用过多 context。Gemma 4 对每个空间 3×3 分组做 pooling 来缩短序列：

```text
57×42 contextual patch grid
          │ 每个 3×3 neighbourhood 取平均
          ▼
19×14 visual-token grid = 266 个真实 soft token
```

顺序很重要。Gemma 4 在 Transformer **之后** pooling，所以它平均的是已经看过整张图的 feature，而不是九个 raw pixel patch。Encoder 先得到细粒度证据，语言模型再接收更便宜的摘要。

为什么 pooler 要使用坐标？因为 padding 后的 batch 是一维张量，而「这九个 patch 相邻」是二维事实。把 `(x, y)` 分别整除 3，就能把每个 patch 分配到对应的 3×3 分组。`(-1, -1)` padding marker 防止 padding 行成为真实分组。Pooling 完成后，`pooler_mask` 会从 280 个输出上限中移除 14 个未使用输出，剩下上面算出的 266 个真实 soft token。

本质上，这就是普通 average pooling，只是换成一种能在同一 batch 中适配不同网格形状的写法。想看具体 one-hot matrix multiplication 的读者，可以在 anatomy notebook 里复现。

### 5. Multimodal embedder：把 vision width 投影到 text width

Pooler 已经产出了正确**数量**的 visual token，但向量仍使用 vision tower 的 hidden width 和表示空间。语言模型要求自己的 embedding width。`Gemma4MultimodalEmbedder` 就是二者之间的 learned adapter：

```python
class Gemma4MultimodalEmbedder(nn.Module):
    def forward(self, inputs_embeds):
        normalized = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normalized)
```

它只有两部分：

1. 一个没有 learned scale 的 RMSNorm，把输入 feature magnitude 放到稳定范围；
2. 一个没有 bias 的线性层，学习从 vision width 到 text width 的映射。

在 E2B 的例子里，形状从 `[266, 768]` 变成 `[266, 1536]`。Connector **不会**把 266 再压成更小的数量；它改变每个 token 的表示，使语言模型可以把它与 1,536 维 word embedding 一起接收。到 08 章，这 266 个向量会替换 02 章预留的 266 个 image placeholder 位置。

所以，最朴实的架构描述完全正确：Gemma 4 是一个 vision encoder，后接一个投影到 text space 的 projector。代码把这个 projector 叫作 `Gemma4MultimodalEmbedder`；「connector」是论文里更宽泛的叫法。

## 现在再读 `Gemma4VisionModel.forward`

建立各阶段以后，这段短代码就成了总结，而不是谜题：

```python
padding_positions = (pixel_position_ids == -1).all(dim=-1)
output_length = pixel_values.shape[-2] // self.config.pooling_kernel_size**2

# 1. raw patch pixels → 带位置的 vision features
hidden_states = self.patch_embedder(
    pixel_values, pixel_position_ids, padding_positions
)

# 2. 让每个真实 patch 获得上下文
hidden_states = self.encoder(
    inputs_embeds=hidden_states,
    attention_mask=~padding_positions,
    pixel_position_ids=pixel_position_ids,
).last_hidden_state

# 3. 空间 3×3 压缩，再丢弃池化后的 padding
hidden_states, pooler_mask = self.pooler(
    hidden_states, pixel_position_ids, padding_positions, output_length
)
hidden_states = hidden_states[pooler_mask]
```

随后，`Gemma4Model.get_image_features` 执行第 4 步：

```python
vision_outputs = self.vision_tower(
    pixel_values=pixel_values,
    pixel_position_ids=image_position_ids,
)
vision_outputs.pooler_output = self.embed_vision(
    inputs_embeds=vision_outputs.last_hidden_state
)
```

两个 class 的分工现在很清楚：`vision_tower` 理解并压缩图像，`embed_vision` 把结果映射到语言模型需要的 width。

## 之后复现源码时值得知道的细节

这些细节对源码复现有意义，但不是理解架构的主干：

- **大坐标表。** Learned x、y table 各有 10,240 个 entry，长宽比变化时无需 resize 一张固定正方形 position grid。
- **Q/K/V normalization。** Attention 会 normalize query、key 和 value；value norm 没有 learned scale。这是数值稳定性设计，不是一个新的架构阶段。
- **Clippable linear。** E2B 可以用 checkpoint 中的 bound 在 linear layer 前后 clamp activation；大模型关闭这条路径。
- **Float32 pooling。** Pooler 会把平均后的 feature 乘以 `√hidden_size`，因此用 float32 完成这一步，避免 fp16 overflow。
- **可选输出 standardization。** 大模型的 vision tower 会在 pooling 后使用 checkpoint 中的 bias 和 scale；E2B 不会。

第一个 notebook 使用迷你 CPU 模型逐项验证这些细节。

## 源码地图

| `modeling_gemma4.py` 中的符号 | 作用 |
|---|---|
| [`Gemma4VisionPatchEmbedder`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L579) | `[0,1] → [-1,1]`、patch projection、learned x/y embedding |
| [`Gemma4VisionAttention`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L917) | 带 2D RoPE 的 non-causal self-attention |
| [`Gemma4VisionEncoderLayer`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L986)、[`Gemma4VisionEncoder`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1030) | 16 或 27 层 Transformer stack |
| [`Gemma4VisionPooler`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L624) | 感知坐标的 3×3 average pooling |
| [`Gemma4VisionModel`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2017) | patch embedder + encoder + pooler |
| [`Gemma4MultimodalEmbedder`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2085) | RMSNorm + vision width 到 text width 的投影 |
| [`Gemma4Model.get_image_features`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2212) | vision tower 后接 connector |

## 设计空间：connector 到底连接什么？

03 章比较的是**图像采样策略**：固定画布、tiling、动态分辨率，以及 Gemma 4 的选定像素预算。这些设计决定多少 patch feature 到达 vision tower。本章比较紧接着的另一个决定：**vision feature 如何被压缩、映射，并呈现给语言模型？** 两处 design space 前后相连，但不是重复内容。

VLM 论文对「connector」这个词用得很宽松。它可能只指改变 feature width 的 projector，也可能包含 learned token compressor，甚至可能包括插入 LLM 内部的一整条 cross-attention path。比较时把三个问题分开：

1. 它会不会减少 visual token 数？
2. 它如何把 vision feature width 映射到 language feature width？
3. Visual vector 会被插入 text sequence，还是作为独立 memory 供 cross-attention 读取？

| 系列 | Vision encoder 与 LLM 之间发生什么？ | 对 token 数的影响 | Vision 如何进入 LLM | 主要取舍 |
|---|---|---|---|---|
| [**LLaVA**](https://arxiv.org/abs/2304.08485) | 原始版本对每个 patch feature 独立应用 linear，后续版本使用小 MLP | 基础 connector 不减 token；每个保留 patch 对应一个输出 | visual vector 加入 input embedding sequence | 最便宜、最容易训练，但 LLM 要为每个 visual patch 付 context 成本 |
| [**BLIP-2**](https://arxiv.org/abs/2301.12597) | 32 个 learned query 在 Q-Former 中 cross-attend 所有 image feature，再投影 32 个结果 | 稠密 patch sequence → 固定 32 个 | query 输出作为紧凑 visual prefix 来 condition 语言模型 | 压缩能按内容自适应，但增加一个可观的模块及其预训练目标 |
| [**Flamingo**](https://arxiv.org/abs/2204.14198) | Perceiver Resampler 为每个 image/video 生成 64 个 visual latent；LLM 内部多处插入 gated cross-attention | 可变 vision grid → 每个视觉输入固定 64 个 latent | text state 反复 cross-attend 独立 visual memory | 适合交错 image/video 与 few-shot prompt，但要修改 LLM 架构，并反复支付 cross-attention 成本 |
| [**Qwen2-VL**](https://arxiv.org/abs/2409.12191) | 拼接相邻 2×2 vision feature，再用 learned merger/MLP 映射 | 四个 patch feature → 一个 visual token | merged vector 进入 language sequence | 保留动态分辨率网格并进行 learned local merge；成本仍随图像面积变化 |
| [**InternVL 1.5**](https://arxiv.org/abs/2404.16821) | pixel shuffle 把 2×2 空间邻域移入 channel，再由 MLP 映射到 language width | 每个 tile 内四个 feature → 一个；总数仍随 tile 数增长 | 投影后的 tile 与 thumbnail vector 进入 language sequence | 保留高分辨率 tile 细节，但 tile 多时 visual prefix 仍然很长 |
| **Gemma 4** | 对 3×3 空间分组的 contextual feature 取平均，再做 RMSNorm + 一个 linear projector | 九个 patch feature → 一个 soft token，受所选档位限制 | projected vector 替换预留的 image position | 极便宜、可预测；统一平均不如 learned query 自适应 |

现在可以具体地比较：

- **LLaVA connector 主要对齐 feature width。** 基础 LLaVA 不要求 connector 判断哪些 patch 更重要。
- **BLIP-2 和 Flamingo 学出固定长度摘要。** Learned query 或 latent 决定保留什么，代价是更多容量与训练复杂度。
- **Qwen2-VL、InternVL 和 Gemma 4 都保留空间网格并做局部降采样。** Qwen 和 InternVL 使用 learned rearrangement/MLP path；Gemma 4 等 encoder contextualise patch 后，再用更简单的 3×3 average。
- **Flamingo 的 fusion 方式不同。** 它把 visual memory 留在 text sequence 外，并在 LLM 内增加 cross-attention；其他几种主要把 vision output 变成类似 input embedding 的向量。

这部分正文是 self-contained 的，notebook 是验证和实现练习，而不是缺失的前置内容：notebook 01 打开 Gemma 4 的精确 RoPE 与 pooling 代码；notebook 02 测试这座 tower 在 VQA/OCR/grounding 上带来什么；notebook 03 用相似任务比较 Gemma 4 固定预算与 Qwen3-VL 动态分辨率的 token 数。

## 复习题与答案

1. **`pixel_values` 的一行是什么？** 一个摊平的 16×16 RGB patch，也就是 `16 × 16 × 3 = 768` 个 rescaled pixel value。它的 `(x, y)` 坐标单独存放在 `image_position_ids`。

2. **默认档位的 2,520 行从哪里来？** 最大 280 个 soft token，每个 token 在 pooling 前对应 9 个 patch。具体图像可以只有更少的真实行，其余用零 padding，位置标为 `(-1, -1)`。

3. **为什么既需要坐标又需要 padding mask？** 坐标为 position embedding 和 spatial pooling 保存二维网格；mask 防止 padding 行参与 attention 或通过 pooling 留下来。Gemma 4 从坐标等于 `(-1, -1)` 推导这个 mask。

4. **为什么同时使用 learned position embedding 和 2D RoPE？** Learned x/y embedding 让初始 patch feature 知道位置；RoPE 让每一层 attention 对 patch 之间的横向、纵向位移敏感。它们在计算的不同位置注入空间信息。

5. **为什么在 encoder 后 pooling，而不是之前？** 后做 pooling 可以让细粒度 patch 先交换信息，再把带上下文的 feature 做 3×3 压缩。如果更早 pooling raw patch，会在 Transformer 使用局部证据前就把它丢掉。

6. **`Gemma4MultimodalEmbedder` 改变什么？** 改变 feature width，不改变 token 数。E2B 上，它通过 RMSNorm 和 learned linear projection 把 `[266, 768]` 变成 `[266, 1536]`。

7. **`[0,1] → [-1,1]` 算 normalization 吗？** 广义上算，它是固定的 centring 和 rescaling；但它不是 image processor 的 `do_normalize` 所控制的逐 channel mean/std normalization。

8. **一张图在 280 上限下只有 266 个真实 token，另外 14 个输出去哪了？** `pooler_mask` 会移除它们。它们不会进入 connector，也不占语言模型 context position。

## Notebooks

| Notebook | 在正文之外增加什么？ | 硬件 |
|---|---|---|
| [`01_vision_tower_anatomy.ipynb`](../../book/04-vision-tower/notebooks/01_vision_tower_anatomy.ipynb) | 实现 deep dive：hook 每个阶段、复现 2D RoPE、可视化 3×3 分组、检查 float32 pooling | 🟢 CPU（迷你 config）/ 🟡 真权重 |
| [`02_image_understanding.ipynb`](../../book/04-vision-tower/notebooks/02_image_understanding.ipynb) | E2B 真权重行为实验：VQA、OCR、结构化抽取、grounding 和 soft-token sweep | 🟡 24GB VRAM |
| [`03_compare_qwen3vl.ipynb`](../../book/04-vision-tower/notebooks/03_compare_qwen3vl.ipynb) | Design-space 实验：在相似任务上比较 Qwen3-VL 动态分辨率与 visual-token 数 | 🟡 12GB+ VRAM |
