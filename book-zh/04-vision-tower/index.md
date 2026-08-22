# 04 · Vision Tower — 从 patch 到 soft token

**在数据流中的位置**：`pixel_values ──► Gemma4VisionModel ──► pooler ──► Gemma4MultimodalEmbedder ──► soft tokens`

**Mental-model checkpoint：**本章打开 00 章三个学习模块中的两个：**vision tower** 给 patch embeddings 加入图像上下文，随后 **connector** 压缩 token 数量并投影 feature width。Gemma 4 虽然把它们放在一条很短的路径里，但要始终把这两个工作分开理解。

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
| [`Gemma4VisionPatchEmbedder`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L579) | patch 投影 + 学习式 2D 位置表（`_position_embeddings`） |
| [`Gemma4VisionRotaryEmbedding`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L707)、[`apply_multidimensional_rope`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L861) | 2D RoPE：按轴旋转一半 head 维度 |
| [`Gemma4VisionAttention`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L917) / [`Gemma4VisionEncoderLayer`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L986) / [`Gemma4VisionEncoder`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1030) | 主干堆叠 |
| [`Gemma4VisionPooler`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L624)、[`_avg_pool_by_positions`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L637) | 由 patch 坐标驱动的 3×3 池化 |
| [`Gemma4MultimodalEmbedder`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2085) | 视觉维度 → 文本嵌入空间 |
| [`Gemma4Model.get_image_features`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2212) | 一次调用跑完上面全部 |

## 源码走读

`Gemma4VisionModel.forward` 短到几乎可以整段引用，而它就是本章的骨架：

```python
pooling_kernel_size = self.config.pooling_kernel_size
output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)

padding_positions = (pixel_position_ids == -1).all(dim=-1)
inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
output = self.encoder(inputs_embeds=inputs_embeds,
                      attention_mask=~padding_positions,
                      pixel_position_ids=pixel_position_ids, **kwargs)
hidden_states, pooler_mask = self.pooler(output.last_hidden_state, pixel_position_ids,
                                         padding_positions, output_length)
hidden_states = hidden_states[pooler_mask]      # 剥掉 padding
if self.config.standardize:
    hidden_states = (hidden_states - self.std_bias.float()) * self.std_scale.float()
```

注意 03 章那个 `(-1, -1)` 哨兵买到了什么：一行 `padding_positions = (pixel_position_ids == -1).all(dim=-1)`，之后每个阶段都知道 2520 个槽位里哪些是真的。位置张量同时干了两件事 —— 平时该由 attention mask 干的活，**以及**充当坐标系。

### 1. Patch embedder：[0,1] 在这里变成 [-1,1]

```python
def forward(self, pixel_values, pixel_position_ids, padding_positions):
    # Gemma4 applies no normalization and instead scales in model code
    pixel_values = 2 * (pixel_values - 0.5)
    hidden_states = self.input_proj(pixel_values)
    position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
    return hidden_states + position_embeddings
```

就是它 —— 03 章缺失的那次归一化，在模型内部的一行算术里。`input_proj` 是一个无 bias 的 `nn.Linear(3 * 16², hidden_size)`：E2B 上是 768 → 768，大尺寸上是 768 → 1152。这就是整个"视觉茎"。没有卷积，没有 ResNet，没有 patch-merging 金字塔。

### 2. 两套位置信号，以及为什么都要

那张学习式的表大得刻意：

```python
self.position_embedding_table = nn.Parameter(torch.ones(2, self.position_embedding_size, self.hidden_size))
```

形状 `(2, 10240, hidden_size)` —— 每个轴一张表，各 10,240 个槽位。按每 patch 16 像素算，可覆盖每边约 164,000 像素的图像。它永远用不完，而这正是重点：查表是按**绝对坐标**做的，所以长宽比变化时不需要插值任何东西。对比 ViT 那张固定 14×14 的位置嵌入网格 —— 每个高分辨率 VLM 都得在加载时对它做双线性缩放。

查表把两个轴相加：

```python
clamped_positions = pixel_position_ids.clamp(min=0)
x_emb = F.embedding(clamped_positions[..., 0], self.position_embedding_table[0])
y_emb = F.embedding(clamped_positions[..., 1], self.position_embedding_table[1])
position_embeddings = x_emb + y_emb
position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
```

注意 `clamp(min=0)` 和它的注释：padding 位置是 `-1`，那是非法索引，于是被夹到 0 纯粹为了不崩，然后被无条件置零。**先夹后掩**这个套路在这份代码库里还会出现三次；它存在的理由是：即使某个 gather 的结果你马上要丢掉，这个 gather 本身也必须合法。

**然后 RoPE 再做一次，这次是相对的。** 学习式的表给绝对位置。`Gemma4VisionRotaryEmbedding` 给相对位置，按轴分开，而 `compute_default_rope_parameters` 里的注释点明了微妙之处：

```python
# The reference implementation computes RoPE frequencies INDEPENDENTLY
# for each spatial dimension using the partitioned head_dim (head_dim // ndim),
# so both x and y dimensions get identical frequency ranges.
# This is different from splitting the global inv_freq between dimensions.
spatial_dim = dim // 2
inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2).float() / spatial_dim))
```

仔细读，因为这是最容易做错的地方。你**不是**取长度为 `head_dim/2` 的常规 1D `inv_freq` 然后把一半给 x、一半给 y —— 那样会让两个轴拿到不同的频段，横向距离与纵向距离从此不可比。正确做法是每个轴拿到它**自己**的、在 `head_dim/2` 个通道上算出的完整频谱。`head_dim=64` 时 `spatial_dim=32`，x 与 y 各旋转 32 个通道，跨越同一频率范围。由构造保证对称。

`forward` 随后在两个轴上循环并拼接：

```python
for i in range(2):
    dim_position_ids = position_ids[:, :, i]
    freqs = (inv_freq_expanded.float() @ dim_position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    all_cos.append(emb.cos()); all_sin.append(emb.sin())
cos = torch.cat(all_cos, dim=-1)
```

而施加时把 head 维度切成 `ndim` 等份，每份用它那个轴的 cos/sin 旋转：

```python
ndim = position_ids.shape[-1]                                   # 2
num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
x_parts   = torch.split(x,   [num_rotated_channels_per_dim] * ndim, dim=-1)
y_parts = [apply_rotary_pos_emb(x=x_parts[k], cos=cos_parts[k], sin=sin_parts[k], unsqueeze_dim=unsqueeze_dim)
           for k in range(ndim)]
return torch.cat(y_parts, dim=-1)
```

`apply_multidimensional_rope` 对 `ndim` 是泛化的 —— 传 3D 位置它就做三向旋转。这份泛化大概正是音频塔和未来任何视频原生变体能共用这套机件的原因。

所以：**绝对位置活在残差流里（在茎处加一次）；相对位置活在注意力分数里（每层都施加一次）。** 它们回答不同的问题 —— "这个 patch 在页面的哪里"与"这两个 patch 相距多远" —— 而 Gemma 4 两个都要。

### 3. 编码器块

`Gemma4VisionEncoderLayer` 是 Gemma 风格的块：每个子层前**和**后各有一个 RMSNorm（`input_layernorm` / `post_attention_layernorm`、`pre_feedforward_layernorm` / `post_feedforward_layernorm`），一个门控 MLP，没有因果 mask —— `self.is_causal = False`，因为图像没有阅读顺序。

有两个细节是 Gemma 4 特有的。

**QKV norm，包括一个无缩放的 value norm：**

```python
self.q_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
self.k_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
```

在注意力之前归一化 query 和 key 现在是常规操作（它阻止 logits 在长训练中漂移）。归一化 **value** 就少见了，而不带可学缩放地做 —— `with_scale=False`，即纯归一化、零参数 —— 更少见。你会在文本解码器里遇到一模一样的三件套（06 章）。

**Clippable linear。** 视觉塔和音频塔里每个投影都是 `Gemma4ClippableLinear`：

```python
if self.use_clipped_linears:
    hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
hidden_states = self.linear(hidden_states)
if self.use_clipped_linears:
    hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
```

边界是**从 checkpoint 加载的 buffer**，初始化为 ±inf，所以没有用截断训练过的模型不受影响。E2B 的视觉塔发布时带 `use_clipped_linears: true`；31B 是 `false`。这是烘焙进已发布权重里的激活截断 —— 一项只因为有人在训练时测过数值范围才存在的量化与稳定性措施。

### 4. Pooler：按坐标做 3×3，而不是按下标

池化一个 patch 网格的朴素做法是 reshape 然后取平均。这里行不通，因为每张图的网格形状都不同，而且张量是补过 padding 的。于是 pooler 把池化操作构造成一次**由坐标驱动的矩阵乘法**：

```python
k = int((input_seq_len // length) ** 0.5)                       # 3
clamped_positions = pixel_position_ids.clamp(min=0)
max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
output = weights.transpose(1, 2) @ hidden_states.float()
```

逐行读：把每个 `(x, y)` 整除 3 得到它所属 3×3 格子的坐标；用图像自身的宽度把格子坐标压平成单个下标；把该下标 one-hot 成一个 `(patch → soft token)` 的分配矩阵、乘以 1/9；然后矩阵乘。padding patch 在调用前已被 `masked_fill` 置零，因此对任何平均值都无贡献。

这是一种不寻常的平均池化写法，而且是正确的那种：它对任意网格形状、任意 padding 模式都成立，一次批量算子搞定，不需要在图像上写 Python 循环。返回的 `mask`（`torch.logical_not((weights == 0).all(dim=1))`）标出哪些输出槽位至少收到一个真实 patch —— 那就是用来剥掉 padding 的 `pooler_mask`。

接着是一步缩放，其注释精确告诉你曾经出过什么事：

```python
# The scaling expands the activation magnitude, which can exceed the float16 range, so it is
# computed in float32 and the pooled features are returned in float32.
hidden_states = hidden_states.float() * self.root_hidden_size
```

乘以 `√hidden_size`（768 时约 27.7）可能撑破 fp16 的 65504 上限。因此无论模型是什么 dtype，pooler 一律返回 **float32**，由调用方在标准化后再转回去。如果你哪天自己造塔，这是那类几乎不可能从 loss 曲线上发现的 bug。

`standardize`（仅大尺寸）随后施加从 checkpoint 加载的 `std_bias` 与 `std_scale` —— 在 soft token 离开塔之前做一次学习式白化。

### 5. 进入文本空间

```python
class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(self, multimodal_config, text_config):
        self.multimodal_hidden_size = getattr(multimodal_config, "output_proj_dims", multimodal_config.hidden_size)
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, text_config.hidden_size, bias=False)
        self.embedding_pre_projection_norm = Gemma4RMSNorm(self.multimodal_hidden_size, eps=self.eps, with_scale=False)

    def forward(self, inputs_embeds):
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))
```

**整个 connector 就是：一个 RMSNorm 加一个无 bias 的线性层。** 在上面那一整套机件之后 —— 预算求解、双位置编码、坐标驱动池化 —— 真正通往语言模型的那座桥，是整个文件里最无聊的模块。这就是 LLaVA 的教训，在 2026 年依然成立：projector 不需要聪明；其他一切才需要。

这个类与音频共用，所以它在存在时读 `output_proj_dims`（音频塔自己的输出宽度 1536），否则回落到 `hidden_size`。每个模型有两个实例，`embed_vision` 和 `embed_audio`（01 章 §4），这也正是它们在 10 章可以被分别冻结的原因。

`Gemma4Model.get_image_features` 把它们串起来：

```python
vision_outputs = self.vision_tower(pixel_values=pixel_values, pixel_position_ids=image_position_ids, **kwargs)
vision_outputs.pooler_output = self.embed_vision(inputs_embeds=vision_outputs.last_hidden_state)
return vision_outputs
```

结果是一串扁平的 soft token，位于文本嵌入空间中，等着 08 章把它们散射进 prompt。

## 设计空间

| 模型 | 图像 → LLM token | 空间位置 | Connector |
|---|---|---|---|
| **Flamingo**（2022） | Perceiver resampler → 64 个 latent | latent 上的 1D | LLM 内部的门控交叉注意力层 |
| **BLIP-2**（2023） | Q-Former → 32 个 query token | 学习式 query | Q-Former + 线性层 |
| **LLaVA**（2023） | 576 个固定 patch | 1D 学习式 | **单个 MLP** |
| **Qwen2/3-VL** | 原生分辨率、2×2 merge | M-RoPE（t/h/w） | MLP merger |
| **InternVL 3.5** | Tile + 缩略图 | 每 tile 内 1D | MLP + pixel shuffle |
| **Gemma 4** | 预算内的 patch、3×3 池化 | 学习式 2D 表 **+** 2D RoPE | RMSNorm + 线性层 |

有两条轴值得分开看。

**压缩**：所有人都压缩，只是对"在哪压"意见不一。Q-Former 用一个**学习出来的**模块压到固定输出尺寸 —— 表达力强，但那是一个必须训练的瓶颈，而且不论内容多少都是固定预算。Pixel shuffle 和 2×2 merge 靠**重排**通道来压缩，免费且近乎无损，但只能按整数倍。Gemma 4 的 3×3 平均池化是所有选项里最粗暴的，而它有效 —— 这本身就说明编码器已经把大部分重活干完了。

**位置**：LLaVA 把网格压成一条线然后指望 LLM 自己想明白。Qwen 的 M-RoPE 和 Gemma 4 的 2D RoPE 都诚实地编码了网格，并且从不同方向到达了几乎相同的地方 —— M-RoPE 在（时间、高、宽）之间切分 head 维度，Gemma 4 在（x, y）之间切分且每轴共享同一频谱。Gemma 4 额外保留了一张绝对学习表，而 Qwen 放弃了它。这些额外参数很便宜（10240 × hidden × 2），换来的是与分辨率无关的绝对定位。

Gemma 4 真正不寻常的地方**不是**它没用已发布的 SigLIP checkpoint。多数开源 VLM 外挂 SigLIP-so400m 并围绕它 224/384 的固定网格做适配。Gemma 4 从一开始就带着预算方案训练了自己的塔，这也是它的预处理看起来和别人完全不一样的原因。

## 自测

1. 视觉塔同时拥有一张学习式绝对位置表**和** 2D RoPE。分别删掉其中一个，具体会退化什么？
2. 为什么每个轴要拿到自己的完整频谱，而不是把单个 `inv_freq` 对半分？
3. `clamp(min=0)` 在 `_position_embeddings` 和 `_avg_pool_by_positions` 里都出现了。没有它会怎样？为什么夹到什么值并不重要？
4. 模型是 bfloat16 时 pooler 却返回 float32。为什么？逼出这个决定的那个魔数是多少？
5. 一张图在 280 预算下产出 266 个真实 soft token。执行 `hidden_states[pooler_mask]` 之后，另外 14 个槽位里是什么？
6. Connector 只有一个 RMSNorm 加一个线性层。结合上游的一切，论证为什么这就够了 —— 然后论证 BLIP-2 当年为什么认为不够。

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_vision_tower_anatomy.ipynb` | hook 住每个阶段打印形状；手写 `apply_multidimensional_rope` 并 `assert_close`；可视化哪些 patch 池化成了哪个 soft token | 🟢 CPU（迷你 config）/ 🟡 真权重 |
| `02_image_understanding.ipynb` | 真权重 E2B 下这座塔换来了什么：VQA、OCR、文档结构化抽取、grounding 画框 | 🟡 24GB 显存 |
| <a href="../../04-vision-tower/notebooks/03_compare_qwen3vl.html"><code>03_compare_qwen3vl.ipynb</code></a> | 设计空间对照组：Qwen3-VL 的原生分辨率 + 2×2 merge 在同一批任务上的表现，视觉 token 数并排对比 | 🟡 12GB+ 显存 |
