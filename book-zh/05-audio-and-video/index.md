# 05 · Audio and Video — 另外两个前端

**在数据流中的位置**：`audio ──► 特征提取 ──► Gemma4AudioModel ──► soft tokens`，以及 `video ──► 抽帧 ──► Gemma4VisionModel`

两个模态，新增机件的量却完全不同。视频完全复用视觉塔——视频就是帧，帧就是图像，唯一的新部件是决定抽**哪些**帧的采样器。音频则拿到了一座专门设计的编码器，长得跟视觉塔毫无关系：卷积下采样、相对位置编码、带显式左上下文的分块注意力。这个不对称就是本章的主题。

音频**仅 E2B / E4B 原生支持**。31B 和 26B-A4B 的 checkpoint 里没有音频塔——`Gemma4Model.__init__` 在 `audio_config is None` 时直接跳过（01 章）。

## 你会学到

1. 波形如何变成模型输入：16kHz 下 128 个 mel 频带，再经堆叠卷积投影做 4× 时间下采样——"每个音频 token 40ms"正是从这里来的
2. 分块注意力：`attention_chunk_size=12`、`attention_context_left=13`、`attention_context_right=0`，以及流式友好的编码器为什么长这样而不用全注意力
3. Conformer 家族的积木——因果 1D 卷积、轻量卷积模块、相对位置编码、logit 截断——以及语音编码器为什么保留了视觉早已抛弃的卷积
4. 视频采样怎么工作，`do_sample_frames` / `num_frames` / `fps` 到底控制什么，一段视频真实的 token 成本是多少
5. **设计空间**：Gemma 4 音频塔 vs Whisper 的 encoder-decoder vs CTC/transducer 流式 ASR——通用多模态模型相对专用 ASR 让出了什么

## 源码地图

| 文件 | 符号 | 作用 |
|---|---|---|
| `feature_extraction_gemma4.py` | `Gemma4AudioFeatureExtractor` | 波形 → 16kHz 下 128 维 mel 特征 |
| `video_processing_gemma4.py` | `Gemma4VideoProcessor` | 抽帧与逐帧预处理 |
| `modeling_gemma4.py` | `Gemma4AudioSubSampleConvProjection` | 4× 时间下采样 |
| | `Gemma4AudioAttention` | 带左上下文的分块注意力 |
| | `Gemma4AudioCausalConv1d`、`Gemma4AudioLightConv1d` | Conformer 式积木 |
| | `Gemma4AudioModel` | 组装好的音频塔与它的 mask 管线 |

## 源码走读 —— 音频

`Gemma4AudioModel` 的 docstring 直接点出了它的血统：

> *An audio encoder based on the [Universal Speech Model](https://huggingface.co/papers/2303.01037) architecture.*

这是 USM/Conformer 的后代，不是改造过的 ViT —— 而在读完 04 章之后再读它特别有教益，恰恰因为几乎没有东西可以迁移过来。

### 1. 波形 → mel 帧

`Gemma4AudioFeatureExtractor` 发布时的设置：

| 字段 | 值 | 含义 |
|---|---|---|
| `sampling_rate` | 16000 | 16kHz 单声道 |
| `feature_size` | 128 | mel 频带数 |
| `frame_length` | 320 | 20ms 分析窗 |
| `hop_length` | 160 | **每帧 10ms** |
| `fft_length` | 512 | |
| `min_frequency` / `max_frequency` | 0 / 8000 | 完整奈奎斯特带 |
| `preemphasis_htk_flavor` | `true` | HTK 兼容的预加重 |

每 10ms 一个 mel 帧。记住这个数。

### 2. 下采样：时间 4×，频率 4×

```python
class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
    self.norm = nn.LayerNorm(out_channels, ...)
    self.act  = nn.ReLU()
```

两层叠起来，作用在被当作**单通道图像**的 mel 频谱上（`input_features.unsqueeze(1)` → `[batch, 1, time, 128]`）。每层把两个轴各砍一半，两层之后：时间 ÷4，mel 频带 128 → 32。

**每帧 10ms × 4 = 每 token 40ms。** 这就是 02 章 `audio_ms_per_token = 40` 的由来，也是 processor 能在不跑模型的情况下预测 token 数的原因。一段 3.4 秒的音频：54,400 采样点 → 339 个 mel 帧 → 170 → **85 个音频 token**，恰好是 3400ms ÷ 40。

通道的算术值得再看一眼：

```python
proj_input_dim = (config.subsampling_conv_channels[0] // 4) * config.subsampling_conv_channels[1]
self.input_proj_linear = nn.Linear(proj_input_dim, config.hidden_size, bias=False)
```

`(128 // 4) * 32 = 1024 = hidden_size`。那个 `// 4` 是**频率**方向的缩减（128 个 mel 频带 → 32），32 是第二个卷积的输出通道数。把 32 个频率位置 × 32 个通道拍平，正好落在模型宽度上。这里没有任何东西暴露在 config 里 —— `subsampling_conv_channels = (128, 32)` 是暴露的，但 kernel、stride、padding 是硬编码的，这正是 02 章 `_compute_audio_num_tokens` 的 docstring 警告过的耦合。

mask 一路跟随，用最省的方式下采样：

```python
if mask is not None:
    hidden_states = hidden_states * mask[:, None, :, None]   # 先把 padding 置零
...
    mask = mask[:, ::2]                                       # 再抽样
```

### 3. 带左上下文的分块注意力

本章的核心。config 里的三个数：

```python
self.chunk_size        = config.attention_chunk_size        # 12
self.max_past_horizon  = config.attention_context_left - 1  # 12
self.max_future_horizon = config.attention_context_right    # 0
self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon   # 24
```

query 被切成不重叠的 12 个 token 的块；key 和 value 被聚成**重叠的** 24 长度窗口，向过去伸出 12 个 token、向未来伸出 0 个。

```python
query_states = self._convert_to_block(query_states)       # [B, num_blocks, 12, H, D]
key_states   = self._extract_block_context(key_states)    # [B, num_blocks, 24, H, D]
value_states = self._extract_block_context(value_states)
```

`_convert_to_block` 是补齐加 reshape。`_extract_block_context` 是补齐加 `unfold` —— 那个能在不多复制的前提下物化重叠视图的滑窗技巧。

换算成现实：12 个 token 是 480ms，24 是 960ms。**这个编码器向后看不超过约一秒，向前完全不看。** 这是刻意的流式友好设计 —— `attention_context_right = 0` 意味着它原则上可以在实时麦克风上因果运行。对比 Whisper：它在补齐到 30 秒的窗口上做双向注意力，因此必须等一句话说完才能开始。

注意力开销是论证的另一半。对一段 10 分钟录音（15,000 token）做全注意力，每个头是 2.25 亿对；分块注意力是 `num_blocks × 12 × 24` ≈ 36 万。语音是局部的 —— 音素不取决于八分钟前说了什么 —— 所以那个二次项几乎买不到东西，模型把长程预算花在语言解码器里。

**相对位置，Transformer-XL 风格。** 这里没有 RoPE。取而代之的是经典分解，源码直接引用了论文：

```python
relative_key_states = self.relative_k_proj(position_embeddings)
matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)          # 内容 × 内容
matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)  # 内容 × 位置
matrix_bd = self._rel_shift(matrix_bd)                            # appendix B of 1901.02860
attn_weights = matrix_ac + matrix_bd
```

`_rel_shift` 是那套 pad-reshape-slice 的手法，把一个相对距离分数矩阵变成正确对齐的逐 query 偏移。不画图看着像胡来；它是 notebook 里较好的练习之一。

**两个你在这个模型别处找不到的缩放技巧：**

```python
self.q_scale = (self.head_dim**-0.5) / math.log(2)
self.k_scale = math.log(1 + math.e) / math.log(2)
...
query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
```

把两个缩放都除以 `log 2`，就把随后的 softmax 变成以 2 为底 —— 在多数硬件上 `exp2` 比 `exp` 便宜，而把常数折进投影里让这件事免费。而 `per_dim_scale` 是一个逐维度的可学 query 缩放，经 `softplus` 保正：模型可以学到某些 head 维度比另一些更重要，这是单个标量 `1/√d` 表达不了的。

**Logit softcap**，与解码器的 `final_logit_softcapping` 同一个思路：

```python
attn_weights = torch.tanh(attn_weights / self.softcap) * self.softcap   # softcap = 50.0
```

平滑天花板而不是硬夹，梯度得以存活。

**mask 必须被重塑成匹配的形状。** 库先构造一个常规 4D mask，再把它折进块布局：

```python
attention_mask = create_bidirectional_mask(
    config=self.config, inputs_embeds=hidden_states, attention_mask=output_mask,
    and_mask_function=sliding_window_mask_function((self.config.attention_context_left - 1,
                                                    self.config.attention_context_right)))
attention_mask = self._convert_4d_mask_to_blocked_5d(attention_mask)
```

`_convert_4d_mask_to_blocked_5d` 用显式构造的 `kv_indices` 把 `[B, 1, seq, seq]` gather 成 `[B, 1, num_blocks, 12, 24]`。注意用的是 `create_bidirectional_mask` —— 在自己那 24 个 token 的窗口内，一个音频 token 是双向可见的。"分块"不等于"因果"。

### 4. 这一层：重排过的 Conformer

```python
hidden_states = self.feed_forward1(hidden_states)
residual = hidden_states
hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
hidden_states = self.norm_pre_attn(hidden_states)
hidden_states, _ = self.self_attn(...)
hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
hidden_states = self.norm_post_attn(hidden_states)
hidden_states += residual
hidden_states = self.lconv1d(hidden_states)
hidden_states = self.feed_forward2(hidden_states)
hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
hidden_states = self.norm_out(hidden_states)
```

前馈、注意力、卷积、前馈这个 macaron 三明治是 Conformer 的标志：注意力处理全局结构，深度可分卷积（`Gemma4AudioLightConv1d`，内部是**因果的** `Gemma4AudioCausalConv1d`）处理局部结构，两个减半的前馈把它们包起来。注意 residual 是在 `feed_forward1` **之后**取的，不是之前。

再注意每层出现三次的 `torch.clamp`，而且上界本身还要被 dtype 夹一次：

```python
gradient_clipping = min(self.gradient_clipping, torch.finfo(self.norm_pre_attn.weight.dtype).max)
```

在 clippable linear、logit softcap、float32 注意力之间，音频塔是整个模型里数值上最有防御性的代码。语音编码器处理的输入动态范围极大，这一点写在脸上。

最后：

```python
self.output_proj = nn.Linear(config.hidden_size, config.output_proj_dims, bias=True)
```

1024 → 1536，而且是塔里唯一**带 bias** 的投影。在 E2B 上 `output_proj_dims = 1536` 恰好等于文本模型的 `hidden_size`，所以 `embed_audio` 的活是同宽度精修而不是改尺寸。

### 5. 为什么大模型没有音频塔

01 章那张表：31B 与 26B-A4B 上 `"audio_config": null`。上面的一切大约是 12 层 × 1024 宽加上卷积栈 —— 参数量不大，但它是一个**单独训练的编码器**，需要配对语音数据、自己的评测体系、自己的数值稳定性工作。Gemma 4 把这份投入放进了为手机准备的尺寸里，那里端上语音接口才是重点。一个产品决策，被一个 JSON 里的 `null` 暴露无遗。

## 源码走读 —— 视频

视频几乎不需要新机件，而它那一点点新东西很能说明问题。

`Gemma4VideoProcessor` 发布时的设置与图像处理器的差别，正是你希望看到的方向：

| 字段 | 图像 | 视频 |
|---|---|---|
| `max_soft_tokens` | 280 | **70** |
| `num_frames` | — | 32 |
| `do_sample_frames` | — | `true` |
| `do_normalize` | `false` | `true`（mean 0、std 1 —— 等于没做） |

**一个视频帧只值一张静态图的四分之一。** 32 帧 × 70 soft token = **2,240 个 token**，对比一张照片的 280。整段视频只花八张图的上下文，这个交易才让视频塞得进 prompt。它也预告了你该期待什么：Gemma 4 会把片子的**叙事**读得不错，把**细节**读得很差。要认出视频里的车牌，那是静态图的活。

抽帧由 `do_sample_frames`、`num_frames`、`fps` 控制 —— 默认均匀采样到固定帧数。然后这些帧走的是和任何图像一样的 `Gemma4VisionModel`；没有时序注意力，没有 3D 卷积，没有视频专用塔。

时间信息改由 **prompt** 承载，正如 02 章所示：

```python
timestamp_str = [f"{int(seconds // 60):02d}:{int(seconds % 60):02d}" for seconds in metadata.timestamps]
video_replacement = " ".join([f"{t} {self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}"
                              for t in timestamp_str])
```

`00:00 <boi>…<eoi> 00:02 <boi>…<eoi> …` —— 字面文本时间戳与帧 token 交错。语言模型像读任何其他文本一样读它们。如果 `fps` 推断不出来，processor 会警告并假设 24，你的时间戳就此悄悄错掉，在流水线里值得专门防一手。

## 设计空间

**音频编码器：**

| | Gemma 4 | Whisper | CTC/transducer（Parakeet、Canary） |
|---|---|---|---|
| 注意力 | 分块，24 token 窗口，无前瞻 | 全注意力，覆盖补齐的 30s 窗口 | 由构造即流式 |
| 位置 | Transformer-XL 相对 | 正弦绝对 | 各异 |
| 输出 | 进入 LLM 的 soft token | 文本，经自己的解码器 | 文本，经对齐 |
| 流式 | 结构上可行 | 否 —— 需要整个窗口 | 是，这是它存在的全部理由 |
| 长音频 | 受 LLM 上下文限制 | 30s 分块后拼接 | 无上界 |

有意思的对比不是准确率，而是**输出是给谁用的**。Whisper 的编码器喂给一个只会吐转写的解码器。Gemma 4 的喂给一个通用语言模型，于是"转写这段"和"说话人的情绪是什么，跟他正在讲的幻灯片一致吗"是同一条代码路径。你放弃了专用模型的词错率，换来与其他所有模态的可组合性。notebook 会把两者跑在同一段音频上，让你看见这个交易的大小，而不是靠猜。

**视频：** 领域分成两派 —— 把帧重采样成类文本序列、让 LLM 做时序推理的（Gemma 4，以及多数开源 VLM），和拥有真正时序机件的（3D patch、时序注意力、或时间感知位置编码：Qwen-VL 的 M-RoPE 时间轴，以及 [21 章](../21-video-generation/overview.md)的视频生成模型）。抽帧要简单得多，而且对问答来说够用；对任何需要精确运动的任务则不够。

## 自测

1. 从特征提取器的设置和 SSCP 栈推导出每个音频 token 40ms。要让它变成 20ms，你得改哪些参数？
2. `attention_context_right = 0`。它使什么成为可能？代价是什么？
3. 音频塔为什么用 Transformer-XL 相对位置而不是 04 章的 2D RoPE？
4. `q_scale` 和 `k_scale` 除以 `log 2` 是为了什么？
5. 一段 32 帧的视频和一张照片，哪个更费上下文？差多少？这预示了模型在视频上的什么弱点？
6. 第 17 帧的时间戳在模型输入的哪个位置被表示？（有且只有一处。）

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_audio_tower_anatomy.ipynb` | 追踪一段波形经 mel 提取与卷积下采样的形状变化；手写重建分块注意力 mask 并与库 `assert_close` | 🟢 CPU |
| `02_asr_and_video.ipynb` | 真权重 E2B：转写、音频问答、视频问答——视频的 token 成本是测出来的，不是猜的 | 🟡 24GB 显存 |
| <a href="../../05-audio-and-video/notebooks/03_whisper_asr_compare.html"><code>03_whisper_asr_compare.ipynb</code></a> | 设计空间对照组：同一段音频交给 Whisper。专用模型强在哪，又做不到什么 | 🟢 CPU / 🟡 |
