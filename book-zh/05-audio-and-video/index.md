# 05 · Audio and Video — 另外两个前端

**在数据流中的位置**：`audio ──► 特征提取 ──► audio tower ──► projector ──► LLM soft tokens ──► 文本`，以及 `video ──► 抽帧 ──► Gemma4VisionModel`

两个模态，新增机件的量却完全不同。视频完全复用视觉塔——视频就是帧，帧就是图像，唯一的新部件是决定抽**哪些**帧的采样器。音频则拿到了一座专门设计的编码器，长得跟视觉塔毫无关系：卷积下采样、相对位置编码、带显式左上下文的分块注意力。这个不对称就是本章的主题。

音频**仅 E2B / E4B 原生支持**。31B 和 26B-A4B 的 checkpoint 里没有音频塔——`Gemma4Model.__init__` 在 `audio_config is None` 时直接跳过（01 章）。

## 你会学到

1. 从零建立 speech recognition 的数据流：波形、STFT、Mel、Log-Mel、audio soft token 和文字 token 分别是什么
2. 16kHz 下的 128 个 Mel 频带如何经卷积变成时间序列；为什么卷积既提取局部模式又降低 token 成本
3. Gemma 4 的 USM/Conformer 塔：分块局部注意力、因果 1D 卷积、相对位置编码和数值保护
4. **Higgs Audio v3 STT 完整实例**：Whisper-Large-v3 encoder 如何接 Qwen3-1.7B，逐层 shape、projector、placeholder 替换、训练 loss 与生成
5. 音频和文字必须按什么顺序放；30 秒究竟限制哪一层；长录音如何切块而不把“切块”和“流式”混为一谈
6. 视频采样怎么工作，以及一段视频真实的 token 成本是多少

## 源码地图

| 文件 | 符号 | 作用 |
|---|---|---|
| `feature_extraction_gemma4.py` | [`Gemma4AudioFeatureExtractor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/feature_extraction_gemma4.py#L49) | 波形 → 16kHz 下 128 维 mel 特征 |
| `video_processing_gemma4.py` | [`Gemma4VideoProcessor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/video_processing_gemma4.py#L158) | 抽帧与逐帧预处理 |
| `modeling_gemma4.py` | [`Gemma4AudioSubSampleConvProjection`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L385) | 4× 时间下采样 |
| | [`Gemma4AudioAttention`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L249) | 带左上下文的分块注意力 |
| | [`Gemma4AudioCausalConv1d`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L451)、[`Gemma4AudioLightConv1d`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L484) | Conformer 式积木 |
| | [`Gemma4AudioModel`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1931) | 组装好的音频塔与它的 mask 管线 |
| [Higgs `modeling_higgs_audio.py`](https://huggingface.co/bosonai/higgs-audio-v3-stt/blob/main/modeling_higgs_audio.py) | `HiggsAudioEncoder`、`HiggsAudioFeatureProjector` | Whisper 塔、池化、projector 与 Qwen 融合 |
| [Higgs `higgs_audio_collator.py`](https://huggingface.co/bosonai/higgs-audio-v3-stt/blob/main/higgs_audio_collator.py) | `HiggsAudioSampleCollator` | 4 秒切块、Log-Mel batching、placeholder 复制 |

## 先把 speech recognition 整条链建起来

如果只盯着 `Conv1d` 或 Transformer，很容易把三个完全不同的“token”混在一起。本章统一用这三个名字：

| 名称 | 例子 | 是离散 ID 吗 | 表示什么 |
|---|---|---:|---|
| **Mel frame** | 10ms 附近的 128 个数 | 否 | 这一小段时间里各频带有多强 |
| **audio soft token** | 一个 1536D / 2048D 向量 | 否 | audio tower 压缩出的上下文化声音表示 |
| **text token** | `" speech" → 6212` | 是 | tokenizer 词表里的离散单位 |

audio tower 的输出不是文字，也不是“猜出来的 token ID”。它只是被 projector 映射到 LLM 的 hidden dimension，使它能占据文本 embedding 序列中的若干位置。最后只有 LLM 的 `lm_head` 才把 hidden state 变成词表 logits。因而更准确的说法是：

> audio tower 把声音映射到 **LLM hidden space**，不是直接映射到 text vocabulary space。

以转写一句 `OK, the meeting starts at five` 为例，推理是：

```text
16kHz waveform
    │
    ├─ 每 10ms 左右取一个短窗
    ▼
Log-Mel matrix                    [time, 128]
    │
    ├─ convolution / subsampling
    ├─ audio Transformer or Conformer
    ▼
contextual audio sequence         [audio_tokens, audio_hidden]
    │
    ├─ projector
    ▼
LLM-space audio soft tokens       [audio_tokens, llm_hidden]
    │
    ├─ 插入 prompt 的 audio placeholder 位置
    ▼
causal language model
    │
    ├─ next-token generation
    ▼
"OK, the meeting starts at five"
```

这里没有要求一个 audio soft token 对齐一个单词。一个 soft token 可能覆盖 40–80ms，通常只够覆盖一个音素的一部分；单词边界和语言纠错是在后面的上下文化与自回归解码中形成的。

### 1. 波形为什么不能直接当 token

16kHz 表示每秒 16,000 个采样点。让 LLM 直接把每个采样点当一个位置，一分钟就是 960,000 个位置，而且单个采样点几乎没有语义。语音的有用结构位于多个尺度：

- 几毫秒：周期性、爆破和噪声；
- 几十毫秒：近似稳定的音素频谱；
- 数百毫秒：音节和单词；
- 数秒以上：句法、话题和上下文。

前端先把最密的采样点压成约 100 个 frame/秒，再由 audio tower 压成 12.5–25 个 soft token/秒。LLM 因此接收的是“声音摘要序列”，不是原始空气振动。

### 2. STFT：问“这个短窗里有哪些频率”

对 waveform `x[n]`，每隔 `H` 个采样点取一个长度为 `N` 的窗并做 Fourier transform：

\[
X(t,k)=\sum_{n=0}^{N-1}x[tH+n]w[n]e^{-j2\pi kn/N}.
\]

`t` 是时间帧，`k` 是 FFT frequency bin。取幅度后，相位被暂时丢掉：

\[
S(t,k)=|X(t,k)|.
\]

Gemma 4 用 20ms 窗、10ms hop；Whisper 用 25ms 窗（400 samples）、10ms hop。两者都会得到大约 100 frame/秒。窗比 hop 长，所以相邻帧重叠：这不是重复浪费，而是避免一个音素恰好落在边界上就被切坏。

### 3. Mel 与 log：把工程坐标换成人耳坐标

FFT 的频率格是线性的，人耳对低频分得细、对高频分得粗。Mel filterbank 用一组重叠三角滤波器把 FFT bins 聚合为 128 个频带：

\[
M(t,m)=\sum_k F_{m,k}S(t,k),\qquad m=1,\ldots,128.
\]

再取 log：

\[
L(t,m)=\log(M(t,m)+\epsilon).
\]

取 log 的直觉是把“倍数差异”变成“加法差异”，压低巨大动态范围，也更接近响度感知。最终：

\[
L\in\mathbb{R}^{T\times128}.
\]

一列/一行的方向在不同库里不同：Gemma 4 的模型输入是 `[B,T,128]`；PyTorch Whisper/Higgs 是 `[B,128,T]`。数据没有不同，只是 layout 不同。

### 4. ASR 的学习目标

对 LLM-style ASR，训练样本可以抽象成：

```text
prefix:  instruction + audio soft tokens + assistant marker
target:  transcript text tokens
```

prefix 的 label 设成 `-100`，只在 transcript 上算普通 next-token cross-entropy：

\[
\mathcal L=-\sum_j\log p(y_j\mid A,\text{instruction},y_{<j}).
\]

它不需要 CTC 的逐帧对齐，也不表示第 73 个 audio token 就对应第 9 个文字 token。代价是时间戳、强制对齐和严格流式稳定性通常不如专门为这些目标训练的 ASR 系统。

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
| `preemphasis` | 0.0 | 发布默认值不做预加重；`htk_flavor` flag 因而不生效 |

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

同样取 8 秒，完整 shape ledger 是：

```text
waveform                  [B, 128000]
Log-Mel                   [B, 800, 128]
unsqueeze                 [B, 1, 800, 128]
Conv2D 1→128, stride 2    [B, 128, 400, 64]
Conv2D 128→32, stride 2   [B, 32, 200, 32]
flatten freq×channel      [B, 200, 1024]
12-layer audio tower      [B, 200, 1024]
output_proj               [B, 200, 1536]
embed_audio               [B, 200, text_hidden]
```

注意它和 Higgs 的第一个分叉：Gemma 4 的 Conv2D 同时沿**时间和频率**滑动；Higgs 的 Conv1D 把全部 Mel bins 当 channels，只沿**时间**滑动。

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

query 被切成不重叠的 12 个 token 的块；为了能一次矩阵乘处理整块 query，key 和 value 先被收集成**重叠的** 24 槽候选窗口：前一块 12 个，加当前块 12 个。

```python
query_states = self._convert_to_block(query_states)       # [B, num_blocks, 12, H, D]
key_states   = self._extract_block_context(key_states)    # [B, num_blocks, 24, H, D]
value_states = self._extract_block_context(value_states)
```

`_convert_to_block` 是补齐加 reshape。`_extract_block_context` 是补齐加 `unfold` —— 那个能在不多复制的前提下物化重叠视图的滑窗技巧。

但 24 是**计算张量的宽度**，不是每个 query 最终都能看到 24 个 token。真正的方向约束在后面的 mask：

```python
dist = q_idx - kv_idx
left_mask  = (dist >= 0) & (dist < 12)
right_mask = (dist < 0)  & (-dist < 0)   # 永远为 False
```

所以一个位置实际能看自己和至多约 11 个过去位置，不能看未来；在 40ms/token 下是约 480ms 的有效历史。之所以仍构造 `[12 queries × 24 candidate keys]`，是为了让同一块里的 12 个 query 共享一次规整的 batched matmul，而 mask 再把各 query 不该看的那半边抹掉。

这是一种**流式友好**而不是“开箱即用的流式服务”。attention 与内部 depthwise Conv1D 是因果的；但当前 feature extractor 以半因果窗构造 STFT，前面的两层 `Conv2d(padding=1)` 也不是带 cache 的在线实现，`Gemma4AudioCausalConv1d` 源码里甚至还留着 cache 的 TODO。要真正接麦克风，需要保存 STFT/卷积/注意力边界状态，而不是反复把越来越长的整段音频重跑一遍。

对比 Higgs 的 Whisper tower：Whisper self-attention 在**每个送入 tower 的 chunk 内双向**，因此能利用该 chunk 的未来声音，代价是至少等到 chunk 收齐。Higgs 可以一块一块地产生中间结果，但那叫 chunked inference；它不等于 encoder 本身因果。

注意力开销是论证的另一半。对一段 10 分钟录音（15,000 token）做全注意力，每个头是 2.25 亿对；这个 block 实现物化的是 `num_blocks × 12 × 24` ≈ 36 万对，其中一部分随后被 mask。语音的声学证据高度局部；跨句的长程关系交给后面的语言模型处理，更划算。

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

这里最容易产生一个错误结论：**它没有把 softmax 改成以 2 为底。** 后面调用的仍是普通 `F.softmax`。`log(2)` 来自这套 USM 参数化的初始化校准：`per_dim_scale` 初始化为 0，而 `softplus(0)=log(2)`，恰好抵消 query scale 里的 `/log(2)`，使初始 query 缩放回到熟悉的 `1/√d`。训练后，每一维都有一个经 `softplus` 保持为正的可学习温度。`k_scale=softplus(1)/log(2)` 是固定的 key 校准；不要把这些常数解释成另一种 softmax 实现。

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

`_convert_4d_mask_to_blocked_5d` 用显式构造的 `kv_indices` 把 `[B, 1, seq, seq]` gather 成 `[B, 1, num_blocks, 12, 24]`。这里调用 `create_bidirectional_mask` 只表示“不额外叠加库的默认 causal mask”；传入的 `and_mask_function` 已经用上面的 `dist` 判断编码了方向。最终仍是**局部因果**，不能仅凭函数名判断可见性。

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

1024 → 1536，而且是塔里唯一**带 bias** 的投影。随后还有模型级的 `embed_audio`：先做无 scale RMSNorm，再线性映射到 text hidden size。不要把 `output_proj` 和这个 multimodal embedder 合成脑中一个模糊的“projector”；前者属于 audio tower，后者是所有模态进入 LLM 前共享的接口设计。

### 5. soft token 怎样进入 Gemma 4 decoder

processor 已经按 `40ms/token` 在文本中放好了 `<audio_soft_token>` 槽位。forward 先正常取得 text embeddings，再运行 audio tower：

```python
audio_output   = self.get_audio_features(input_features, input_features_mask)
audio_features = audio_output.pooler_output       # 已经通过 embed_audio
audio_features = audio_features[audio_output.attention_mask]  # 剥掉 batch padding
inputs_embeds  = inputs_embeds.masked_scatter(audio_mask, audio_features)
```

最终 LLM 看到的是一个普通长度维上的混合序列：

```text
[instruction text embeddings]
[audio soft embedding 1] ... [audio soft embedding N]
[assistant marker]
```

没有 cross-attention memory，也没有第二条 decoder 支路。音频槽位在进入 decoder 前已经被真实向量替换，之后与文本位置一起走 Gemma 4 的 causal decoder。这一点和稍后的 Higgs 相同。

Gemma 4 的默认 feature extractor 还有一个很实际的限制：`max_length=480_000` samples、`truncation=True`。16kHz 下正好是 **30 秒**。30 秒约产生 3000 个 Mel frame，再经 4× 下采样得到约 **750 个 soft token**。这是公开 processor 的 clip 上限，不是“Conformer 数学上只能处理 750 个位置”。超过 30 秒若不自行切块，默认预处理会截断。

### 6. 为什么大模型没有音频塔

01 章那张表：31B 与 26B-A4B 上 `"audio_config": null`。上面的一切大约是 12 层 × 1024 宽加上卷积栈 —— 参数量不大，但它是一个**单独训练的编码器**，需要配对语音数据、自己的评测体系、自己的数值稳定性工作。Gemma 4 把这份投入放进了为手机准备的尺寸里，那里端上语音接口才是重点。一个产品决策，被一个 JSON 里的 `null` 暴露无遗。

## 对照实例：Higgs Audio v3 STT

[Higgs Audio v3 STT](https://huggingface.co/bosonai/higgs-audio-v3-stt) 是很好的对照：它也把连续 audio features 直接插进 decoder-only LLM，但 audio tower 不是 USM/Conformer，而是 **Whisper-Large-v3 encoder**；文本端是 **Qwen3-1.7B-Base**。当前 checkpoint 共约 2.68B 参数，STT 模型卡为 Apache-2.0。下面只讨论这个 STT checkpoint，不讨论同名语音生成模型的 codec/output 分支。

### 1. 先读 config，不靠模型名字猜

发布的 [`config.json`](https://huggingface.co/bosonai/higgs-audio-v3-stt/blob/main/config.json) 给出：

| 部件 | 关键值 |
|---|---|
| Whisper input | 16kHz、128 Mel bins、约 100 frame/s |
| Whisper encoder | 32 layers、20 heads、`d_model=1280`、FFN 5120 |
| 位置上限 | `max_source_positions=1500`（第一次 stride-2 后） |
| tower 输出速率 | `frame_rate=25`，即 AvgPool 后 25 token/s |
| projector | depthwise temporal Conv1D stride 2 + `1280→2048→2048` MLP |
| 给 Qwen 的速率 | `25 / 2 = 12.5` soft token/s |
| Qwen | 28 layers、hidden size 2048、context 32768 |
| 当前 collator | `chunk_size_seconds=4.0`、padding 到 batch 内最长 |

这张表已经揭示一个容易漏掉的事实：Higgs 有**三次时间降采样**，不只是 Whisper 开头那个 stride-2 Conv1D。

### 2. `Log-Mel → Conv1D` 到底算了什么

Higgs/Whisper 输入 layout 是：

\[
X\in\mathbb R^{B\times128\times T}.
\]

第一层：

```python
self.conv1 = nn.Conv1d(128, 1280, kernel_size=3, padding=1)
```

对输出通道 `o` 和时刻 `t`：

\[
h^{(1)}_{o,t}=b_o+\sum_{c=1}^{128}\sum_{\delta=-1}^{1}
W^{(1)}_{o,c,\delta}X_{c,t+\delta}.
\]

所以一个 filter 看的不是 3 个标量，而是 **128 个频带 × 3 个相邻时间帧**，即约 30ms 的局部频谱块。模型有 1280 个这样的 filter，于是每个时刻从“128 个能量值”变成“1280 个学到的声学特征”。padding 使时间长度不变：

```text
[B, 128, T] → Conv1D k3,s1 → [B, 1280, T]
```

第二层：

```python
self.conv2 = nn.Conv1d(1280, 1280, kernel_size=3, stride=2, padding=1)
```

它仍在相邻时间上混合，但只每隔两个位置输出一次：

```text
[B, 1280, T] → Conv1D k3,s2 → [B, 1280, ceil(T/2)]
```

然后 `permute(0, 2, 1)` 得到 Transformer 熟悉的 `[B, sequence, hidden]`，加最多 1500 个 learned position embeddings。10ms/frame 经过 stride 2 后是约 20ms/token、50 token/s。

### 3. Whisper Transformer 输出什么

32 层 Whisper encoder 对每个位置做双向 self-attention 与 FFN：

\[
H=\operatorname{WhisperEncoder}(X),\qquad
H\in\mathbb R^{B\times\lceil T/2\rceil\times1280}.
\]

这里的 `H[t]` 已经不再只表示第 `t` 个 20ms 小窗。它融合了该 chunk 内其他位置的信息，因而是 contextual acoustic representation：可能编码音素、说话方式、上下文消歧以及噪声。它仍然不是 transcript，也不保证与文字一一对齐。

Higgs 随后做：

```python
hidden_states = hidden_states.permute(0, 2, 1)
hidden_states = self.avg_pooler(hidden_states)  # kernel=2, stride=2
hidden_states = hidden_states.permute(0, 2, 1)
```

序列速率从 50 降到 **25 token/s**。所以常见的“Whisper encoder 30 秒输出 1500 个向量”在 Higgs 里少算了一步：裸 Whisper Transformer 的确是 1500，但 `HiggsAudioEncoder.last_hidden_state` 已经 AvgPool 成约 **750**。

### 4. Projector 不是简单的 `1280 → 2048`

当前 checkpoint 使用 `projector_type="mlp"`、`projector_temporal_downsample=2`：

```python
self.temporal = nn.Conv1d(
    1280, 1280, kernel_size=3, stride=2, padding=1,
    groups=1280, bias=True,
)
self.linear1 = nn.Linear(1280, 2048)
self.relu    = nn.ReLU()
self.linear2 = nn.Linear(2048, 2048)
```

`groups=1280` 表示这是 **depthwise temporal convolution**：每个 hidden channel 只在自己的时间邻域里卷积，不在这一步混合通道。时间长度再减半到 12.5 token/s；后面的 MLP 才负责通道混合与进入 Qwen hidden dimension。

一个 8 秒例子的 shape 是：

```text
waveform                       [B, 128000]
Log-Mel                        [B, 128, 800]
conv1 k3,s1                    [B, 1280, 800]
conv2 k3,s2                    [B, 1280, 400]
permute + 32-layer Transformer [B, 400, 1280]
AvgPool1d k2,s2                [B, 200, 1280]
depthwise Conv1d k3,s2         [B, 100, 1280]
MLP                            [B, 100, 2048]
```

这张 ledger 是为了展示长度算术，暂时把 8 秒当成一次 tower call；当前默认 collator 实际会把它拆成两个 4 秒样本分别编码，再按顺序合成同样约 100 个 soft tokens。所以 8 秒约占 100 个 Qwen context 位置；30 秒约占 375 个。Higgs 的 token 成本是 Gemma 4 的一半：12.5/s 对 25/s。

### 5. 一个 `<|AUDIO|>` 怎样变成 100 个向量

官方 [`transcribe.py`](https://huggingface.co/bosonai/higgs-audio-v3-stt/blob/main/transcribe.py) 构造的逻辑 prompt 是：

```text
<|im_start|>user
Transcribe the speech ...
<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>
<|im_start|>assistant
```

`<|AUDIO|>` 只是一个占位符，不是让一个 text embedding 承载整段声音。`merge_input_ids_with_audio_features` 会把这一个位置扩张并替换为整个 soft-token 序列：

```text
instruction embeddings
<|audio_bos|> embedding
A1 A2 A3 ... A100             # 8 秒音频
<|audio_eos|> embedding
assistant marker
```

然后 Qwen 像处理一段很长的 prefix 一样处理它们，自回归生成 transcription。这里与 Gemma 4 一样**没有额外 cross-attention**。原版 Whisper decoder 则不同：它保留 encoder states 为 memory，每一层 decoder 通过 cross-attention 读取。

### 6. 输入永远是“audio + 一段文字”吗

从语义接口看，用户可以只说“转写这个文件”，甚至高级封装可以只接收 audio；默认 instruction 和 chat control tokens 由库补上。从真正送入 decoder 的张量看，它几乎一定是**混合 prefix**：至少有 role/边界/assistant marker，再加 audio soft tokens。

要特别区分两段文字：

- `Transcribe the speech` 是**输入 instruction**；
- `OK, ...` 是模型要生成的**目标 transcription**，推理前并没有提供给模型。

audio 也不必物理上永远排在 instruction 前。只要 instruction 与 audio 都位于 assistant generation point 之前，causal decoder 生成答案时就能看到二者。官方 Higgs helper 选择“instruction 在前、audio 在后”；下面两种 prefix 原理上都可见：

```text
[instruction] [audio] [assistant starts here]
[audio] [instruction] [assistant starts here]
```

但应遵循 checkpoint 训练时的 chat template，不要随意交换。真正的硬约束是：**要影响输出的信息必须出现在待生成输出之前**。

### 7. 30 秒限制到底在哪里

Whisper 的 30 秒来自：100 个 Mel frame/s × 30s = 3000；开头 stride-2 后得到 1500，恰好等于 `max_source_positions=1500`。因此单次 Whisper tower call 的位置表最多覆盖约 30 秒。这是 tower 的窗口上限，不是整个应用只能处理一个 30 秒文件。

当前 Higgs STT collator 更激进：checkpoint 配置为 **4 秒一块**。对 65 秒录音，它大致执行：

```text
65s waveform
  → 4s + 4s + ... + 1s       # 17 chunks
  → 每块独立跑 Whisper tower
  → 每块各自产生 soft tokens
  → 在 <|audio_bos|> ... <|audio_eos|> 内按顺序拼接
  → 一次交给 Qwen 生成全文
```

collator 会把一个 `<|AUDIO|>` 复制成所需数量的 placeholders；当前 `duplicate_audio_triplet=false`，所以通常只复制中间的 `<|AUDIO|>`，外层保留一对 BOS/EOS。

这让输入文件可以长于 30 秒，但带来三个代价：

1. **tower 上下文在每个 4 秒边界重置。** 横跨边界的词和语气只能靠 Qwen 后处理；
2. **LLM context 仍有限。** 32,768 context 要同时容纳约 12.5 audio token/s、instruction 和输出 transcript；
3. **生成长度也有限。** 即使 audio prefix 塞得下，`max_new_tokens` 太小仍会截断全文。

因此长录音的生产方案通常是 VAD/句界切分，加少量 overlap，逐段转写，再做去重、时间戳拼接和语言级校正。固定 4 秒切块解决“tower 吃不下”，VAD 解决“不要切在词中间”，它们不是同一个问题。

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

**先把 Gemma 4 与 Higgs 对齐到同一张数据流：**

| 维度 | Gemma 4 E2B/E4B | Higgs Audio v3 STT |
|---|---|---|
| waveform 前端 | 16kHz，128-bin Log-Mel，10ms hop | 16kHz，Whisper 128-bin Log-Mel，10ms hop |
| 第一阶段卷积 | 两层 2D conv，同时压时间与频率 | 两层 temporal Conv1D，把 128 Mel 当 channels |
| 主编码器 | 12×1024 USM/Conformer | 32×1280 Whisper-Large-v3 Transformer encoder |
| 声学注意力 | 约 480ms 的局部因果 mask，相对位置 | chunk 内双向全注意力，learned absolute position |
| tower 后压缩 | 2D conv 已做时间 4× | Transformer 后 AvgPool 2× |
| LLM projector | tower `1024→1536`，再 norm+linear | depthwise temporal conv 2×，再 `1280→2048→2048` MLP |
| audio token 率 | 25/s（40ms/token） | 12.5/s（80ms/token） |
| 融合 | placeholder 的 text embedding 被 soft tokens 替换 | `<|AUDIO|>` 被可变长 soft-token 序列替换 |
| decoder | Gemma 4 causal LM | Qwen3-1.7B causal LM |
| cross-attention | 无 | 无 |
| 默认长音频行为 | processor 在 30s 截断，应用需自行切块 | 单 tower 窗仍 ≤30s；官方 collator 默认切成 4s 多块 |
| 主要定位 | 通用音频理解、ASR、翻译并与图像/视频组合 | 以 speech-to-text 为中心 |

它们**相同的架构判断**是：声音先变成一串连续 soft tokens，再与文字 prefix 一起交给 decoder-only LLM；语言模型负责统一生成接口。它们**不同的判断**是声学编码器应该为何优化：

- Higgs 复用一个强大的离线 ASR encoder，用 chunk 内双向注意力换更完整的声学上下文，再强力压到 12.5 token/s；
- Gemma 4 使用更局部、较因果的 USM/Conformer 前端，保留 25 token/s 的时间分辨率，并把它作为通用多模态系统的一部分。

再把“原版 Whisper”放回来，三者的 decoder 接法不同：

| 系统 | encoder 输出给谁 | 文本怎样生成 |
|---|---|---|
| 原版 Whisper | 独立 decoder 的 cross-attention memory | Whisper decoder 自回归生成 |
| Higgs STT | 与 instruction 拼成 Qwen prefix | Qwen causal self-attention 生成 |
| Gemma 4 | 与其他模态/文字拼成 Gemma prefix | Gemma causal self-attention 生成 |
| CTC / RNN-T | 对齐头或 transducer joint network | CTC collapse 或增量 token emission |

因而不能简单说“接 LLM 就一定比专用 ASR 差/好”。应分别测 WER/CER、时间戳、热词、噪声、多说话人、首 token 延迟、峰值显存和长音频稳定性。通用模型的独特收益是可组合性：`转写`、`翻译`、`总结`、`判断情绪`、`把讲话与画面核对`可以走同一套 mixed-prefix 机制；专用 ASR 的优势通常是对齐、可控解码和工程吞吐。

**视频：** 领域分成两派 —— 把帧重采样成类文本序列、让 LLM 做时序推理的（Gemma 4，以及多数开源 VLM），和拥有真正时序机件的（3D patch、时序注意力、或时间感知位置编码：Qwen-VL 的 M-RoPE 时间轴，以及 [21 章](../21-video-generation/overview.md)的视频生成模型）。抽帧要简单得多，而且对问答来说够用；对任何需要精确运动的任务则不够。

## 自测

1. 16kHz 音频用 10ms hop，一秒有多少 Mel frame？为什么 20ms window 不意味着每秒只有 50 frame？
2. 从 Gemma 4 的两层 stride-(2,2) conv 推导 25 audio token/s。要改成 20ms/token，至少要改哪一处？
3. Higgs 的 8 秒音频从 800 Mel frame 到 100 Qwen soft tokens，依次是哪三个 2× 下采样？
4. 为什么 `Conv1d(128,1280,kernel_size=3)` 的一个 filter 实际有 `128×3` 个权重，而不是 3 个？
5. Gemma 4 先构造 24 槽 key 候选，为什么一个 query 最终看不到全部 24 个？写出 `dist` mask 判断。
6. `q_scale` 里的 `/log(2)` 在初始化时被什么抵消？为什么不能说 softmax 已经变成以 2 为底？
7. “audio tower 输出被映射到 text space”哪里不精确？hidden space 与 vocabulary space 的边界在哪？
8. 为什么 instruction 可以放在 audio 前或后，却必须放在 assistant 输出之前？
9. 65 秒 Higgs 输入为什么没有违反 Whisper 的 30 秒限制？它又牺牲了什么？
10. 一段 32 帧的视频和一张照片，哪个更费上下文？第 17 帧的时间戳在哪里表示？

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_audio_tower_anatomy.ipynb` | 追踪一段波形经 mel 提取与卷积下采样的形状变化；手写重建分块注意力 mask 并与库 `assert_close` | 🟢 CPU |
| `02_asr_and_video.ipynb` | 真权重 E2B：转写、音频问答、视频问答——视频的 token 成本是测出来的，不是猜的 | 🟡 24GB 显存 |
| <a href="../../05-audio-and-video/notebooks/03_whisper_asr_compare.html"><code>03_whisper_asr_compare.ipynb</code></a> | 设计空间对照组：同一段音频交给 Whisper。专用模型强在哪，又做不到什么 | 🟢 CPU / 🟡 |
