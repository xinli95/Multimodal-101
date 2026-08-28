# FLUX.2 [klein] 4B Deep Dive：沿 Diffusers 拆解图像生成

本节不从模型名词表开始，而是跟着一次真实的 Diffusers 调用，把现代图像生成模型从输入 prompt 一直拆到 RGB 像素。遇到 VAE、Flow Matching、MMDiT、Euler solver 等概念时，再补上理解源码所需要的数学。

读完后，应该能明确回答四个问题：

1. VAE 明明把空间压缩 8 倍，为什么 channel 会从 3 变成 32，进入 Transformer 后又变成 128？
2. 训练时构造的 $z_t$ 到底是什么，$t$ 的范围和边界在哪里？
3. Transformer 所谓“预测速度场”具体输出什么，Euler 方法怎样把噪声一步步变成图片？
4. FLUX.2 的 5 层 double-stream 和 20 层 single-stream Transformer 在代码里分别做了什么？

这里分析的是 Hugging Face Diffusers 的实现。`pipelines/flux2/` 是组装和调度层，真正的神经网络与采样器还分布在另外两个目录：

| 层次 | 主要源码 | 作用 |
|---|---|---|
| Pipeline | [`pipeline_flux2_klein.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py) | prompt 编码、latent 准备、采样循环、VAE 解码 |
| Transformer | [`transformer_flux2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py) | 条件速度场 $v_\theta(z_t,t,c)$ |
| Scheduler | [`scheduling_flow_match_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py) | 时间网格与 Euler 更新 |
| VAE | [`autoencoder_kl_flux2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py) | RGB 与 32-channel latent 互相转换 |
| Model config | [FLUX.2-klein-4B `transformer/config.json`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/transformer/config.json) | 4B checkpoint 的真实层数和宽度 |

`pipeline_flux2.py` 面向完整 FLUX.2 系列，`pipeline_flux2_klein.py` 才是本节的主入口；`pipeline_flux2_klein_inpaint.py` 在相同主干上增加 mask 和初始图，`pipeline_flux2_klein_kv.py` 则为参考图编辑缓存 K/V。先读普通 Klein pipeline，最容易看清模型本体。

## 1. 先建立端到端数据流

以 batch size 1、1024×1024 输出、默认 512 个文本 token 为例：

```text
prompt
  │
  ▼
Qwen3 tokenizer + text encoder
取第 9 / 18 / 27 层并拼接
  │
  └── text states [1, 512, 7680]
                         │
                         ▼
                  Linear 7680 → 3072
                         │
                         ├─────────────────────────┐
Gaussian noise           │                         │ timestep t
[1, 128, 64, 64]         │                         │
  │                      │                         ▼
  ▼                      │                  sinusoidal + MLP
flatten                   │                  shift/scale/gate
[1, 4096, 128]            │                         │
  │                      │                         │
  ▼                      ▼                         │
Linear 128 → 3072   text tokens 3072              │
  │                      │                         │
  └──────────────┬───────┘                         │
                 ▼                                 │
      5 × double-stream block ◀────────────────────┘
      图像/文本两条残差流，联合 attention
                 │
                 ▼
      concatenate text + image tokens
      [1, 512 + 4096, 3072]
                 │
                 ▼
      20 × single-stream block
      联合 attention 与 SwiGLU MLP 并行
                 │
                 ▼
      丢掉 text / reference tokens
      AdaLN + Linear 3072 → 128
                 │
                 └── velocity [1, 4096, 128]
                              │
                              ▼
                    Euler 更新 noisy latent
                              │ 重复约 4 步
                              ▼
                    unpack + unpatchify
                    [1, 32, 128, 128]
                              │
                              ▼
                         VAE decoder
                              │
                              ▼
                    RGB [1, 3, 1024, 1024]
```

最重要的结论是：**Transformer 不直接画 RGB，也不在一次 forward 中吐出最终图片。** 它每次只预测当前位置上的移动方向；scheduler 用这个方向更新 latent，反复数步后，VAE 才把最终 latent 解码成像素。

## 2. 三件套：VAE、文本编码器、速度场 Transformer

现代 latent image generator 通常由三部分组成：

1. **VAE**：学习 RGB 图片与连续 latent 之间的可逆近似映射。生成发生在更小的 latent 网格上。
2. **文本编码器**：把 prompt 变成条件 token。早期模型常用 CLIP/T5，Klein 4B 使用 Qwen3。
3. **生成主干**：过去常是 U-Net；FLUX.2 使用 Transformer，在 latent token 上预测 Flow Matching 的速度场。

还需要一个没有可学习大参数、却决定推理轨迹的组件：**scheduler/solver**。Transformer 给出“方向”，solver 决定用多大的步长、走到哪个时间点。模型与 solver 的职责不能混为一谈。

## 3. VAE：空间压缩 8 倍为什么得到 32 channels

### 3.1 空间尺寸与 channel 是两件事

输入图片的形状是：

$$
x\in\mathbb{R}^{B\times3\times H\times W}.
$$

FLUX.2 VAE 编码 1024×1024 图片后得到：

$$
z_0\in\mathbb{R}^{B\times32\times128\times128}.
$$

“压缩 8 倍”只表示高度和宽度分别变为原来的 $1/8$。它从未承诺 channel 也减少。编码器通常在降低空间分辨率的同时增加 channel，把局部像素块中的颜色、纹理、边缘和语义信息搬到 feature channels 中。

比较元素数量：

$$
3\times1024\times1024=3{,}145{,}728,
$$

$$
32\times128\times128=524{,}288.
$$

整体仍约压缩了 6 倍，只是压缩发生在“空间大幅缩小、channel 适当增多”的组合上。32 个 channel 不是 32 种可人工命名的颜色或纹理，而是 VAE 端到端学出的连续特征基底。真实配置可在 [VAE config](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/vae/config.json) 中核对。

### 3.2 VAE 压缩和 Transformer patchify 不同

VAE 得到 `[B,32,128,128]` 后，pipeline 又做一次 $2\times2$ patchify：

$$
[B,32,128,128]
\rightarrow[B,32\times2\times2,64,64]
=[B,128,64,64].
$$

这一步没有再次学习语义压缩，也没有丢掉元素；它只是 rearrange：把每个相邻 $2\times2$ 位置搬到 channel 维。随后展平空间：

$$
[B,128,64,64]\rightarrow[B,4096,128].
$$

因此：

- **32** 是 VAE latent channel；
- **128** 是一个 $2\times2$ latent patch 展开后的 token width；
- **4096** 是 $64\times64$ 个 image tokens；
- **3072** 才是经过线性投影后的 Transformer hidden size。

四个数字属于四个不同概念。`_patchify_latents()`、`_pack_latents()` 与逆操作集中在 [Klein pipeline 的 latent utilities](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L370-L426)。

### 3.3 为什么 T2I 直接采样 128-channel noise

训练图片要先经过 VAE，因此数据端自然产生 `[B,32,128,128]`，再 patchify。纯文生图推理没有输入图片，Diffusers 可以直接在等价的 packed 表示中采样：

```text
Gaussian noise [B, 128, 64, 64]
             → [B, 4096, 128]
```

它与先采样 `[B,32,128,128]` 再做固定 rearrange 在数学上等价，却少一步张量变换。生成循环始终停留在这个 packed latent 空间；只有解码前才 unpatchify 回 32 channels。

## 4. Flow Matching 的 prerequisite：状态、路径和速度

### 4.1 先不要把 $z_t$ 当作一张“半成品图片”

设：

- $z_0$：真实图片经 VAE 得到的数据 latent；
- $\epsilon\sim\mathcal N(0,I)$：与 $z_0$ 同形状的高斯噪声；
- $t\in[0,1]$：连续时间。

最简单的 Rectified Flow 路径是线性插值：

$$
z_t=(1-t)z_0+t\epsilon.
$$

边界非常清楚：

$$
t=0\Rightarrow z_t=z_0,
\qquad
t=1\Rightarrow z_t=\epsilon.
$$

$z_t$ 的严格含义是：**在预先规定的概率路径上，时间 $t$ 对应的随机状态。** 它通常可以被视觉化为不同噪声程度的 latent，但“30% 噪声图片”只是直觉，不是定义。每个坐标都同时包含数据 latent 与随机噪声的线性组合。

训练不需要真的从 $t=0$ 逐步模拟到 $t=1$。一次训练样本可以直接：

1. 从数据集中取图片并编码为 $z_0$；
2. 独立采样 $\epsilon\sim\mathcal N(0,I)$；
3. 随机采样一个 $t\in[0,1]$；
4. 一步计算出 $z_t$；
5. 让网络预测该位置的目标速度。

因此训练成本不是“每张图走完整条轨迹”。在大量图片、噪声和时间样本上做随机覆盖后，模型才逐渐学出整个空间中的向量场。

### 4.2 速度标签从哪里来

对路径求导：

$$
\frac{d z_t}{dt}=\epsilon-z_0.
$$

于是最基本的 Flow Matching loss 是：

$$
\mathcal L
=\mathbb E_{z_0,\epsilon,t}
\left[
\left\|v_\theta(z_t,t,c)-(\epsilon-z_0)\right\|_2^2
\right],
$$

其中 $c$ 是 prompt 条件。网络输入当前状态 $z_t$、时间 $t$ 和文本条件，输出一个与 $z_t$ 同形状的速度。

单个训练 pair 的直线路径速度确实是常量 $\epsilon-z_0$；困难在于推理时只给模型某个 $z_t$，它不知道该状态最初配对的是哪张训练图片和哪份噪声。它必须从海量数据中学习条件期望意义上的局部方向：在这个位置、这个时间、这个 prompt 下，往哪里移动最可能抵达数据分布。

### 4.3 “图片到噪声”与“噪声到图片”都对

按上面的时间约定，正向时间是：

```text
t = 0                                  t = 1
data latent  ───────────────────────►  Gaussian noise
```

所以模型学习的速度可以称为“图片到噪声”的速度场。推理则从 $t=1$ 出发，沿同一个微分方程向较小的 $t$ 积分：

```text
t = 1                                  t = 0
Gaussian noise ─────────────────────►  generated latent
```

不是重新训练了一个反向模型，而是把时间步长取成负数。类似于同一张地图上的箭头：顺着时间走从数据到噪声；逆着时间积分就从噪声回到数据。

### 4.4 为什么代码里的 timestep 看起来是 0 到 1000

数学推导通常使用 $t\in[0,1]$。Diffusers scheduler 为兼容训练时间索引，会把它表示成近似 $[0,1000]$ 的数：

$$
\text{timestep}=1000\sigma.
$$

不要把代码里的 `1000` 理解为必须走 1000 次：

- 连续时间范围仍是 $[0,1]$；
- 1000 是数值表示尺度；
- `num_inference_steps=4` 才表示实际调用 Transformer 四次；
- scheduler 还会根据 image token 数对 $\sigma$ 网格做 shift，所以四个采样点未必等距。

时间网格的构造见 [`set_timesteps()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L283-L370)。

## 5. 推理：条件向量场与 Euler 方法

### 5.1 条件向量场是什么

对任意状态 $z$ 和时间 $t$，网络给每个 latent 坐标分配一个速度：

$$
v_\theta(z,t,c).
$$

把整个 latent 空间想成一张极高维地图，每个位置都有一个箭头：

- 箭头方向：下一瞬间往哪里走；
- 箭头长度：移动多快；
- $t$：当前噪声阶段；
- $c$：prompt 条件。

换一个 prompt，箭头地图也会改变，所以叫**条件向量场**。同一份初始噪声，在“红色跑车”和“水彩鲸鱼”两个条件下会被导向不同的数据区域。

### 5.2 Euler 就是“当前位置速度 × 一小段时间”

连续动力系统满足：

$$
\frac{dz}{dt}=v_\theta(z,t,c).
$$

Euler 方法用局部直线近似：

$$
z_{k+1}=z_k+\Delta t_k\,v_\theta(z_k,t_k,c).
$$

采样从噪声端向数据端进行，因此 $\Delta t_k<0$，正好把“图片到噪声”的速度反向使用。Diffusers 采用 $\sigma$ 作为数值坐标，核心更新是：

$$
z_{k+1}
=z_k+(\sigma_{k+1}-\sigma_k)v_\theta(z_k,t_k,c).
$$

源码最终就是 `sample + dt * model_output`，位于 scheduler 的 [`step()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L423-L522)。Pipeline 中变量仍叫 `noise_pred`，但在 Flow Matching 语义中它是 velocity/vector field，不应按传统 DDPM 的纯噪声预测来理解。

### 5.3 四步采样不等于只学四个时间点

训练时 $t$ 是连续随机变量，模型学习整个时间范围上的速度。推理时 solver 只挑若干离散点近似积分。普通模型使用更多点会降低离散误差；Klein 4B 经过步数/指导蒸馏，专门把多步教师轨迹压进约四步学生轨迹，才能在极稀疏的时间网格上维持质量。

## 6. Prompt：Qwen3 如何变成 7680 维条件

[`_get_qwen3_prompt_embeds()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L208-L262) 做了五件事：

1. 用 chat template 包装 prompt，并关闭 thinking 输出；
2. tokenize、padding/truncation，默认最多 512 tokens；
3. 运行 Qwen3 text encoder 并请求所有 hidden states；
4. 取第 9、18、27 层；
5. 在 feature 维拼接。

Qwen3 每层 hidden size 为 2560，因此：

$$
3\times2560=7680,
\qquad
c_{\text{text}}\in\mathbb R^{B\times L\times7680}.
$$

取多个深度的直觉是把较局部的词法信息、中层组合关系和深层整体语义一并暴露给生成主干，而不是强迫最后一层独自保留所有细节。进入 Flux Transformer 后，`context_embedder` 再把它投影到共同 hidden size：

$$
[B,L,7680]\rightarrow[B,L,3072].
$$

## 7. FLUX.2 Transformer：速度场网络本体

Klein 4B checkpoint 的关键配置是：

| 参数 | 值 | 推导/含义 |
|---|---:|---|
| `in_channels` | 128 | 一个 $2\times2$ VAE latent patch |
| `joint_attention_dim` | 7680 | 三层 Qwen3 hidden state 拼接 |
| `num_attention_heads` | 24 | attention heads |
| `attention_head_dim` | 128 | 每个 head 的维度 |
| Transformer hidden size | 3072 | $24\times128$ |
| `num_layers` | 5 | double-stream blocks |
| `num_single_layers` | 20 | single-stream blocks |
| `mlp_ratio` | 3 | FFN 中间宽度 9216 |
| `axes_dims_rope` | `[32,32,32,32]` | 四轴 RoPE，共 128 维 |
| `guidance_embeds` | false | 4B 不向主干传独立 guidance embedding |

注意 `Flux2Transformer2DModel` 类定义中的 Python 默认值服务于多个 FLUX.2 变体，不等于 Klein 4B 的真实配置；研究具体 checkpoint 必须读其 `config.json`。

### 7.1 三路输入如何对齐

图像 token：

$$
[B,4096,128]\xrightarrow{W_x}[B,4096,3072].
$$

文字 token：

$$
[B,L,7680]\xrightarrow{W_c}[B,L,3072].
$$

时间 $t$：

```text
t → 256-d sinusoidal embedding → MLP → 3072-d time embedding
                                    → shift / scale / gate
```

图像和文字被投影到相同宽度后才能做联合 attention；时间不作为普通 token 拼进去，而是通过 AdaLN 风格的 modulation 改变每层计算。初始化路径可在 [`Flux2Transformer2DModel.__init__`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1059-L1219) 中核对。

### 7.2 前 5 层：double-stream MMDiT

Double-stream block 保留两条 residual stream：

$$
X_i\in\mathbb R^{B\times N_i\times3072},
\qquad
X_t\in\mathbb R^{B\times N_t\times3072}.
$$

图像和文字先分别做 modulation、QKV projection：

$$
Q_i,K_i,V_i=f_i(X_i),
\qquad
Q_t,K_t,V_t=f_t(X_t).
$$

随后沿 token 维拼接：

$$
Q=[Q_t;Q_i],\quad K=[K_t;K_i],\quad V=[V_t;V_i],
$$

再执行一次完整 self-attention：

$$
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)V.
$$

输出按 token 数拆回文字与图片，两边分别经过自己的 output projection、gate 和 FFN。结果是：

- image query 可以读取 text key/value；
- text query 也可以读取 image key/value；
- 两种模态参与同一注意力图；
- 但它们仍保留不同参数和 residual stream。

这就是 MMDiT 中“multi-modal”与“double-stream”同时成立的方式。Block 见 [`Flux2TransformerBlock`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L876-L968)，联合 QKV 见 [`Flux2AttnProcessor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L327-L395)。

### 7.3 后 20 层：single-stream joint Transformer

五层双流交互后，代码直接拼接：

$$
X=[X_t;X_i].
$$

默认 T2I shape 是：

$$
[B,512+4096,3072]=[B,4608,3072].
$$

从这里起，网络不再用两套 block 参数显式区分文字与图像；区别由 token 内容和四维位置坐标保留。

Klein 的 single-stream block 还把 attention 与 MLP 做成并行分支：

$$
A=\operatorname{Attention}(X),
\qquad
M=\operatorname{SwiGLU}(X),
$$

$$
Y=W_o[A;M],
\qquad
X\leftarrow X+g(t)\odot Y.
$$

SwiGLU 的中间宽度为 $3072\times3=9216$。实现见 [`Flux2SingleTransformerBlock`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L807-L873) 与 fused parallel processor [对应代码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L572-L630)。

### 7.4 四维 RoPE：$(T,H,W,L)$

普通语言模型只需要 token 序列位置。FLUX.2 要同时表达文字、多张参考图和二维图像网格，因此给每个 token 四个坐标：

$$
(T,H,W,L).
$$

Diffusers 构造的典型 ID 是：

| token 类型 | 坐标 |
|---|---|
| 文本第 $l$ 个 token | $(0,0,0,l)$ |
| 待生成图像位置 $(h,w)$ | $(0,h,w,0)$ |
| 第一张参考图位置 $(h,w)$ | $(10,h,w,0)$ |
| 第二张参考图位置 $(h,w)$ | $(20,h,w,0)$ |

四个轴各使用 32 个旋转维度：

$$
32+32+32+32=128=\text{head dim}.
$$

因此 attention 可以同时感知文本顺序、二维相对位置和参考图身份。ID 生成位于 [`_prepare_text_ids()` / `_prepare_latent_ids()` / `_prepare_image_ids()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L266-L366)，RoPE 实现在 [`Flux2PosEmbed`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L971-L998)。

### 7.5 时间调制：同一组参数如何服务所有噪声阶段

时间 embedding 被转换成若干组 shift、scale、gate。核心形式是：

$$
\hat X=\operatorname{Norm}(X)\odot(1+\operatorname{scale}(t))
+\operatorname{shift}(t),
$$

$$
X\leftarrow X+\operatorname{gate}(t)\odot F(\hat X).
$$

噪声很大时，网络更需要确定全局构图和语义；接近数据端时，更需要处理纹理与边缘。时间调制让同一套 Transformer 权重随噪声阶段改变工作模式。实现位于 [`Flux2TimestepGuidanceEmbeddings` 和 `Flux2Modulation`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1001-L1056)。

### 7.6 输出头为何仍是 128 维

single-stream blocks 结束后，代码先移除文本 token；若有参考图，也移除参考图 token，只保留待生成目标的 $N_i$ 个位置。最后经过 AdaLN 和线性层：

$$
[B,N_i,3072]\rightarrow[B,N_i,128].
$$

速度必须与被更新的 latent 同形状，Euler 才能执行逐元素加法。完整 forward 可沿 [`Flux2Transformer2DModel.forward`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1226-L1424) 阅读。

## 8. 对照 `pipeline.__call__()` 阅读一次生成

把数学对象映射回 [`Flux2KleinPipeline.__call__()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L614-L919)。省略设备管理、callback 和异常检查后，逻辑近似为：

```python
prompt_tokens = encode_prompt(prompt)       # [B, L, 7680]
z = prepare_gaussian_latents()              # [B, 4096, 128]
ids = prepare_latent_ids()                  # [B, 4096, 4]
timesteps = scheduler.set_timesteps(steps)

for t in timesteps:
    velocity = transformer(
        hidden_states=z,
        timestep=t,
        encoder_hidden_states=prompt_tokens,
        img_ids=ids,
        txt_ids=text_ids,
    )                                       # [B, 4096, 128]
    z = scheduler.step(velocity, t, z)

z = unpack_and_unpatchify(z)                # [B, 32, 128, 128]
image = vae.decode(z)                        # [B, 3, 1024, 1024]
```

逐段对应真实实现：

1. [`encode_prompt()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L428-L460) 产生文字 hidden states 与 text IDs。
2. [`prepare_latents()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L479-L510) 采样 packed Gaussian latent。
3. Pipeline 根据 image sequence length 和步数计算 scheduler shift，建立离散 $\sigma$ 网格。
4. 每一步调用 Transformer 预测 velocity；若有参考图，先把目标与参考 tokens 拼接。
5. Scheduler 用 Euler step 只更新目标 latent。
6. 最终 latent 经反归一化、unpack、unpatchify 和 VAE decode 回到 RGB。

## 9. 生成与编辑为什么能共用同一个模型

有参考图时，pipeline 先用 VAE 编码参考图片，再在每一步拼接：

$$
X_{\text{image}}=[X_{\text{target noisy}};X_{\text{reference clean}}].
$$

Transformer 可以通过联合 attention 读取参考图；输出后 pipeline 只截取目标 token 对应的速度，参考 latent 本身不被 scheduler 更新。参考图通过不同 $T$ 坐标区分，因此多参考图不需要每张图一套 encoder。

这使文生图和编辑成为同一条计算图的两个特例：

- T2I：text + noisy target；
- reference generation/editing：text + noisy target + clean references；
- inpainting：再增加初始图、mask 与 strength/timestep 逻辑。

标准 pipeline 每一步都会重新计算参考 token 的 K/V；KV 变体会缓存不随步骤变化的参考部分，减少多参考图编辑的重复计算。

## 10. Guidance、蒸馏与 Klein 的四步生成

传统 Classifier-Free Guidance 做有条件和无条件两次 forward：

$$
v_{\text{cfg}}
=v_{\text{uncond}}
+s\left(v_{\text{cond}}-v_{\text{uncond}}\right).
$$

$s>1$ 会把轨迹推向更符合 prompt 的方向，但将 Transformer 计算量近似翻倍，也可能造成过饱和。

Klein 4B 是蒸馏模型。Diffusers 只有在 `guidance_scale > 1 and not is_distilled` 时才执行正负 prompt 两次 forward；对蒸馏 Klein，传统 CFG 分支被跳过。因此通用 pipeline 函数签名中的 `guidance_scale=4.0` 和 `num_inference_steps=50` 不能直接当成 4B 的推荐设置；[官方模型卡](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)示例使用约四步。

一次典型 Klein 4B T2I 的大模块调用量近似是：

```text
Qwen3 encode × 1
Flux Transformer × 4
VAE decode × 1
```

速度快不仅因为“模型较小”，还因为蒸馏同时减少了积分步数和 CFG 的双 forward。

## 11. 这个速度场为什么需要大量数据

需要学习的不是一条从某张图片到某份噪声的曲线，而是：

$$
(z_t,t,c)\mapsto v.
$$

输入空间同时包含几乎任意图像内容和风格、全部噪声阶段、语言中的物体与关系、不同分辨率和参考图组合。数据需求来自“覆盖高维状态空间”，但每张图片可以通过不断重采样 $\epsilon$ 和 $t$ 产生大量不同的训练状态；强文本编码器又把部分语言和世界知识迁移进来。

Transformer 架构提供容量和归纳偏置，却不会自动创造没有在数据中学到的概念。训练数据量、caption 质量、审美过滤、文字渲染样本和编辑三元组，都会直接决定向量场在哪些区域可靠。这也是为什么架构近似的模型，在 prompt 遵循、文字渲染和人物一致性上仍会出现巨大差异。

## 12. 源码阅读顺序

第一次阅读不建议从某个 attention processor 的中间开始：

1. [`Flux2KleinPipeline.__init__`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L155-L205)：组件与尺度。
2. [`Flux2KleinPipeline.__call__`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L614-L919)：端到端控制流。
3. `_get_qwen3_prompt_embeds()` 与 `prepare_latents()`：两路输入 shape。
4. [`Flux2Transformer2DModel.__init__`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1059-L1219)：配置如何变成模块。
5. [`Flux2Transformer2DModel.forward`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1226-L1424)：真实数据流。
6. `Flux2TransformerBlock`：双流如何联合 attention。
7. `Flux2SingleTransformerBlock`：合流后的并行 attention/MLP。
8. scheduler `step()`：模型输出如何真正改变 latent。
9. VAE decode：最后如何回到 RGB。

## 13. Mental-model checkpoint

```text
训练：
真实图片 ─VAE→ z₀
随机噪声 ─────→ ε
随机时间 ─────→ t
zₜ = (1-t)z₀ + tε
Transformer 学习 vθ(zₜ,t,prompt) ≈ ε-z₀

推理：
Gaussian noise z₁
  └─ Transformer 预测当前位置速度
      └─ Euler 用负时间步更新 latent
          └─ 重复少量步骤得到 z₀
              └─ unpack / unpatchify / VAE decode → RGB
```

FLUX.2 [klein] 4B 因而可以概括为：**以 Qwen3 提供文本条件，以 5 层双流 MMDiT 加 20 层单流 Transformer 表示条件速度场，在 32-channel VAE latent 的 $2\times2$ patch tokens 上执行蒸馏后的少步 Rectified Flow 反向积分。**
