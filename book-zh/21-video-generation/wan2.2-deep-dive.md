# Wan 2.2 Deep Dive：沿源码逐步拆解视频生成

本节选择 **Wan 2.2-TI2V-5B** 作为主线，用一次真实生成调用把现代视频生成模型拆开。我们不会先罗列所有公式，而是从可运行的推理路径出发：入口、配置、文本编码、视频 latent、Video DiT、Flow Matching、VAE 解码；遇到一个模块，再补上理解它所必需的理论。

这里研究的是当前公开代码和权重的 Wan 2.2，而不是阿里云 API 中更新但未开放权重的 Wan 2.6/2.7。5B 版本也不是最终目标：它先帮助我们看清单个 DiT 的完整主干，随后再进入 A14B 的高噪声/低噪声双专家设计。

> **状态说明（核对于 2026-08-26）：** Wan 官方 GitHub 组织中最新公开的基础视频生成仓库仍是 Wan 2.2，代码与模型采用 Apache-2.0；更高版本的产品/API 编号不等于已有同版本开源权重。因此“最新线上 Wan”与“最新可研究的官方 open-weight Wan”要分开回答。

### Wan 2.2 家族先览

| 官方 checkpoint | 输入 → 输出 | 本章中的位置 |
|---|---|---|
| TI2V-5B | 文本或文本+首帧 → 720p、24 FPS 视频 | 主线；最适合学习统一 T2V/I2V、高压缩 VAE 与 dense DiT |
| T2V-A14B | 文本 → 480p/720p 视频 | 第 10 步；高/低噪声双专家 |
| I2V-A14B | 文本+首帧 → 480p/720p 视频 | 第 10 步；独立的 channel-concat I2V 接口 |
| S2V-14B | 文本+参考图+语音/音频 → 说话或演唱视频 | 在共享 DiT/VAE 基础上增加音频驱动，适合后续专题 |
| Animate-14B | 角色素材+动作/姿态条件 → 角色动画或替换 | 需要专门预处理与控制条件，适合后续专题 |

本章要“讲透”的边界是 Wan 2.2 的通用生成底座以及 T2V/I2V 两条核心路径；S2V 与 Animate 复用大量底座，但其音频、姿态、mask 和人物素材预处理足以各自成为独立章节。

## 阅读路线

| 步骤 | 问题 | 主要源码 |
|---|---|---|
| 1 | 一次生成中，数据和 Tensor 如何流动？ | `wan_ti2v_5B.py`, `textimage2video.py` |
| 2 | CLI 如何选择任务、配置并组装 pipeline？ | `generate.py` |
| 3 | Prompt 如何变成条件序列？ | `modules/t5.py` |
| 4 | 视频为什么先压缩为 latent？ | `modules/vae2_2.py` |
| 5 | 视频 latent 如何变成时空 token？ | `modules/model.py` |
| 6 | 一个 Wan Transformer block 做了什么？ | `modules/model.py`, `modules/attention.py` |
| 7 | 时间步如何调制每个 block？ | `modules/model.py` |
| 8 | Flow Matching、solver 与 CFG 如何组成采样循环？ | `utils/fm_solvers*.py` |
| 9 | I2V 如何固定首帧、只生成其余区域？ | `textimage2video.py` |
| 10 | A14B 的双专家 MoE 如何按噪声阶段切换？ | `text2video.py`, `image2video.py` |
| 11 | 如何把推理理解迁移到 LoRA 和训练目标？ | 推理接口与社区训练栈 |
| 12 | FSDP、Ulysses 与 offload 分别解决什么？ | `distributed/`, pipeline 初始化代码 |
| 13 | 如何用 shape trace、消融和故障表验证理解？ | 配置与各模块边界 |

官方源码入口：[Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)。建议按下列顺序打开文件：

1. [`generate.py`](https://github.com/Wan-Video/Wan2.2/blob/main/generate.py)：命令行入口。
2. [`wan/configs/wan_ti2v_5B.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)：5B 模型配置。
3. [`wan/textimage2video.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/textimage2video.py)：T2V/I2V pipeline 与采样循环。
4. [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)：Video DiT 主干。
5. [`wan/modules/vae2_2.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py)：视频 VAE。
6. [`wan/utils/fm_solvers_unipc.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/fm_solvers_unipc.py)：Flow Matching solver。

## 第 1 步：端到端数据流

先不看实现细节，把一次 5 秒左右的 T2V 生成压缩成一张图：

```text
文本 prompt
   │
   ▼
UMT5-XXL Text Encoder
   │ context: [文本长度, 4096]
   ▼
文本投影 4096 → 3072 ─────────────────────┐
                                          │
高斯噪声视频 latent                       │
[48, 31, 44, 80]                          │
   │                                      │
   ▼                                      │
3D Patch Embedding                        │
Conv3d(kernel=1×2×2, stride=1×2×2)        │
   │                                      │
   ▼                                      │
27,280 个 video token，每个 3,072 维       │
   │                                      │
   ▼                                      │
30 × Wan Transformer Block ◀──────────────┘
   │
   ▼
预测 flow / velocity
   │
   ▼
UniPC 或 DPM++ 更新 latent，约 50 步
   │
   ▼
干净视频 latent [48, 31, 44, 80]
   │
   ▼
Wan 2.2 Video VAE Decoder
   │
   ▼
RGB 视频 [3, 121, 704, 1280]
```

### 1.1 配置先告诉了我们什么

`wan_ti2v_5B.py` 中最重要的参数是：

```python
vae_stride = (4, 16, 16)
patch_size = (1, 2, 2)

dim = 3072
ffn_dim = 14336
num_heads = 24
num_layers = 30

sample_fps = 24
sample_steps = 50
frame_num = 121
guide_scale = 5.0
```

这里有两层完全不同的“压缩”：

```text
RGB 视频
  └─ Video VAE 压缩：(4, 16, 16)
       └─ DiT patchify：(1, 2, 2)
```

VAE 把像素空间压到更小的连续 latent 空间；patchify 不学习新的语义压缩，只把 latent 网格整理成 Transformer token。

### 1.2 为什么生成 121 帧

Wan 2.2 的因果视频 VAE 要求帧数满足：

$$
F = 4n + 1.
$$

latent 的时间长度为：

$$
F_z = \frac{F-1}{4}+1.
$$

代入 $F=121$：

$$
F_z = \frac{121-1}{4}+1=31.
$$

因此 121 帧 RGB 视频被压成 31 个 latent 时间位置。它不是简单的 $121/4$：因果 VAE 单独处理第一帧，再以四帧为单位处理后续内容。121 帧按 24 FPS 播放约为 5.04 秒。

### 1.3 为什么“720p”实际使用 704×1280

空间尺寸必须同时适配 VAE 的 16 倍压缩和 DiT 的 2 倍 patchify：

```text
总对齐倍数 = 16 × 2 = 32
```

1280 能被 32 整除，720 不能；与它接近的 704 可以：

```text
1280 / 32 = 40
704  / 32 = 22
```

所以这里的“720p”描述的是分辨率档位，而实际 Tensor 使用满足网络对齐要求的 704×1280。

### 1.4 从 RGB 视频到 VAE latent

Wan 2.2 的 5B 新 VAE 使用 48 个 latent channel。对于形状：

```text
[C, T, H, W] = [3, 121, 704, 1280]
```

时间压缩 4 倍、空间压缩 16 倍后得到：

```text
C_z = 48
T_z = (121 - 1) / 4 + 1 = 31
H_z = 704 / 16 = 44
W_z = 1280 / 16 = 80

latent = [48, 31, 44, 80]
```

T2V 没有待编码的输入视频，所以推理从同形状的高斯噪声开始：

```python
noise = torch.randn(48, 31, 44, 80)
```

I2V 则会先把首帧编码进这个 latent 网格；第 9 步再具体拆解它怎样通过 mask 在每次更新后重新固定条件区域。

### 1.5 从 latent 网格到 27,280 个 token

`WanModel` 使用一个 3D 卷积完成 patch embedding：

```python
self.patch_embedding = nn.Conv3d(
    in_channels=48,
    out_channels=3072,
    kernel_size=(1, 2, 2),
    stride=(1, 2, 2),
)
```

它不再压缩时间，只把 latent 空间中相邻的 $2\times2$ 位置合为一个 token：

```text
输入 latent       [48,   31, 44, 80]
Conv3d 输出       [3072, 31, 22, 40]
flatten + transpose
Transformer 输入  [31 × 22 × 40, 3072]
                 = [27280, 3072]
```

也就是说，一段约五秒的视频已经对应 27,280 个视频 token。全局 self-attention 的理论复杂度随 $L^2$ 增长：

$$
L^2=27280^2\approx 7.44\times10^8.
$$

这是**每个 attention head** 可能涉及的位置对数量。FlashAttention 避免把完整矩阵写入显存，但不会消除其主要计算量；这解释了为什么视频 DiT 很快就需要序列并行、稀疏注意力、蒸馏或更强的 VAE 压缩。

### 1.6 文本条件如何对齐到 DiT hidden size

Wan 使用 UMT5-XXL Encoder，最大文本长度 512，输出宽度 4096：

```text
prompt → UMT5 → context [L_text, 4096], L_text ≤ 512
```

进入 `WanModel` 后，它经过两层 MLP：

```text
Linear(4096, 3072) → GELU → Linear(3072, 3072)
```

视频 token 和文本 token 不直接拼接。每个 Wan block 依次执行：

1. 视频 token 之间的 self-attention；
2. 视频 query 对文本 context 的 cross-attention；
3. 逐 token 的 FFN。

第一层直觉是：self-attention 维护画面内部和跨帧关系，cross-attention 注入 prompt 语义。二者在训练后会共同承担构图、运动和语义对齐，不能被严格割裂。

### 1.7 时间步不是一个普通 token

每次采样时，模型都必须知道当前 latent 的噪声等级。时间步 $t$ 先变成 256 维正弦编码，再投影到模型宽度：

```text
t → sinusoidal embedding 256
  → MLP 3072
  → projection 6 × 3072
```

每个 Transformer block 使用这六组量分别调制 self-attention 和 FFN 的 shift、scale、gate。其核心形式近似为：

$$
y=\operatorname{Norm}(x)(1+\text{scale})+\text{shift},
$$

$$
x\leftarrow x+\text{gate}\cdot\operatorname{Module}(y).
$$

这是 AdaLN/AdaLN-Zero 风格的条件调制：不同噪声阶段可以让同一个 block 表现出不同的计算行为。

### 1.8 单次 DiT forward

暂时忽略 padding、并行和混合精度，主干可以写成：

```python
def forward(video_latent, timestep, text_context):
    x = patch_embedding(video_latent)
    x = flatten_to_tokens(x)

    time_emb = encode_timestep(timestep)
    text_emb = project_text(text_context)

    for block in blocks:
        x = x + gated_self_attention(x, time_emb)
        x = x + cross_attention(x, text_emb)
        x = x + gated_ffn(x, time_emb)

    prediction = output_head(x, time_emb)
    return unpatchify(prediction)
```

5B 配置包含 30 个 block、24 个 attention head；每个 head 的维度为 $3072/24=128$。

### 1.9 CFG 为什么让每个采样步执行两次 DiT

Classifier-Free Guidance 分别计算正向 prompt 与 negative/unconditional prompt：

```python
pred_cond = model(latent, t, prompt)
pred_uncond = model(latent, t, negative_prompt)

prediction = pred_uncond + guidance_scale * (
    pred_cond - pred_uncond
)
```

对应公式：

$$
v=v_{\text{uncond}}+s\left(v_{\text{cond}}-v_{\text{uncond}}\right).
$$

默认 $s=5.0$。源码把预测变量命名为 `noise_pred`，但 solver 的 `prediction_type` 是 `flow_prediction`；其理论语义更接近 Flow Matching 的速度场，而不一定是传统 DDPM 中的纯噪声 $\epsilon$。

CFG 的代价非常直接：50 个采样步通常需要约 100 次大型 DiT forward。它也是 CFG 蒸馏和 guidance distillation 能显著提速的原因。

### 1.10 第一步检查点

继续阅读前，应当能够不看源码写出下面四组形状：

```text
RGB video     [3, 121, 704, 1280]
VAE latent    [48, 31, 44, 80]
Video tokens  [27280, 3072]
Text context  [≤512, 4096] → [512, 3072]
```

以及最小采样循环：

```python
latent = random_noise()

for t in timesteps:
    cond = model(latent, t, prompt)
    uncond = model(latent, t, negative_prompt)
    velocity = uncond + cfg * (cond - uncond)
    latent = scheduler.step(velocity, t, latent)

video = vae.decode(latent)
```

下一步从 `generate.py` 的 `--task ti2v-5B` 开始，追踪 CLI 参数如何落到 `WAN_CONFIGS`，如何实例化 `WanTI2V`，再进入这里看到的采样循环。

## 第 2 步：从 CLI 走到 pipeline

先准备官方 native inference 环境（Python 3.10+，PyTorch 至少 2.4）：

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt

pip install 'huggingface_hub[cli]'
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir ./Wan2.2-TI2V-5B
```

`requirements.txt` 包含 FlashAttention；若它与其他依赖一起安装失败，官方建议先安装其余包，再单独处理与当前 CUDA/PyTorch 匹配的 `flash_attn`。checkpoint 很大，下载前也应检查磁盘空间。以下所有命令均假设当前目录是克隆出的官方仓库。

先看最小的官方 5B T2V 调用：

```bash
python generate.py \
  --task ti2v-5B \
  --size '1280*704' \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --prompt 'A paper boat crosses a rain-filled neon street.'
```

它经过四层分发：

```text
命令行参数
  → _validate_args：检查任务、尺寸、帧数、checkpoint 和并行设置
  → WAN_CONFIGS['ti2v-5B']：补齐默认 steps / shift / CFG / FPS
  → WanTI2V(config, checkpoint_dir, ...)：装配 T5、VAE、DiT
  → WanTI2V.generate(..., img=None)：选择 t2v() 分支
```

若同一条命令增加 `--image input.jpg`，最后一步会改走 `i2v()`。这就是 **TI2V** 中 `TI` 的含义：同一个 5B checkpoint、同一个 DiT 接口，同时承担 T2V 与 I2V，而不是把两个独立模型打包在一起。

### 2.1 配置、运行参数与模型参数不要混为一谈

三类量具有不同生命周期：

| 类型 | 例子 | 改变它会怎样 |
|---|---|---|
| 架构参数 | `dim=3072`, `num_layers=30`, `patch_size=(1,2,2)` | 必须与 checkpoint 匹配，不能在推理时随意改 |
| 采样参数 | `sample_steps=50`, `sample_shift=5`, `guide_scale=5` | 不改权重，但改变速度、轨迹与结果 |
| 输出参数 | `frame_num=121`, `size=1280*704`, `seed` | 改变输出形状或初始噪声，也会影响显存与计算量 |

`frame_num` 必须满足 $4n+1$；5B 支持的横竖屏尺寸是 `1280*704` 与 `704*1280`。I2V 中 `size` 更接近“最大面积档位”：`best_output_size` 会按输入图像宽高比，寻找同时满足总对齐倍数 32 的实际尺寸。

### 2.2 初始化到底加载了什么

`WanTI2V.__init__` 依次装配：

1. UMT5-XXL tokenizer 与 encoder；
2. `Wan2.2_VAE.pth` 对应的 48-channel 高压缩 VAE；
3. 一个 `WanModel.from_pretrained(checkpoint_dir)` 读取的 5B dense Video DiT；
4. 可选 FSDP、Ulysses sequence parallel、CPU offload 与 dtype 转换。

整个 pipeline 没有 CLIP 图像编码器。5B I2V 的图像条件直接由视频 VAE 编成 clean video latent，再通过 mask 和 token-wise timestep 注入。

### 2.3 prompt extension 不属于生成主干

`--use_prompt_extend` 可以先用本地 Qwen 或 DashScope 把短 prompt 扩写成镜头、主体、动作和氛围更完整的文本。它是前处理器，不参加 DiT 的每一步去噪，也不是 Wan 权重的一部分。调试模型本身时应先关闭它；否则同一个输入字符串可能在进入 T5 前已经被改写。

### 2.4 入口层检查点

阅读 `generate.py` 时，始终追踪下面五个变量即可：

```text
args.task       → 选择 WanTI2V / WanT2V / WanI2V / WanS2V / WanAnimate
cfg             → checkpoint 对应的不可变架构与默认采样配置
args.image      → 对 TI2V-5B 决定 T2V 还是 I2V
args.size       → T2V 的实际尺寸，I2V 的最大面积档位
args.base_seed  → 唯一决定初始随机 latent 的显式随机源
```

## 第 3 步：UMT5 如何把 prompt 变成条件

Wan 2.2 使用 **UMT5-XXL encoder**，不是常见的 CLIP text encoder。UMT5 是 encoder-decoder T5 家族的多语言版本；Wan 只使用 encoder 端，取得每个文本 token 的 contextual representation。

```text
prompt 字符串
  → SentencePiece tokenizer
  → token ids + attention mask，最大长度 512
  → UMT5-XXL encoder
  → 有效文本 hidden states [L_text, 4096]
  → Wan text MLP
  → context [512, 3072]（在 DiT 内补零）
```

选择 UMT5 的直接收益是长文本与中英文等多语言覆盖。代价是它本身很大，因此官方入口提供 `--t5_cpu` 与 `--t5_fsdp`。

### 3.1 cross-attention 如何读取文本

对每个视频 token $x_i$，cross-attention 计算：

$$
Q=W_Qx,\qquad K=W_Kc,\qquad V=W_Vc,
$$

$$
\operatorname{CrossAttn}(x,c)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

其中视频 token 提供 query，文本 token 提供 key/value。因此每个时空位置都能按当前内容选择需要读取的词语。文本没有和视频 token 拼成同一条序列，也不使用视频的 3D RoPE。

### 3.2 positive 与 negative prompt

pipeline 会编码两段文本：

```text
context       = T5(positive prompt)
context_null  = T5(negative prompt)
```

这里的 `context_null` 名称容易误导：官方默认并不是空字符串，而是一段预设的负面描述。CFG 所谓 unconditional branch，在实际部署中经常是 negative-conditioned branch。

### 3.3 能否缓存文本 embedding

同一 prompt 的 50 个采样步使用相同 `context`，所以 T5 只运行一次。批量实验中还可以把 positive/negative embedding 持久化缓存；但 tokenizer、T5 checkpoint、最大长度和 dtype 必须保持一致。更改 prompt extension 结果也会使缓存失效。

## 第 4 步：Wan 2.2 高压缩因果 VAE

直接在 RGB 视频上运行 30 层全局 Transformer 几乎不可承受。VAE 学习两个映射：

$$
E: x_{\text{rgb}}\mapsto z,\qquad
D: z\mapsto \hat{x}_{\text{rgb}}.
$$

DiT 只学习 latent $z$ 的生成分布。VAE 决定了两项关键上限：

- 压缩越强，DiT token 越少、生成越快；
- 重建损失越大，DiT 再强也无法恢复 VAE 已丢掉的纹理或运动。

### 4.1 4×16×16 压缩是怎样得到的

5B 的 `vae2_2.py` 先做一次空间 `patchify(2)`，把：

```text
[3, T, H, W] → [12, T, H/2, W/2]
```

随后 encoder 再进行三次空间 2 倍下采样与两次时间 2 倍下采样，合计得到：

```text
时间 ×4，空间 ×(2×2×2×2)=×16
```

输出 latent channel 是 48。注意“压缩倍数”可能有三种说法：

| 口径 | 结果 |
|---|---|
| 每个轴 | $4\times16\times16$ |
| 相对 RGB 标量数量 | $(3THW)/(48\cdot T/4\cdot H/16\cdot W/16)=64$ |
| 加上 DiT 2×2 patchify 后的 token 网格 | $4\times32\times32$ 像素对应一个时空 token 位置 |

因此“总压缩率 64”与“4×16×16”并不矛盾：前者计入了 channel 从 3 增到 48。

### 4.2 “因果”究竟指什么

`CausalConv3d` 在时间轴只向过去 padding，不读取未来帧。因此 VAE 在时间位置 $t$ 的特征不会依赖 $t$ 之后的 RGB 帧。

```text
VAE encoder/decoder：时间因果
Video DiT self-attention：默认全局、非因果
```

这两个结论必须同时记住。Wan 并不是像自回归语言模型一样逐帧生成；DiT 在每个去噪步可以让所有 latent 时间位置互相注意。因果 VAE 的价值主要是首帧可独立编码、边界更稳定，以及通过 feature cache 分块编码/解码。

### 4.3 VAE 内部的 temporal 与 spatial 建模

VAE 主体由 causal 3D convolution、residual block、上下采样组成。中间的 attention block 会把时间并入 batch，执行逐帧的空间 attention；真正跨时间的信息主要由 causal 3D convolution 传递。这与 DiT 的全局时空 attention 又是不同层次。

### 4.4 为什么第一帧是特殊的

时间下采样以“第一帧 + 后续每四帧一组”的方式工作：

```text
RGB time:    1 + 4 + 4 + ... + 4
latent time: 1 + 1 + 1 + ... + 1
```

所以 $T_z=(T-1)/4+1$，也自然支持把一张输入图编码为只有一个时间位置的 clean latent。encoder 会分块喂入首帧和后续四帧组，decoder 逐 latent 时间片解码，并用 feature cache 维护因果状态。这是一种内存工程策略，不会改变最终的压缩公式。

### 4.5 latent normalization

VAE checkpoint 带有逐 channel 的均值与标准差。DiT 看到的是归一化后的 latent，而 decoder 前会做逆变换。训练 LoRA 或替换 VAE 时，不能只让形状相同；latent 的尺度分布也必须匹配，否则噪声日程和 velocity target 都会失真。

## 第 5 步：patchify 与 3D RoPE

VAE latent 仍是规则网格 $[C_z,T_z,H_z,W_z]$。一个 stride 等于 kernel 的 `Conv3d` 同时完成切块与线性投影：

$$
[48,31,44,80]
\xrightarrow{(1,2,2)}
[3072,31,22,40].
$$

flatten 的顺序保留 `(time, height, width)` 网格元数据；否则 Transformer 只会看到 27,280 个无序向量。

### 5.1 3D RoPE 如何编码位置

Wan 对 self-attention 的 $Q,K$ 应用三维 rotary positional embedding。每个 head 是 128 维，即 64 个二维旋转对。源码将它们分成：

```text
时间轴：22 个复数维度 = 44 个实数维度
高度轴：21 个复数维度 = 42 个实数维度
宽度轴：21 个复数维度 = 42 个实数维度
总计： 64 个复数维度 = 128 个实数维度
```

对某一轴位置 $p$ 和频率 $\omega_k$，一对坐标做旋转：

$$
\begin{bmatrix}x'_{2k}\\x'_{2k+1}\end{bmatrix}
=
\begin{bmatrix}
\cos(p\omega_k)&-\sin(p\omega_k)\\
\sin(p\omega_k)& \cos(p\omega_k)
\end{bmatrix}
\begin{bmatrix}x_{2k}\\x_{2k+1}\end{bmatrix}.
$$

attention 内积因而能感知相对时间差与相对空间位移。RoPE 只作用于 $Q,K$，不作用于 $V$；cross-attention 也不使用视频 3D RoPE。

### 5.2 为什么还有 QK normalization

源码先做：

```python
q = RMSNorm(Wq(x))
k = RMSNorm(Wk(x))
v = Wv(x)
```

高维、长序列 attention 的 logit 很容易尺度漂移。QK RMSNorm 约束 query/key 的均方尺度，使训练和混合精度推理更稳定，但不改变 attention 的 $O(L^2)$ 计算复杂度。

### 5.3 FlashAttention 做了什么、没做什么

官方 native 实现使用 FlashAttention 2/3 来避免显式保存完整 $L\times L$ attention matrix，从而将中间显存显著降低。它是 exact attention 的高效 kernel：

- 改善 IO 和显存；
- 不把全局 attention 变成线性复杂度；
- 不改变模型学到的连接图。

配置 `window_size=(-1,-1)` 表示全局 attention。27,280 token 的主要算力瓶颈依然存在。

## 第 6 步：拆开一个 Wan Transformer block

每个 5B block 的顺序是：

```text
x
├─ AdaLN → self-attention(3D RoPE) → gate → residual
├─ norm → cross-attention(text)              → residual
└─ AdaLN → Linear 3072→14336 → GELU → Linear → gate → residual
```

用近似公式表示：

$$
\tilde{x}=\operatorname{LN}(x)(1+s_1)+b_1,
$$

$$
x\leftarrow x+g_1\operatorname{SelfAttn}(\tilde{x}),
$$

$$
x\leftarrow x+\operatorname{CrossAttn}(\operatorname{LN}(x),c),
$$

$$
\hat{x}=\operatorname{LN}(x)(1+s_2)+b_2,
$$

$$
x\leftarrow x+g_2\operatorname{FFN}(\hat{x}).
$$

六个调制量 $(b_1,s_1,g_1,b_2,s_2,g_2)$ 来自时间 embedding 与该 block 自己的 learned modulation 之和。cross-attention 没有时间 gate，但它接收的是已经被前一残差更新过的 $x$。

### 6.1 self-attention 同时承担空间一致性与时间一致性

Wan 没有显式拆成“空间 block + 时间 block”。同一 attention 序列包含所有 $(t,h,w)$ token，因此一个 token 理论上可直接访问：

- 同一帧的远处对象；
- 相邻帧的对应物体；
- 很久以前的场景状态。

这种统一设计表达力强，但代价昂贵。它也解释了为什么 3D 位置编码与 sequence parallel 是核心，而不是外围优化。

### 6.2 输出 head 与 unpatchify

30 个 block 后，head 再由当前时间 embedding 调制，然后把每个 3072 维 token 投影为：

```text
48 latent channels × 1×2×2 patch = 192 values
```

`unpatchify` 将 `[27280,192]` 重排回 `[48,31,44,80]`。head 预测的不是 RGB，而是与输入 latent 同形状的 flow field。

从头训练时，输出线性层被 zero-initialize，使网络初始输出接近零；这有助于让深残差网络从稳定状态开始。加载 checkpoint 后当然使用已训练权重。

## 第 7 步：时间步为何能按 token 变化

标准扩散模型通常给整张样本一个标量时间步 $t$。Wan 的 forward 还接受 `[B,L]` 的 token-wise timestep：

```text
t [B]   → 扩展为 [B,L]
或
t [B,L] → 保持每个 token 自己的噪声等级
```

每个 $t_i$ 经 256 维 sinusoidal embedding、两层 MLP，再投影为 $6\times3072$。5B T2V 中所有有效 token 使用同一个 $t$；5B I2V 中首帧 token 使用 $t=0$，未来 token 使用当前 scheduler timestep。

这让同一 DiT 能理解一个“混合噪声状态”的样本：

```text
首帧 token：已经干净，不应被改写
未来 token：仍在当前噪声等级，需要继续生成
```

padding token 也会补齐到 sequence-parallel 所需长度；`seq_lens` 保证 attention 不把 padding 当真实视频内容。

## 第 8 步：Flow Matching、shift、solver 与 CFG

这是整条链中最需要区分概念的一步：

```text
Flow Matching：规定训练目标——模型学习什么向量场
noise schedule：规定经过哪些噪声等级
solver：规定如何数值积分这个向量场
CFG：组合两次条件不同的向量场预测
```

### 8.1 一条最简单的概率路径

为了固定符号，本节令：

- $z_0$：干净视频 latent；
- $\epsilon\sim\mathcal N(0,I)$：高斯噪声；
- $\sigma=0$：数据端；$\sigma=1$：噪声端。

使用直线路径：

$$
z_\sigma=(1-\sigma)z_0+\sigma\epsilon.
$$

沿这条路径的目标速度为：

$$
u_\sigma=\frac{d z_\sigma}{d\sigma}=\epsilon-z_0.
$$

训练一个条件向量场 $v_\theta(z_\sigma,\sigma,c)$，最基础的 conditional flow matching 损失是：

$$
\mathcal L_{\text{FM}}
=\mathbb E\left[
w(\sigma)\left\|v_\theta(z_\sigma,\sigma,c)-(\epsilon-z_0)\right\|_2^2
\right].
$$

生成时从 $z_1\sim\mathcal N(0,I)$ 出发，把 ODE 从 $\sigma=1$ 积分回 $0$：

$$
\frac{dz}{d\sigma}=v_\theta(z,\sigma,c).
$$

不同论文可能用 $t=0$ 表示噪声端，符号会整体反过来；判断代码时应看 scheduler 的方向，而不是死记变量名。Wan 的采样 timestep 从高到低，源码变量虽叫 `noise_pred`，scheduler 明确声明 `prediction_type='flow_prediction'`。

> 官方仓库公开的是推理代码，不包含基础模型的完整预训练 dataloader 与 loss 实现。因此上式是与其 flow-prediction 接口一致的标准解释，不应被误读为官方未公开训练配方的逐行复现。

### 8.2 `shift` 如何重排噪声等级

Wan 对基础 $\sigma$ 使用：

$$
\sigma'=\frac{s\sigma}{1+(s-1)\sigma}.
$$

当 $s>1$ 时，内部采样点被推向较高噪声区域。例如 $s=5$ 时，原本 $\sigma=0.5$ 变为 $0.833$。这不是把步数变多，而是在相同步数下重新分配对不同噪声区间的数值计算预算。

`shift` 与分辨率、模型训练分布相关，不是越大越好。5B 默认 5；A14B T2V 默认 12；A14B I2V 默认 5。跨模型照搬参数通常不合理。

### 8.3 UniPC 和 DPM++ 是积分器

采样循环可抽象为：

```python
z = randn(shape)
for sigma in schedule:
    v = guided_model(z, sigma, prompt)
    z = solver.step(v, sigma, z)
```

Euler 只使用当前斜率；UniPC 与 DPM++ multistep 会利用历史模型输出构造更高阶更新，以较少 neural function evaluations 获得更小积分误差。它们不增加模型知识，也不能修复 prompt、VAE 或 checkpoint 不匹配。

### 8.4 CFG 的几何意义与代价

$$
v_{\text{cfg}}=v_u+w(v_c-v_u).
$$

$v_c-v_u$ 是“加入 prompt 条件后，向量场改变的方向”。$w>1$ 沿该方向外推，提高文本对齐，过大则可能导致饱和、运动僵硬或细节破坏。

原始实现每步分别跑 conditional/unconditional 两次 DiT。因此近似 neural function evaluations 为：

$$
NFE\approx2\times\text{sampling steps}.
$$

50 steps 即约 100 次 5B forward。减少 step 与去掉/蒸馏 CFG 是两条不同的加速路线。

## 第 9 步：5B I2V 如何真正“固定首帧”

TI2V-5B 的 I2V 不是额外拼接图像特征，而是在 latent 本身上做 masked generation。

### 9.1 首帧预处理

输入图像先：

1. 按最大面积等比缩放；
2. center crop 到 32 对齐的尺寸；
3. 从 $[0,1]$ 归一化到 $[-1,1]$；
4. 增加长度为 1 的时间轴；
5. 经 Wan 2.2 VAE 编成 `[48,1,H_z,W_z]` 的 clean latent $z_{\text{ref}}$。

### 9.2 mask、初态与每步回钉

定义与目标 latent 同形状、可广播的 mask $m$：

$$
m_t=
\begin{cases}
0,&t=0,\\
1,&t>0.
\end{cases}
$$

初始 latent 是：

$$
z=(1-m)z_{\text{ref}}+m\epsilon.
$$

因此首个 latent 时间片来自参考图，其余位置是噪声。每次 solver 更新后，源码再次执行：

$$
z\leftarrow(1-m)z_{\text{ref}}+mz,
$$

把首帧重新钉回 clean condition，防止数值更新漂移。

### 9.3 token-wise timestep 是第二道约束

mask 以 `[:, ::2, ::2]` 对齐 DiT 的 2×2 patch 网格，构造：

$$
t_i=
\begin{cases}
0,&\text{首帧 token},\\
t,&\text{未来 token}.
\end{cases}
$$

所以模型不仅看见 clean 首帧，还被显式告知这些 token 处于零噪声。**clean latent、每步回钉、token-wise $t=0$** 三者共同构成 5B I2V 条件机制。

### 9.4 T2V 是它的退化特例

T2V 使用全 1 mask，所有 token 都从噪声开始、都使用相同 $t$，没有 clean 区域需要回钉。这个统一表述也提供了扩展思路：若训练分布支持，首帧 mask 可推广到首尾帧、局部时空 inpainting 或已知片段续写。

## 第 10 步：A14B 双专家到底是什么

Wan 2.2 A14B 使用两个约 14B 的完整 DiT checkpoint：

```text
高噪声 expert：采样早期，决定全局布局、镜头与大尺度运动
低噪声 expert：采样后期，修正纹理、边缘与细节
```

总参数约 27B，但每个采样步只激活一个约 14B expert，所以名称是 **A14B = Active 14B**。

### 10.1 它不是 token-routed sparse MoE

没有 router 为每个 token 选择 FFN expert。源码做的是一个确定性时间路由：

```python
boundary = config.boundary * 1000

if timestep >= boundary:
    model = high_noise_model
else:
    model = low_noise_model
```

T2V 的 `boundary=0.875`，I2V 的 `boundary=0.900`。这里比较的是 shift 后 scheduler 的 timestep；不能简单理解为“40 步中的 87.5% 都用某个专家”。实际各专家步数取决于非线性 schedule。

每个阶段还可以使用不同 CFG：

```text
T2V-A14B：low 3.0，high 4.0
I2V-A14B：low 3.5，high 3.5
```

### 10.2 5B 与 A14B 结构对照

| 项目 | TI2V-5B | T2V/I2V-A14B |
|---|---:|---:|
| DiT expert | 1 个 dense 5B | 2 个完整 expert，总约 27B，每步 active 14B |
| hidden dim | 3072 | 5120 |
| FFN dim | 14336 | 13824 |
| heads / layers | 24 / 30 | 40 / 40 |
| head dim | 128 | 128 |
| VAE | Wan 2.2 VAE，48 channels | Wan 2.1 VAE，16 channels |
| VAE stride | `(4,16,16)` | `(4,8,8)` |
| DiT patch | `(1,2,2)` | `(1,2,2)` |
| 默认帧数 / FPS | 121 / 24 | 81 / 16 |
| 默认 steps | 50 | 40 |
| I2V 条件 | masked clean latent + token-wise timestep | mask + encoded reference 拼接到 noisy latent channel |

### 10.3 A14B 的 token 数为何反而更大

以 81 帧、720×1280 为例：

```text
RGB              [3, 81, 720, 1280]
Wan 2.1 latent   [16, 21, 90, 160]
patch grid       [21, 45, 80]
video tokens     21×45×80 = 75,600
hidden width     5,120
```

它比 5B 示例的 27,280 token 多约 2.77 倍，hidden 也更宽。这正是高压缩 VAE 对 5B 效率如此关键的原因。

### 10.4 A14B I2V 使用另一种条件接口

A14B I2V 会构造一个“首帧 RGB + 未来零帧”的视频，送入 Wan 2.1 VAE 得到 16-channel reference latent，再与一个 4-channel temporal mask 拼成 20-channel 条件 $y$。DiT forward 又把 noisy latent $x$ 的 16 channels 与 $y$ 拼接：

```text
noisy latent x       16 channels
condition y          4-channel mask + 16-channel encoded reference
patch input          36 channels
model output         16-channel flow
```

它不采用 5B 的逐步首帧回钉。由此得到一个重要工程结论：**“都是 Wan2.2 I2V”不等于 condition tensor 可以互换**。LoRA、Control 模块和 checkpoint 转换必须先确认是 5B TI2V 还是 A14B I2V。

### 10.5 两个专家如何占用显存

“每步 active 14B”描述计算量，不自动保证只存一份权重。BF16 理论裸权重估算：

```text
5B：约 10 GB
14B expert：约 28 GB
两个 expert：约 54 GB（总参数约 27B）
```

这还不含 T5、VAE、activation、CUDA workspace 与 allocator 碎片。CPU offload 可以只把当前 expert 搬到 GPU；FSDP 则把参数分片到多卡。实际峰值必须以目标软件栈实测，不能拿裸权重估算当显存需求承诺。

## 第 11 步：从推理反推训练与 LoRA

先画出一个与公开推理接口一致的基础训练循环：

```python
video, caption = sample_batch()
z0 = vae.encode(video)                 # 通常冻结并可缓存
context = t5(caption)                   # 通常冻结并可缓存

sigma = sample_noise_level()
eps = randn_like(z0)
z_sigma = (1 - sigma) * z0 + sigma * eps
target = eps - z0

pred = dit(z_sigma, sigma, context)
loss = weighted_mse(pred, target, sigma)
loss.backward()
optimizer.step()
```

真实大模型训练还会包含视频清洗与 captioning、图像/视频混训、宽高比分桶、时长分桶、噪声采样权重、条件 dropout、分布式并行和精度策略。Wan 2.2 官方仓库没有公开这些基础训练实现与完整超参数，所以不能仅靠 `generate.py` 复刻预训练。

### 11.1 CFG 为什么要求 condition dropout

要在同一个模型中得到 $v_c$ 与 $v_u$，训练时需要让一部分样本丢弃或替换文本条件，使网络同时学会 conditional 与 unconditional/negative-conditioned prediction。推理时才可以用两者差值做 CFG。若微调数据永远保留强 caption，过拟合后 unconditional branch 可能退化。

### 11.2 I2V 训练还需构造混合噪声状态

对 5B I2V，训练样本应让参考首帧保持 clean、未来 latent 加噪，并提供相应的 token-wise timestep/mask。只拿 T2V 的“全 latent 同噪声”训练循环，无法教会模型可靠利用首帧约束。

对 A14B I2V，则要复现 20-channel condition 与 36-channel patch input 的协议。两条训练数据管线并不相同。

### 11.3 LoRA 应插在哪里

最常见的目标是 attention 线性层：

```text
self-attention: q, k, v, o
cross-attention: q, k, v, o
可选：FFN 的两层 Linear
```

对冻结权重 $W$，LoRA 学习低秩增量：

$$
W'=W+\frac{\alpha}{r}BA,
$$

其中 $A\in\mathbb R^{r\times d_{in}}$、$B\in\mathbb R^{d_{out}\times r}$，$r$ 远小于原矩阵维度。先只调 attention 通常更省显存；角色外观或强风格可能需要扩大到 FFN，但也更容易过拟合。

### 11.4 A14B LoRA 的专家归属

两个 expert 是两套参数。只给 low-noise expert 训练 LoRA，主要改变细节阶段；只给 high-noise expert 训练，主要改变构图与运动阶段。若工具把同一个 adapter 生硬挂到两边，要先确认参数名、shape 和训练 timestep 分布是否匹配。

一个稳妥实验顺序是：

1. 先在 TI2V-5B 上验证数据、caption、latent cache 与 loss 是否正确；
2. 用短视频/较低分辨率做过拟合测试，确认能记住一个小数据集；
3. 再扩大数据并评估 prompt 泛化、身份保持、运动与过拟合；
4. 最后才迁移到 A14B，并分别观察高/低噪声 expert 的梯度与 adapter 效果。

### 11.5 可缓存与不可缓存的量

| 可缓存 | 前提 |
|---|---|
| VAE latent | crop、帧采样、VAE 与 normalization 固定 |
| T5 embedding | caption、tokenizer、T5 与 prompt extension 固定 |
| negative embedding | negative prompt 与 T5 固定 |

随机噪声、$\sigma$、condition dropout 与增强通常应在线采样，否则会损失训练分布多样性。

官方 README 将 DiffSynth-Studio 列为支持 Wan 2.2 LoRA/full training 的社区栈；它能提供工程入口，但不会替代对上述 tensor contract 的理解。

## 第 12 步：多卡与显存工程

### 12.1 四类开关解决不同问题

| 机制 | 解决什么 | 主要代价 |
|---|---|---|
| `--t5_cpu` | T5 不占 GPU 常驻显存 | 文本编码较慢，但每个 prompt 通常只做一次 |
| `--offload_model True` | DiT/expert 不使用时移到 CPU | PCIe 传输与同步开销 |
| `--dit_fsdp`, `--t5_fsdp` | 将模型参数分片到多 GPU | 通信、启动与实现复杂度 |
| `--ulysses_size N` | 将长视频 attention 序列分摊到 N 个 rank | all-to-all 通信；head 数需适配并行规模 |

FSDP 是**参数并行**；Ulysses 是**序列/attention 并行**。一个模型可能参数放得下，却被 75,600-token activation 卡住，此时只有 FSDP 不一定够。

### 12.2 Ulysses 的核心数据交换

源码先沿 sequence 维把 $x$、时间调制切给不同 rank；计算 QKV 后用 all-to-all 在 sequence 与 head 维之间重排，让每个 rank 拥有部分 heads、但能对完整 sequence 做 attention；输出后再聚合序列。

这不减少全局总 FLOPs，却降低单卡持有的序列 activation。`seq_len` 会向并行规模取整，3D RoPE 频率也按 rank 切片，最后 `gather_forward` 恢复完整 token 序列再 unpatchify。

### 12.3 dtype 不是一键无损压缩

`--convert_model_dtype` 将 DiT 参数转为配置中的 BF16，activation 也在 autocast 下计算；部分 normalization、时间 embedding 与调制明确回到 FP32，以避免数值不稳定。量化到 FP8/INT8 是另一个问题，需要专用 kernel、校准与质量验证，不能从这个 flag 推导出来。

## 第 13 步：一套可复现的源码阅读实验

不必先下载完整权重，也能验证大量结构结论。

### 13.1 静态检查清单

```bash
rg "vae_stride|patch_size|dim|ffn_dim|num_heads|num_layers" wan/configs
rg "patch_embedding|time_projection|WanAttentionBlock" wan/modules/model.py
rg "mask2|temp_ts|latent = \(1\. - mask" wan/textimage2video.py
rg "high_noise_model|low_noise_model|boundary" wan/text2video.py wan/image2video.py
```

目标不是记住行号，而是能够从配置独立算出 latent shape、token count、head dim 和 I2V input channels。

### 13.2 在关键边界打印 shape

若有可运行环境，可用 forward hook 或临时断点只观察以下位置：

```text
T5 输出
VAE encode 输出
patch_embedding 输出
第一个 block 的输入/输出
head 输出
VAE decode 输出
```

不要在 attention 内打印完整 tensor；27,280-token 的日志本身就会拖垮运行。只输出 `shape`, `dtype`, `device`, `min/max/mean/std`。

### 13.3 三个最有信息量的消融

固定 prompt 与 seed，每次只改一个变量：

1. CFG：`3 / 5 / 7`，观察语义贴合、饱和与运动；
2. steps：`20 / 30 / 50`，观察 solver 离散误差与耗时；
3. shift：围绕模型默认值小范围变化，观察构图与细节阶段的权衡。

比较时必须固定 checkpoint、solver、尺寸、帧数、negative prompt 与 prompt extension。否则无法把差异归因到单个变量。

### 13.4 常见故障的定位顺序

| 现象 | 优先检查 |
|---|---|
| shape mismatch at patch embed | 5B/A14B checkpoint 与 condition channel 是否混用 |
| 输出尺寸异常 | `(width,height)` 顺序、总对齐倍数、I2V 最大面积逻辑 |
| 首帧漂移 | 5B I2V mask、每步回钉、token-wise timestep |
| 画面全坏或色彩漂移 | VAE 版本、latent mean/std、dtype |
| 文本不跟随 | positive/negative context、CFG、prompt extension、T5 checkpoint |
| 多卡 hang | world size、Ulysses size、NCCL 初始化与各 rank 控制流是否一致 |
| OOM | 先区分参数、attention activation、VAE decode 哪一阶段峰值最高 |

## 一张总图：Wan 2.2 的完整心智模型

```text
                           ┌─────────────────────────────┐
prompt ──UMT5-XXL────────▶│ text context [≤512,4096]   │
                           └──────────────┬──────────────┘
                                          │ cross-attention
RGB/reference image                       │
   │                                      │
   ├─ T2V: 不需要编码                     │
   └─ I2V: causal VAE encode              │
             │                            │
             ▼                            ▼
noise / masked clean latent ──▶ patchify ──▶ Video DiT
                                      3D RoPE self-attention
                                      text cross-attention
                                      timestep AdaLN/gates
                                              │
                         CFG: cond/uncond ─────┤ flow field
                                              ▼
                                     UniPC / DPM++
                                     sigma: 1 → 0
                                              │
                                   [5B I2V 每步回钉首帧]
                                              ▼
                                      clean video latent
                                              │
                                      causal VAE decode
                                              ▼
                                           RGB video

A14B 额外规则：高 sigma 用 high-noise DiT，低 sigma 用 low-noise DiT。
```

## 最终掌握检查

如果可以不看本文回答下面问题，就已经从“会调用 Wan”进入了“理解 Wan”：

1. 为什么 121 帧变成 31 个 latent 时间位置？
2. 为什么 5B 的 720p 档位用 704×1280，而 A14B 可用 720×1280？
3. VAE compression 与 DiT patchify 的职责有何不同？
4. 27,280 个 token 是如何算出的？
5. 3D RoPE 分别给时间、高度、宽度多少 head dimension？
6. self-attention、cross-attention 与 timestep modulation 各注入什么信息？
7. Flow Matching target、solver、noise schedule 与 CFG 为什么不是同一个概念？
8. 5B I2V 如何用 clean latent、mask、回钉和 token-wise timestep 固定首帧？
9. A14B 的 MoE 为什么不是 token routing？
10. 为什么 A14B I2V adapter 不能直接当作 5B TI2V adapter？
11. FSDP 与 Ulysses 分别缓解哪种显存压力？
12. 哪些训练细节能由公开源码确认，哪些不能？

## 术语速查

| 术语 | 在本章中的准确含义 |
|---|---|
| latent | VAE 压缩、归一化后供 DiT 生成的连续视频表示 |
| Video DiT | 在带噪视频 latent token 上预测 flow field 的 Transformer |
| Flow Matching | 用条件向量场连接噪声分布与数据分布的训练框架 |
| scheduler | 构造离散 timestep / sigma 序列及相关转换的组件 |
| solver | 根据模型向量场预测执行 ODE 数值更新的方法 |
| CFG | conditional 与 unconditional/negative prediction 的线性外推 |
| causal VAE | VAE 在时间上不读取未来帧；不代表 DiT 是自回归的 |
| 3D RoPE | 在 self-attention Q/K 上编码时间、高度、宽度位置的旋转位置编码 |
| A14B | 总参数约 27B、每一步只激活约 14B expert |
| offload | 在 CPU/GPU 间迁移不活跃模块以换取较低 GPU 常驻显存 |
| FSDP | 多 GPU 参数分片 |
| Ulysses | 通过 all-to-all 重排 sequence/head 的 attention 并行 |

## 资料与下一步

- [Wan2.2 官方仓库与运行命令](https://github.com/Wan-Video/Wan2.2)
- [TI2V-5B 配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)
- [5B T2V/I2V pipeline](https://github.com/Wan-Video/Wan2.2/blob/main/wan/textimage2video.py)
- [Video DiT 与 3D RoPE](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)
- [Wan 2.2 高压缩 VAE](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py)
- [A14B T2V 专家切换](https://github.com/Wan-Video/Wan2.2/blob/main/wan/text2video.py)
- [A14B I2V 条件构造](https://github.com/Wan-Video/Wan2.2/blob/main/wan/image2video.py)
- [Wan 2.1 技术报告：基础架构、训练与评估背景](https://arxiv.org/abs/2503.20314)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

读完本章后，最合适的下一个实践不是马上全量训练，而是实现一个最小 shape tracer：从配置自动计算 RGB、VAE latent、patch grid、token count、attention head dimension 和理论 $L^2$，再用一次低分辨率/短帧推理核对。能稳定解释这些边界后，再进入 LoRA 数据管线与训练 loss，调试成本会低一个数量级。
