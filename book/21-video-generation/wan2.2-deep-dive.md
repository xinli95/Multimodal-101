# Wan 2.2 Deep Dive: A Source-Guided Dissection

This section uses **Wan 2.2-TI2V-5B** to dissect a modern video generator through a real inference path. Instead of front-loading every equation, we move through the entry point, configuration, text encoding, video latents, Video DiT, Flow Matching, and VAE decoding, introducing theory exactly when the implementation needs it.

The subject here is Wan 2.2, whose code and weights are public, rather than the newer Wan 2.6/2.7 API models whose weights are not open. The 5B model is a starting point rather than the destination: it exposes a single-DiT pipeline clearly before we move to the A14B high-noise/low-noise expert design.

> **Status, checked 2026-08-26:** Wan's official GitHub organization still exposes Wan 2.2 as its newest open base video-generation repository, under Apache-2.0. A newer product/API version number does not imply that matching open weights exist. “Newest online Wan” and “newest official open-weight Wan” are separate questions.

### Wan 2.2 family at a glance

| Official checkpoint | Input → output | Role in this chapter |
|---|---|---|
| TI2V-5B | text, or text + first frame → 720p at 24 FPS | Main path: unified T2V/I2V, high-compression VAE, dense DiT |
| T2V-A14B | text → 480p/720p video | Step 10: high-/low-noise experts |
| I2V-A14B | text + first frame → 480p/720p video | Step 10: a separate channel-concatenation I2V contract |
| S2V-14B | text + reference image + speech/audio → speaking or singing video | Adds audio driving to the shared foundation; a later specialized topic |
| Animate-14B | character assets + motion/pose controls → animation or replacement | Requires specialized preprocessing and controls; a later topic |

The self-contained scope here is Wan 2.2's shared generative foundation and its core T2V/I2V paths. S2V and Animate reuse much of that foundation, but their audio, pose, mask, and character preprocessing merit dedicated chapters.

## Reading path

| Step | Question | Main source |
|---|---|---|
| 1 | How do data and tensors flow through one generation? | `wan_ti2v_5B.py`, `textimage2video.py` |
| 2 | How does the CLI choose a task, config, and pipeline? | `generate.py` |
| 3 | How does a prompt become a conditioning sequence? | `modules/t5.py` |
| 4 | Why is video compressed into a latent first? | `modules/vae2_2.py` |
| 5 | How does a latent video become spacetime tokens? | `modules/model.py` |
| 6 | What happens inside one Wan Transformer block? | `modules/model.py`, `modules/attention.py` |
| 7 | How does the timestep modulate every block? | `modules/model.py` |
| 8 | How do Flow Matching, the solver, and CFG form the sampling loop? | `utils/fm_solvers*.py` |
| 9 | How does I2V pin the first frame while generating the rest? | `textimage2video.py` |
| 10 | How does A14B switch experts across noise regimes? | `text2video.py`, `image2video.py` |
| 11 | How does inference understanding transfer to LoRA and training? | inference contracts and community trainers |
| 12 | What do FSDP, Ulysses, and offload each solve? | `distributed/`, pipeline initialization |
| 13 | How can shape tracing, ablations, and a fault table test understanding? | configuration and module boundaries |

Official source: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2). Keep these files open in order:

1. [`generate.py`](https://github.com/Wan-Video/Wan2.2/blob/main/generate.py): CLI entry point.
2. [`wan/configs/wan_ti2v_5B.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py): 5B configuration.
3. [`wan/textimage2video.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/textimage2video.py): T2V/I2V pipeline and sampling loop.
4. [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py): Video DiT backbone.
5. [`wan/modules/vae2_2.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py): video VAE.
6. [`wan/utils/fm_solvers_unipc.py`](https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/fm_solvers_unipc.py): Flow Matching solver.

## Step 1: End-to-end tensor flow

First compress a roughly five-second T2V generation into one map:

```text
text prompt
   │
   ▼
UMT5-XXL Text Encoder
   │ context: [text length, 4096]
   ▼
text projection 4096 → 3072 ──────────────┐
                                          │
Gaussian video latent                     │
[48, 31, 44, 80]                          │
   │                                      │
   ▼                                      │
3D Patch Embedding                        │
Conv3d(kernel=1×2×2, stride=1×2×2)        │
   │                                      │
   ▼                                      │
27,280 video tokens, 3,072 dimensions     │
   │                                      │
   ▼                                      │
30 × Wan Transformer Block ◀──────────────┘
   │
   ▼
predict flow / velocity
   │
   ▼
UniPC or DPM++ updates the latent for about 50 steps
   │
   ▼
clean video latent [48, 31, 44, 80]
   │
   ▼
Wan 2.2 Video VAE Decoder
   │
   ▼
RGB video [3, 121, 704, 1280]
```

### 1.1 What the configuration already tells us

The important fields in `wan_ti2v_5B.py` are:

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

There are two very different layers of compression:

```text
RGB video
  └─ Video VAE compression: (4, 16, 16)
       └─ DiT patchify: (1, 2, 2)
```

The VAE moves pixels into a smaller continuous latent space. Patchification does not learn another semantic compression; it arranges the latent grid into Transformer tokens.

### 1.2 Why 121 frames

Wan 2.2's causal video VAE expects:

$$
F = 4n + 1.
$$

The latent time length is:

$$
F_z = \frac{F-1}{4}+1.
$$

For $F=121$:

$$
F_z = \frac{121-1}{4}+1=31.
$$

The 121 RGB frames therefore become 31 latent time positions. This is not simply $121/4$: the causal VAE handles the first frame independently, then processes subsequent content in groups of four frames. At 24 FPS, 121 frames play for about 5.04 seconds.

### 1.3 Why "720p" is represented as 704×1280

Spatial dimensions must satisfy both the VAE's 16× compression and the DiT's 2× patchification:

```text
total alignment = 16 × 2 = 32
```

1280 is divisible by 32 while 720 is not; nearby 704 is:

```text
1280 / 32 = 40
704  / 32 = 22
```

Here, "720p" names a resolution tier, while the actual tensor uses 704×1280 to satisfy model alignment.

### 1.4 From RGB video to VAE latent

The new 5B VAE uses 48 latent channels. Starting from:

```text
[C, T, H, W] = [3, 121, 704, 1280]
```

4× temporal and 16× spatial compression produce:

```text
C_z = 48
T_z = (121 - 1) / 4 + 1 = 31
H_z = 704 / 16 = 44
W_z = 1280 / 16 = 80

latent = [48, 31, 44, 80]
```

T2V has no input video to encode, so inference starts from Gaussian noise with the same shape:

```python
noise = torch.randn(48, 31, 44, 80)
```

I2V first encodes its initial image into this grid. Step 9 will show how a mask re-pins that conditioning region after every solver update.

### 1.5 From latent grid to 27,280 tokens

`WanModel` uses a 3D convolution for patch embedding:

```python
self.patch_embedding = nn.Conv3d(
    in_channels=48,
    out_channels=3072,
    kernel_size=(1, 2, 2),
    stride=(1, 2, 2),
)
```

It preserves latent time and groups each spatial $2\times2$ neighborhood into one token:

```text
latent input       [48,   31, 44, 80]
Conv3d output      [3072, 31, 22, 40]
flatten + transpose
Transformer input  [31 × 22 × 40, 3072]
                  = [27280, 3072]
```

A five-second clip already contains 27,280 video tokens. Global self-attention scales quadratically:

$$
L^2=27280^2\approx 7.44\times10^8.
$$

That is the number of possible position pairs **per attention head**. FlashAttention avoids materializing the full matrix in memory, but it does not remove the dominant compute. This is why video DiTs quickly require sequence parallelism, sparse attention, distillation, or higher VAE compression.

### 1.6 Aligning text with the DiT hidden width

Wan uses a UMT5-XXL encoder with maximum length 512 and output width 4096:

```text
prompt → UMT5 → context [L_text, 4096], L_text ≤ 512
```

Inside `WanModel`, a two-layer MLP applies:

```text
Linear(4096, 3072) → GELU → Linear(3072, 3072)
```

Video and text tokens are not concatenated. Every Wan block performs:

1. self-attention among video tokens;
2. cross-attention from video queries to text context;
3. a token-wise FFN.

A useful first approximation is that self-attention maintains spatial and temporal relationships while cross-attention injects prompt semantics. Training entangles these roles, so they are not a strict separation.

### 1.7 The timestep is not an ordinary token

At every sampling step, the model must know the current noise level. Timestep $t$ becomes a 256-dimensional sinusoidal embedding and is then projected to model width:

```text
t → sinusoidal embedding 256
  → MLP 3072
  → projection 6 × 3072
```

Each Transformer block uses the six groups as shift, scale, and gate values for self-attention and the FFN. The core form is approximately:

$$
y=\operatorname{Norm}(x)(1+\text{scale})+\text{shift},
$$

$$
x\leftarrow x+\text{gate}\cdot\operatorname{Module}(y).
$$

This is AdaLN/AdaLN-Zero-style conditioning: the same block can behave differently at different noise regimes.

### 1.8 One DiT forward pass

Ignoring padding, parallelism, and mixed precision, the backbone is:

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

The 5B configuration has 30 blocks and 24 attention heads, giving a head dimension of $3072/24=128$.

### 1.9 Why CFG runs the DiT twice per sampling step

Classifier-Free Guidance computes both positive and negative/unconditional predictions:

```python
pred_cond = model(latent, t, prompt)
pred_uncond = model(latent, t, negative_prompt)

prediction = pred_uncond + guidance_scale * (
    pred_cond - pred_uncond
)
```

Equivalently:

$$
v=v_{\text{uncond}}+s\left(v_{\text{cond}}-v_{\text{uncond}}\right).
$$

The default is $s=5.0$. The source names this value `noise_pred`, but the solver's `prediction_type` is `flow_prediction`; semantically, it is closer to a Flow Matching velocity field than necessarily the pure noise $\epsilon$ of a traditional DDPM.

CFG's cost is immediate: 50 sampling steps usually require about 100 large DiT forwards. This is why CFG and guidance distillation can produce major speedups.

### 1.10 Step-one checkpoint

Before continuing, you should be able to reconstruct these four shapes without the source:

```text
RGB video     [3, 121, 704, 1280]
VAE latent    [48, 31, 44, 80]
Video tokens  [27280, 3072]
Text context  [≤512, 4096] → [512, 3072]
```

And the minimal sampling loop:

```python
latent = random_noise()

for t in timesteps:
    cond = model(latent, t, prompt)
    uncond = model(latent, t, negative_prompt)
    velocity = uncond + cfg * (cond - uncond)
    latent = scheduler.step(velocity, t, latent)

video = vae.decode(latent)
```

Step 2 starts at `generate.py` with `--task ti2v-5B`, tracing how CLI arguments select `WAN_CONFIGS`, instantiate `WanTI2V`, and reach this sampling loop.

## Step 2: From the CLI to the pipeline

First prepare the native inference environment (Python 3.10+, PyTorch 2.4 or newer):

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt

pip install 'huggingface_hub[cli]'
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir ./Wan2.2-TI2V-5B
```

`requirements.txt` includes FlashAttention. If installing everything together fails, the official guidance is to install the other dependencies first and then install a `flash_attn` build compatible with the active CUDA/PyTorch environment. Check disk space before downloading the large checkpoint. All commands below assume the cloned official repository is the working directory.

Start with the smallest official 5B T2V invocation:

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

It passes through four dispatch layers:

```text
CLI arguments
  → _validate_args: validate task, size, frames, checkpoint, parallelism
  → WAN_CONFIGS['ti2v-5B']: fill steps / shift / CFG / FPS defaults
  → WanTI2V(config, checkpoint_dir, ...): assemble T5, VAE, and DiT
  → WanTI2V.generate(..., img=None): select t2v()
```

Adding `--image input.jpg` makes the final call select `i2v()`. Thus **TI2V** means that one 5B checkpoint and one DiT interface implement both T2V and I2V; it is not a wrapper around two independent models.

### 2.1 Keep three kinds of parameters separate

| Kind | Examples | Consequence of changing it |
|---|---|---|
| Architecture | `dim=3072`, `num_layers=30`, `patch_size=(1,2,2)` | Must match the checkpoint |
| Sampling | `steps=50`, `shift=5`, `guide_scale=5` | Changes speed and the numerical trajectory, not the weights |
| Output | `frame_num=121`, `size=1280*704`, `seed` | Changes shape or initial noise, and therefore compute/memory |

`frame_num` must be $4n+1$. The 5B model supports `1280*704` and `704*1280`. For I2V, `size` is effectively a maximum-area tier: `best_output_size` preserves the input aspect ratio while finding dimensions aligned to 32.

### 2.2 What initialization loads

`WanTI2V.__init__` assembles:

1. the UMT5-XXL tokenizer and encoder;
2. the 48-channel high-compression VAE from `Wan2.2_VAE.pth`;
3. one dense 5B `WanModel` Video DiT;
4. optional FSDP, Ulysses sequence parallelism, CPU offload, and dtype conversion.

There is no CLIP image encoder in this pipeline. For 5B I2V, the video VAE turns the reference image into a clean video latent; a mask and token-wise timesteps inject that condition.

### 2.3 Prompt extension is preprocessing

`--use_prompt_extend` can ask a local Qwen model or DashScope to expand a short prompt into a fuller shot description. It is not part of the DiT and does not run at every denoising step. Disable it while debugging the generator itself, or the text that reaches T5 may differ from the string you supplied.

### 2.4 Entry-point checkpoint

While reading `generate.py`, track five values:

```text
args.task       → pipeline class
cfg             → checkpoint architecture and sampling defaults
args.image      → T2V versus I2V for TI2V-5B
args.size       → exact T2V size or I2V maximum-area tier
args.base_seed  → explicit source of the initial random latent
```

## Step 3: UMT5 turns a prompt into conditioning

Wan 2.2 uses an **UMT5-XXL encoder**, not the CLIP text encoder familiar from many image models. UMT5 is multilingual; Wan uses only its encoder to obtain contextual representations for every text token.

```text
prompt
  → SentencePiece tokenizer
  → token ids + mask, maximum length 512
  → UMT5-XXL encoder
  → valid hidden states [L_text, 4096]
  → Wan text MLP
  → context [512, 3072] after padding inside the DiT
```

The tradeoff for long, multilingual conditioning is a large text model, hence `--t5_cpu` and `--t5_fsdp`.

### 3.1 How cross-attention reads text

For video tokens $x$ and text context $c$:

$$
Q=W_Qx,\qquad K=W_Kc,\qquad V=W_Vc,
$$

$$
\operatorname{CrossAttn}(x,c)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

Video supplies queries; text supplies keys and values. Every spacetime location can therefore select the words it needs. Text and video are not concatenated into one sequence, and text does not receive the video's 3D RoPE.

### 3.2 Positive and negative context

The pipeline encodes two strings:

```text
context       = T5(positive prompt)
context_null  = T5(negative prompt)
```

`context_null` is a slightly misleading name: by default it is a configured negative prompt, not necessarily an empty string. The “unconditional” side of deployed CFG is often negative-conditioned.

T5 runs only once per prompt, not once per denoising step. Embeddings can also be cached across jobs if the prompt, tokenizer, T5 checkpoint, maximum length, dtype, and prompt-extension result remain fixed.

## Step 4: The Wan 2.2 high-compression causal VAE

Running a 30-layer global Transformer directly on RGB video is prohibitive. The VAE learns:

$$
E:x_{\text{rgb}}\mapsto z,\qquad D:z\mapsto\hat{x}_{\text{rgb}}.
$$

The DiT models only the latent distribution. Stronger compression reduces DiT tokens, but any detail discarded by the VAE becomes an upper bound that the DiT cannot recover.

### 4.1 Where 4×16×16 comes from

`vae2_2.py` first applies spatial `patchify(2)`:

```text
[3, T, H, W] → [12, T, H/2, W/2]
```

The encoder then performs three more 2× spatial downsamplings and two 2× temporal downsamplings. Total stride is therefore 4 in time and 16 in each spatial axis. The output has 48 channels.

Three compression descriptions are all useful:

| Convention | Result |
|---|---|
| Per-axis VAE stride | $4\times16\times16$ |
| Number of scalars versus RGB | $(3THW)/(48\cdot T/4\cdot H/16\cdot W/16)=64$ |
| Including the DiT's 2×2 patchification | one token position per $4\times32\times32$ RGB region |

“64× overall compression” and “4×16×16” are therefore compatible: the former accounts for channels growing from 3 to 48.

### 4.2 What causal means—and what it does not

`CausalConv3d` pads only into the past on the temporal axis. A VAE feature at time $t$ does not read RGB frames after $t$.

```text
VAE encoder/decoder: temporally causal
Video DiT attention: global and non-causal by default
```

Wan does not autoregressively generate one frame at a time. At each denoising step, every latent time position can attend to every other one. Causality here makes the first frame independently encodable and enables chunked encoding/decoding with feature caches.

The VAE's middle attention reshapes time into the batch dimension, so it is per-frame spatial attention. Cross-time interaction inside the VAE comes mainly from causal 3D convolutions; the DiT later performs global spacetime attention.

### 4.3 Why the first frame is special

Temporal processing follows:

```text
RGB time:    1 + 4 + 4 + ... + 4
latent time: 1 + 1 + 1 + ... + 1
```

Thus $T_z=(T-1)/4+1$. The encoder processes the first frame and later four-frame chunks; the decoder consumes latent slices while carrying causal feature caches. This controls memory without changing the shape formula.

The checkpoint also stores per-channel latent means and standard deviations. DiT input is normalized, and decoding applies the inverse transform. A replacement VAE must match both tensor shape and latent statistics.

## Step 5: Patchification and 3D RoPE

A stride-equals-kernel `Conv3d` turns the latent grid into tokens while projecting channels:

$$
[48,31,44,80]\xrightarrow{(1,2,2)}[3072,31,22,40].
$$

Flattening preserves `(time, height, width)` grid metadata. Without position information, the Transformer would see 27,280 unordered vectors.

### 5.1 How 3D RoPE allocates dimensions

Wan applies 3D rotary embeddings to self-attention $Q,K$. A head has 128 real dimensions, or 64 rotation pairs. The source divides them as:

```text
time:   22 complex = 44 real dimensions
height: 21 complex = 42 real dimensions
width:  21 complex = 42 real dimensions
total:  64 complex = 128 real dimensions
```

For axis coordinate $p$ and frequency $\omega_k$:

$$
\begin{bmatrix}x'_{2k}\\x'_{2k+1}\end{bmatrix}
=
\begin{bmatrix}
\cos(p\omega_k)&-\sin(p\omega_k)\\
\sin(p\omega_k)& \cos(p\omega_k)
\end{bmatrix}
\begin{bmatrix}x_{2k}\\x_{2k+1}\end{bmatrix}.
$$

Attention dot products can consequently express relative temporal and spatial offsets. RoPE modifies $Q,K$, not $V$, and the text cross-attention does not use video 3D RoPE.

### 5.2 QK normalization and FlashAttention

The source computes:

```python
q = RMSNorm(Wq(x))
k = RMSNorm(Wk(x))
v = Wv(x)
```

QK RMS normalization prevents attention-logit scale drift in long, mixed-precision sequences. The native implementation then uses FlashAttention 2/3. FlashAttention avoids materializing the full $L\times L$ matrix and improves IO and memory, but it remains exact global attention with quadratic leading compute. `window_size=(-1,-1)` confirms that it is not a local window.

## Step 6: One Wan Transformer block

Each 5B block follows:

```text
x
├─ AdaLN → self-attention(3D RoPE) → gate → residual
├─ norm → cross-attention(text)              → residual
└─ AdaLN → Linear 3072→14336 → GELU → Linear → gate → residual
```

Approximately:

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

The six values $(b_1,s_1,g_1,b_2,s_2,g_2)$ are the sum of timestep conditioning and a learned per-block modulation. Cross-attention has no timestep gate, but consumes the already updated residual stream.

One self-attention sequence contains all $(t,h,w)$ positions. Wan does not alternate explicit spatial-only and temporal-only blocks. This is expressive but expensive, which makes 3D positions and sequence parallelism fundamental rather than peripheral optimizations.

After 30 blocks, the modulated output head maps every 3,072-dimensional token to:

```text
48 latent channels × 1×2×2 patch = 192 values
```

`unpatchify` reconstructs `[48,31,44,80]`. The output is a latent flow field, not RGB. For training from scratch, the final linear layer is initialized to zero so the deep residual network begins near a zero prediction.

## Step 7: Why the timestep can vary by token

A conventional diffusion call assigns one scalar timestep to a sample. Wan's forward also accepts `[B,L]` token-wise timesteps:

```text
t [B]   → expanded to [B,L]
t [B,L] → each token keeps its own noise level
```

Each value passes through a 256-dimensional sinusoidal embedding, an MLP, then a projection to $6\times3072$. In 5B T2V, every valid token uses the same $t$. In 5B I2V, first-frame tokens use $t=0$ while future tokens use the current scheduler timestep.

The same DiT can therefore understand a mixed-noise sample: a clean first frame that must remain fixed and noisy future positions that still need generation. Padding is added to align sequence-parallel partitions; `seq_lens` prevents padded positions from becoming video content.

## Step 8: Flow Matching, shift, solver, and CFG

Keep four concepts distinct:

```text
Flow Matching → what vector field the model learns
noise schedule → which noise levels are visited
solver         → how that field is numerically integrated
CFG            → how two condition-dependent field predictions are combined
```

### 8.1 A straight probability path

Fix the convention:

- $z_0$ is a clean video latent;
- $\epsilon\sim\mathcal N(0,I)$ is noise;
- $\sigma=0$ is the data end and $\sigma=1$ the noise end.

On a straight path:

$$
z_\sigma=(1-\sigma)z_0+\sigma\epsilon,
\qquad
u_\sigma=\frac{dz_\sigma}{d\sigma}=\epsilon-z_0.
$$

A basic conditional Flow Matching objective is:

$$
\mathcal L_{\text{FM}}
=\mathbb E\left[
w(\sigma)\left\|v_\theta(z_\sigma,\sigma,c)-(\epsilon-z_0)\right\|_2^2
\right].
$$

Generation starts from $z_1\sim\mathcal N(0,I)$ and integrates:

$$
\frac{dz}{d\sigma}=v_\theta(z,\sigma,c)
$$

from $\sigma=1$ back to $0$. Papers sometimes reverse the time convention, so inspect scheduler direction rather than memorizing a variable name. Wan's timesteps descend; although the variable is named `noise_pred`, its scheduler declares `prediction_type='flow_prediction'`.

> The official repository releases inference code, not the full base-model pretraining dataloader and loss. The equation above is the standard interpretation consistent with the public flow-prediction contract, not a claim to reproduce an undisclosed recipe line for line.

### 8.2 Schedule shifting

Wan remaps a base noise level by:

$$
\sigma'=\frac{s\sigma}{1+(s-1)\sigma}.
$$

For $s>1$, internal points move toward the high-noise end: at $s=5$, $\sigma=0.5$ becomes $0.833$. This redistributes a fixed number of evaluations rather than adding steps. It is model- and resolution-dependent: 5B defaults to 5, A14B T2V to 12, and A14B I2V to 5.

### 8.3 UniPC and DPM++ are integrators

```python
z = randn(shape)
for sigma in schedule:
    v = guided_model(z, sigma, prompt)
    z = solver.step(v, sigma, z)
```

Euler uses only the current slope. UniPC and multistep DPM++ reuse previous model evaluations to build higher-order updates, reducing integration error at a given neural-function-evaluation budget. They do not add model knowledge or repair mismatched prompts, VAEs, or checkpoints.

CFG performs:

$$
v_{\text{cfg}}=v_u+w(v_c-v_u).
$$

$v_c-v_u$ is the direction in which text conditioning changes the learned field. A scale above one extrapolates along it. Too large a scale can create saturation, rigid motion, or damaged detail. The native implementation runs conditional and negative/unconditional forwards separately, so $NFE\approx2\times$ sampling steps: 50 steps mean roughly 100 DiT forwards.

## Step 9: How 5B I2V actually pins the first frame

TI2V-5B does not inject a separate image feature sequence. It performs masked generation directly in video-latent space.

### 9.1 Reference preprocessing

The input image is:

1. scaled under a maximum-area constraint;
2. center-cropped to dimensions aligned to 32;
3. normalized from $[0,1]$ to $[-1,1]$;
4. given a time axis of length one;
5. encoded by the Wan 2.2 VAE into a clean `[48,1,H_z,W_z]` latent $z_{\text{ref}}$.

Define a broadcastable mask:

$$
m_t=\begin{cases}0,&t=0,\\1,&t>0.\end{cases}
$$

The initial latent is:

$$
z=(1-m)z_{\text{ref}}+m\epsilon.
$$

The first latent slice comes from the image and every future slice from noise. After every solver update, the source re-applies:

$$
z\leftarrow(1-m)z_{\text{ref}}+mz,
$$

so numerical integration cannot drift the condition.

The mask is subsampled with `[:, ::2, ::2]` to match the DiT's 2×2 patch grid, producing token timesteps:

$$
t_i=\begin{cases}
0,&\text{first-frame token},\\
t,&\text{future token}.
\end{cases}
$$

Three mechanisms work together: **clean reference latent, re-pinning after every step, and token-wise $t=0$**. T2V is the degenerate case with an all-one mask: every position starts noisy, receives the same $t$, and nothing is pinned.

This unified formulation suggests extensions such as endpoints, spacetime inpainting, or known-clip continuation—but only if training exposed the model to those mask patterns.

## Step 10: What A14B's two experts really are

Wan 2.2 A14B contains two complete, roughly 14B-parameter DiT checkpoints:

```text
high-noise expert → early sampling: layout, camera, large-scale motion
low-noise expert  → late sampling: texture, edges, fine detail
```

Total capacity is about 27B parameters, but only one roughly 14B expert runs at a step. **A14B means Active 14B.**

### 10.1 This is not token-routed sparse MoE

There is no router choosing an FFN expert per token. The source uses deterministic timestep routing:

```python
boundary = config.boundary * 1000
if timestep >= boundary:
    model = high_noise_model
else:
    model = low_noise_model
```

T2V uses `boundary=0.875`; I2V uses `0.900`. This compares the shifted scheduler timestep, so it does not mean that exactly 87.5% or 90% of the discrete sampling iterations use a particular expert. The nonlinear schedule determines the actual count.

Each regime may also use its own CFG scale:

```text
T2V-A14B: low 3.0, high 4.0
I2V-A14B: low 3.5, high 3.5
```

### 10.2 5B and A14B side by side

| Property | TI2V-5B | T2V/I2V-A14B |
|---|---:|---:|
| DiT experts | one dense 5B | two complete experts, about 27B total / 14B active |
| Hidden width | 3072 | 5120 |
| FFN width | 14336 | 13824 |
| Heads / layers | 24 / 30 | 40 / 40 |
| Head dimension | 128 | 128 |
| VAE | Wan 2.2, 48 channels | Wan 2.1, 16 channels |
| VAE stride | `(4,16,16)` | `(4,8,8)` |
| DiT patch | `(1,2,2)` | `(1,2,2)` |
| Default frames / FPS | 121 / 24 | 81 / 16 |
| Default steps | 50 | 40 |
| I2V condition | masked clean latent + token-wise timestep | mask + encoded reference concatenated to noisy latent channels |

At 81 frames and 720×1280, A14B sees:

```text
RGB              [3, 81, 720, 1280]
Wan 2.1 latent   [16, 21, 90, 160]
patch grid       [21, 45, 80]
video tokens     21×45×80 = 75,600
hidden width     5,120
```

That is about 2.77× as many tokens as the 5B example, at a wider hidden dimension. This makes the efficiency contribution of the new high-compression VAE concrete.

### 10.3 A14B I2V uses a different conditioning contract

A14B I2V builds a video containing the reference RGB frame followed by zero frames and encodes it with the Wan 2.1 VAE. It concatenates the resulting 16-channel reference latent with a 4-channel temporal mask to form a 20-channel condition $y$. The DiT then concatenates noisy $x$ and $y$:

```text
noisy latent x       16 channels
condition y          4-channel mask + 16-channel encoded reference
patch input          36 channels
model output         16-channel flow
```

It does not use the 5B model's per-step first-frame re-pinning. Consequently, “Wan2.2 I2V” is not one interchangeable tensor interface. Adapters, Control modules, and checkpoint conversions must distinguish TI2V-5B from I2V-A14B.

### 10.4 Active compute versus resident memory

“14B active” describes compute, not automatic storage of only one checkpoint. Approximate BF16 raw-weight sizes are:

```text
5B:          about 10 GB
one 14B expert: about 28 GB
two experts: about 54 GB for roughly 27B total parameters
```

These figures exclude T5, VAE, activations, CUDA workspaces, and allocator fragmentation. CPU offload can move the inactive expert out of GPU memory; FSDP can shard weights. Measure the actual target stack rather than treating raw-weight arithmetic as a memory guarantee.

## Step 11: From inference to training and LoRA

A base training loop consistent with the public inference contract looks like:

```python
video, caption = sample_batch()
z0 = vae.encode(video)                 # usually frozen; cacheable
context = t5(caption)                   # usually frozen; cacheable

sigma = sample_noise_level()
eps = randn_like(z0)
z_sigma = (1 - sigma) * z0 + sigma * eps
target = eps - z0

pred = dit(z_sigma, sigma, context)
loss = weighted_mse(pred, target, sigma)
loss.backward()
optimizer.step()
```

Production-scale training additionally needs video filtering and captioning, mixed image/video batches, aspect-ratio and duration buckets, a noise-level sampling distribution, condition dropout, distributed parallelism, and precision policy. The official Wan2.2 repository does not publish that complete base-training stack, so `generate.py` alone cannot reproduce pretraining.

### 11.1 Why CFG needs condition dropout

One network must learn both $v_c$ and $v_u$. During training, some examples therefore need their text condition dropped or replaced. Only then can inference extrapolate the difference. A fine-tune that always retains an extremely strong caption can damage the unconditional branch.

For 5B I2V, training must also expose a clean reference slice, noisy future latents, and corresponding token-wise timesteps/masks. A T2V-only all-latent-same-noise loop does not teach reliable first-frame conditioning. A14B I2V instead requires its 20-channel condition and 36-channel patch-input contract.

### 11.2 Where LoRA fits

Common targets are:

```text
self-attention: q, k, v, o
cross-attention: q, k, v, o
optional: the two FFN linear layers
```

For frozen $W$, LoRA learns:

$$
W'=W+\frac{\alpha}{r}BA,
$$

with $A\in\mathbb R^{r\times d_{in}}$, $B\in\mathbb R^{d_{out}\times r}$, and small rank $r$. Attention-only tuning is the lower-memory starting point. Strong identity or style adaptation may benefit from FFN targets, with greater overfitting risk.

A14B has two parameter sets. A low-noise-only LoRA mostly changes the detail regime; a high-noise-only LoRA mostly changes composition and motion. Before attaching one adapter to both, verify parameter names, shapes, and the training timestep distribution.

A robust progression is:

1. validate data, captions, caches, and loss on TI2V-5B;
2. overfit a tiny dataset at short duration/lower resolution;
3. scale data and evaluate prompt generalization, identity, motion, and overfitting;
4. only then move to A14B and inspect high-/low-noise expert behavior separately.

### 11.3 What can be cached

| Cacheable | Required invariants |
|---|---|
| VAE latents | crop, frame sampling, VAE, and normalization |
| T5 embeddings | caption, tokenizer, T5, prompt extension |
| Negative embeddings | negative prompt and T5 |

Noise, $\sigma$, condition dropout, and most augmentation should remain online random variables. The official README lists DiffSynth-Studio as a community stack supporting Wan 2.2 LoRA and full training; it supplies engineering machinery, not a substitute for understanding these tensor contracts.

## Step 12: Multi-GPU and memory engineering

### 12.1 Different switches solve different bottlenecks

| Mechanism | What it addresses | Main cost |
|---|---|---|
| `--t5_cpu` | Removes resident T5 GPU memory | Slower prompt encoding, usually once per prompt |
| `--offload_model True` | Moves inactive DiT/expert modules to CPU | PCIe transfer and synchronization |
| `--dit_fsdp`, `--t5_fsdp` | Shards model parameters across GPUs | Communication and complexity |
| `--ulysses_size N` | Splits long attention work over N ranks | all-to-all communication; parallel size must fit head layout |

FSDP is **parameter parallelism**; Ulysses is **sequence/attention parallelism**. A checkpoint can fit while 75,600-token activations do not, in which case FSDP alone is insufficient.

The Ulysses implementation first partitions $x$ and timestep modulation along sequence. After QKV projection, all-to-all exchanges sequence and head dimensions so each rank owns a subset of heads over the needed sequence. Outputs are gathered before unpatchification. Global FLOPs remain; per-device activation memory decreases.

`--convert_model_dtype` converts DiT parameters to configured BF16 and autocasts activations. Norms, timestep embeddings, and modulation deliberately use FP32 in sensitive locations. FP8 or integer quantization is a separate technique requiring kernels, calibration, and quality checks.

## Step 13: A reproducible source-reading lab

Much of the architecture can be verified without downloading full weights.

### 13.1 Static inspection

```bash
rg "vae_stride|patch_size|dim|ffn_dim|num_heads|num_layers" wan/configs
rg "patch_embedding|time_projection|WanAttentionBlock" wan/modules/model.py
rg "mask2|temp_ts|latent = \(1\. - mask" wan/textimage2video.py
rg "high_noise_model|low_noise_model|boundary" wan/text2video.py wan/image2video.py
```

The goal is to derive latent shape, token count, head dimension, and I2V input channels independently from configuration.

### 13.2 Shape tracing

In a runnable environment, inspect only these boundaries:

```text
T5 output
VAE encode output
patch_embedding output
first block input/output
head output
VAE decode output
```

Log `shape`, `dtype`, `device`, and summary statistics—not full 27,280-token tensors.

### 13.3 Three informative ablations

Hold prompt and seed fixed and change one variable at a time:

1. CFG `3 / 5 / 7`: prompt adherence versus saturation and motion;
2. steps `20 / 30 / 50`: integration error versus runtime;
3. shift near the model default: high-noise versus low-noise trajectory allocation.

Keep checkpoint, solver, size, frames, negative prompt, and prompt extension fixed, or the comparison is not attributable.

### 13.4 Debugging order

| Symptom | Check first |
|---|---|
| Patch-embed shape mismatch | Mixed 5B/A14B checkpoint or condition channels |
| Wrong output dimensions | `(width,height)` order, total alignment, I2V area logic |
| First-frame drift | 5B mask, re-pinning, token-wise timestep |
| Broken colors/output | VAE version, latent mean/std, dtype |
| Weak prompt following | both contexts, CFG, prompt extension, T5 checkpoint |
| Multi-GPU hang | world size, Ulysses size, NCCL init, identical rank control flow |
| OOM | identify whether weights, attention activations, or VAE decoding peak |

## One complete mental model

```text
                           ┌─────────────────────────────┐
prompt ──UMT5-XXL────────▶│ text context [≤512,4096]   │
                           └──────────────┬──────────────┘
                                          │ cross-attention
RGB/reference image                       │
   ├─ T2V: no encoding                    │
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
                              [5B I2V re-pins first slice]
                                              ▼
                                      clean video latent
                                              │
                                      causal VAE decode
                                              ▼
                                           RGB video

A14B additionally routes high sigma to the high-noise DiT and low sigma to
the low-noise DiT.
```

## Mastery check

You understand Wan rather than merely calling it if you can answer these without the chapter:

1. Why do 121 RGB frames become 31 latent time positions?
2. Why does 5B use 704×1280 while A14B can use 720×1280?
3. How do VAE compression and DiT patchification differ?
4. Where do 27,280 tokens come from?
5. How are 3D RoPE dimensions divided across time, height, and width?
6. What information enters through self-attention, cross-attention, and timestep modulation?
7. Why are Flow Matching, solver, schedule, and CFG separate concepts?
8. How do clean latents, masks, re-pinning, and token-wise timesteps implement 5B I2V?
9. Why is A14B not token-routed MoE?
10. Why are A14B I2V and 5B TI2V adapters not directly interchangeable?
11. Which memory pressure does FSDP address, and which does Ulysses address?
12. Which training claims are verifiable from public source and which remain undisclosed?

## Glossary

| Term | Precise meaning here |
|---|---|
| latent | Continuous, normalized video representation produced by the VAE |
| Video DiT | Transformer predicting a flow field over noisy video-latent tokens |
| Flow Matching | Training framework for a conditional vector field between noise and data distributions |
| scheduler | Component that constructs discrete timesteps/sigmas and conversions |
| solver | Numerical method that updates the latent using vector-field predictions |
| CFG | Linear extrapolation between conditional and negative/unconditional predictions |
| causal VAE | VAE that does not read future frames; this does not make the DiT autoregressive |
| 3D RoPE | Rotary positions for time, height, and width on self-attention Q/K |
| A14B | Roughly 27B total parameters with about 14B active at each step |
| offload | Moving inactive modules between CPU and GPU to reduce resident GPU memory |
| FSDP | Parameter sharding across GPUs |
| Ulysses | Attention parallelism using all-to-all sequence/head rearrangement |

## Sources and next step

- [Official Wan2.2 repository and commands](https://github.com/Wan-Video/Wan2.2)
- [TI2V-5B configuration](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)
- [5B T2V/I2V pipeline](https://github.com/Wan-Video/Wan2.2/blob/main/wan/textimage2video.py)
- [Video DiT and 3D RoPE](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)
- [Wan 2.2 high-compression VAE](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py)
- [A14B T2V expert switching](https://github.com/Wan-Video/Wan2.2/blob/main/wan/text2video.py)
- [A14B I2V conditioning](https://github.com/Wan-Video/Wan2.2/blob/main/wan/image2video.py)
- [Wan 2.1 technical report for architectural and training background](https://arxiv.org/abs/2503.20314)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

The best next exercise is a small shape tracer that computes RGB shape, VAE latent shape, patch grid, token count, head dimension, and theoretical $L^2$ from configuration, then checks those values during a short, low-resolution inference. Once every boundary is explainable, moving to a LoRA data pipeline and training loss becomes much less expensive to debug.
