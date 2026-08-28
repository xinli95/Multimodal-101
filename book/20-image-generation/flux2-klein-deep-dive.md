# FLUX.2 [klein] 4B Deep Dive: Image Generation Through Diffusers Source

This section follows one real Diffusers call from a prompt to RGB pixels. It introduces VAE compression, Flow Matching, MMDiT, and Euler integration only when each concept appears in the implementation.

By the end, four questions should have precise answers:

1. If the VAE downsamples by 8, why do channels grow from 3 to 32 and then to 128 at the Transformer boundary?
2. What is the training state $z_t$, and where are the boundaries of $t$ and the noise distribution?
3. What does it mean to predict a velocity field, and how does Euler integration turn noise into an image?
4. What do FLUX.2's 5 double-stream and 20 single-stream blocks do in code?

The `pipelines/flux2/` directory is orchestration rather than the whole model:

| Layer | Main source | Responsibility |
|---|---|---|
| Pipeline | [`pipeline_flux2_klein.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py) | Prompt encoding, latent preparation, sampling loop, VAE decode |
| Transformer | [`transformer_flux2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py) | Conditional velocity field $v_\theta(z_t,t,c)$ |
| Scheduler | [`scheduling_flow_match_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py) | Time grid and Euler updates |
| VAE | [`autoencoder_kl_flux2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py) | RGB ↔ 32-channel latents |
| Config | [FLUX.2-klein-4B `transformer/config.json`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/transformer/config.json) | Actual 4B depth and width |

`pipeline_flux2.py` serves the larger family; `pipeline_flux2_klein.py` is our entry point. The inpaint variant adds an initial image and mask, while the KV variant caches reference-image keys and values.

## 1. End-to-end data flow

For batch size 1, a 1024×1024 output, and 512 text tokens:

```text
prompt → Qwen3 (layers 9/18/27) → text [1,512,7680] → Linear → [1,512,3072]

Gaussian [1,128,64,64] → flatten [1,4096,128] → Linear → [1,4096,3072]
                                                        │
timestep → sinusoidal embedding → MLP → shift/scale/gate│
                                                        ▼
                          5 × double-stream joint-attention block
                                                        │
                         concat text + image [1,4608,3072]
                                                        │
                          20 × single-stream joint block
                                                        │
                     drop text/reference → Linear 3072→128
                                                        │
                              velocity [1,4096,128]
                                                        │
                              Euler update, ~4 steps
                                                        │
             unpack + unpatchify [1,32,128,128] → VAE decode
                                                        │
                                      RGB [1,3,1024,1024]
```

The Transformer does not draw RGB or produce a finished image in one forward. It predicts a local direction; the scheduler advances the latent, and the VAE decodes only after several updates.

## 2. Why 8× VAE compression produces 32 channels

For a 1024×1024 image:

$$
[B,3,1024,1024]\xrightarrow{\mathrm{VAE}}[B,32,128,128].
$$

“8×” describes each spatial axis. It does not promise fewer channels. Encoders trade a large spatial grid for more feature channels, moving local color, edge, texture, and semantic information into a learned basis. The total value count still falls from $3{,}145{,}728$ to $524{,}288$. The published [VAE config](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/vae/config.json) confirms `latent_channels=32`.

### 2.1 VAE compression is not Transformer patchification

The pipeline then performs a fixed 2×2 rearrangement:

$$
[B,32,128,128]\rightarrow[B,128,64,64]\rightarrow[B,4096,128].
$$

No elements are lost: each 2×2 neighborhood moves into the feature dimension. Therefore:

- **32** is the VAE latent channel count;
- **128** is one flattened 2×2 latent patch;
- **4096** is the number of image tokens, $64\times64$;
- **3072** is the Transformer hidden size after projection.

Packing and its inverse live in the pipeline's [latent utilities](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L370-L426). Pure T2I samples Gaussian noise directly as `[B,128,64,64]`, the already-patchified representation.

## 3. Flow Matching: state, path, and velocity

Let $z_0$ be a real-image VAE latent, $\epsilon\sim\mathcal N(0,I)$ an equal-shaped Gaussian sample, and $t\in[0,1]$. The simplest Rectified Flow path is

$$
z_t=(1-t)z_0+t\epsilon,
$$

with explicit boundaries

$$
t=0\Rightarrow z_t=z_0,
\qquad
t=1\Rightarrow z_t=\epsilon.
$$

$z_t$ is the random state at time $t$ on a prescribed probability path. A “partly noisy image” is useful intuition, not the definition.

Training need not simulate the entire path. One example encodes one image, samples one $\epsilon$ and one $t$, computes $z_t$ directly, and regresses the local velocity. Differentiating the path gives

$$
\frac{dz_t}{dt}=\epsilon-z_0,
$$

so the basic conditional objective is

$$
\mathcal L=\mathbb E_{z_0,\epsilon,t}
\left[\left\|v_\theta(z_t,t,c)-(\epsilon-z_0)\right\|_2^2\right].
$$

One paired line has constant velocity, but inference hides the original pair. From only $(z_t,t,c)$, the model must learn the conditional expected local direction over the whole dataset.

Increasing time runs data → noise. Generation starts at $t=1$ and integrates the same field toward smaller $t$; no separate reverse model is trained. Diffusers often displays a roughly 0–1000 timestep scale, with `timestep = 1000 * sigma`. That is a numeric convention, not 1000 model calls. The scheduler may also shift the sigma grid with image sequence length; see [`set_timesteps()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L283-L370).

## 4. Conditional vector field and Euler integration

The network assigns a velocity to every latent coordinate:

$$
v_\theta(z,t,c).
$$

Think of latent space as a high-dimensional map with an arrow at every point. $t$ selects the noise regime; changing prompt condition $c$ changes the arrow map.

The continuous system is

$$
\frac{dz}{dt}=v_\theta(z,t,c).
$$

Euler integration makes a local straight-line approximation:

$$
z_{k+1}=z_k+\Delta t_kv_\theta(z_k,t_k,c).
$$

Sampling goes noise → data, so $\Delta t_k<0$. In Diffusers' sigma coordinate:

$$
z_{k+1}=z_k+(\sigma_{k+1}-\sigma_k)v_\theta(z_k,t_k,c).
$$

The scheduler's [`step()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L423-L522) reduces to `sample + dt * model_output`. The pipeline name `noise_pred` is historical; here it semantically carries velocity, not a classic DDPM epsilon estimate.

Training samples continuous times. Inference chooses a finite grid. Klein's few-step distillation is why roughly four evaluations can approximate the path well—it does not mean the model learned only four times.

## 5. Qwen3: prompt to 7680-dimensional condition

[`_get_qwen3_prompt_embeds()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L208-L262) applies a chat template, tokenizes to at most 512 positions by default, runs Qwen3 with hidden-state outputs, selects layers 9, 18, and 27, then concatenates features.

Each selected layer has width 2560:

$$
3\times2560=7680,
\qquad
c_{\text{text}}\in\mathbb R^{B\times L\times7680}.
$$

Multiple depths expose lexical, compositional, and higher-level semantic features. `context_embedder` maps them to the common width:

$$
[B,L,7680]\rightarrow[B,L,3072].
$$

## 6. The FLUX.2 velocity Transformer

| Parameter | Value | Meaning |
|---|---:|---|
| `in_channels` | 128 | One 2×2 VAE-latent patch |
| `joint_attention_dim` | 7680 | Three Qwen3 layers |
| heads × head dimension | 24 × 128 | Hidden size 3072 |
| `num_layers` | 5 | Double-stream blocks |
| `num_single_layers` | 20 | Single-stream blocks |
| `mlp_ratio` | 3 | FFN width 9216 |
| `axes_dims_rope` | `[32,32,32,32]` | Four RoPE axes |
| `guidance_embeds` | false | No separate guidance embedding |

Generic class defaults serve multiple FLUX.2 variants; architectural facts must come from the checkpoint config.

### 6.1 Inputs and timestep modulation

Image and text projections are

$$
[B,4096,128]\rightarrow[B,4096,3072],
\qquad
[B,L,7680]\rightarrow[B,L,3072].
$$

Time follows a separate path:

```text
t → 256-d sinusoidal embedding → MLP → 3072-d time embedding
                                    → shift / scale / gate
```

It is not appended as an ordinary token. AdaLN-style modulation changes each block for the current noise regime. The module graph is built in [`Flux2Transformer2DModel.__init__`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1059-L1219).

### 6.2 Five double-stream MMDiT blocks

Image and text retain separate residual streams and separate modulation/QKV projections. Their Q, K, and V are then concatenated over tokens:

$$
Q=[Q_t;Q_i],\quad K=[K_t;K_i],\quad V=[V_t;V_i].
$$

One joint attention lets image queries read text keys/values and vice versa. The output is split back, then each stream uses its own projection and FFN. This is simultaneously multimodal, jointly attentive, and double-stream. See [`Flux2TransformerBlock`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L876-L968) and [`Flux2AttnProcessor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L327-L395).

### 6.3 Twenty single-stream blocks

After five double-stream blocks:

$$
X=[X_t;X_i],
$$

with default T2I shape `[B,4608,3072]`. Content and coordinates now distinguish modalities rather than separate block parameters. Attention and SwiGLU run in parallel:

$$
A=\operatorname{Attention}(X),\qquad M=\operatorname{SwiGLU}(X),
$$

$$
Y=W_o[A;M],\qquad X\leftarrow X+g(t)\odot Y.
$$

The MLP width is $3072\times3=9216$. See [`Flux2SingleTransformerBlock`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L807-L873) and its [parallel processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L572-L630).

### 6.4 Four-axis RoPE: $(T,H,W,L)$

| Token | Coordinates |
|---|---|
| Text token $l$ | $(0,0,0,l)$ |
| Target position $(h,w)$ | $(0,h,w,0)$ |
| First reference | $(10,h,w,0)$ |
| Second reference | $(20,h,w,0)$ |

Each axis rotates 32 dimensions, totaling the 128-dimensional head. Attention can therefore represent text order, 2D position, and reference identity. IDs are built by the pipeline's [`_prepare_*_ids()` methods](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L266-L366); [`Flux2PosEmbed`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L971-L998) applies RoPE.

### 6.5 Modulation and output

Time produces shift, scale, and gate values:

$$
\hat X=\operatorname{Norm}(X)(1+\operatorname{scale}(t))+\operatorname{shift}(t),
$$

$$
X\leftarrow X+\operatorname{gate}(t)F(\hat X).
$$

This lets one set of weights act differently near noise and data. See [`Flux2TimestepGuidanceEmbeddings` and `Flux2Modulation`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1001-L1056).

Finally, text and reference tokens are removed and target states project from `[B,N_i,3072]` to `[B,N_i,128]`. Velocity must match the latent for Euler's elementwise update. The complete path is in [`Flux2Transformer2DModel.forward`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1226-L1424).

## 7. One `pipeline.__call__()`

Ignoring validation, device management, and callbacks, [`Flux2KleinPipeline.__call__()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L614-L919) is:

```python
prompt_tokens = encode_prompt(prompt)       # [B, L, 7680]
z = prepare_gaussian_latents()              # [B, 4096, 128]
timesteps = scheduler.set_timesteps(steps)

for t in timesteps:
    velocity = transformer(z, t, prompt_tokens, img_ids, txt_ids)
    z = scheduler.step(velocity, t, z)

z = unpack_and_unpatchify(z)                # [B, 32, 128, 128]
image = vae.decode(z)                        # [B, 3, 1024, 1024]
```

[`encode_prompt()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L428-L460) creates text states and IDs; [`prepare_latents()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L479-L510) samples packed Gaussian latents. Each scheduler step updates only the target latent.

## 8. Why generation and editing share one backbone

With references, the pipeline VAE-encodes them and concatenates

$$
X_{\text{image}}=[X_{\text{target noisy}};X_{\text{reference clean}}].
$$

Joint attention reads the references, but the output is cropped to target positions; reference latents are not advanced by the scheduler. Distinct $T$ coordinates identify multiple references. Thus one graph covers T2I, reference editing, and—with initial-image/mask logic—inpainting. The KV variant caches invariant reference K/V to avoid repeated work.

## 9. Guidance, distillation, and four-step Klein

Classic classifier-free guidance uses two forwards:

$$
v_{\text{cfg}}=v_{\text{uncond}}+s(v_{\text{cond}}-v_{\text{uncond}}).
$$

Klein 4B is distilled. Diffusers enters this branch only when `guidance_scale > 1 and not is_distilled`, so distilled Klein skips classic CFG. The generic pipeline's 50-step defaults are not the 4B recommendation; the [official model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) demonstrates roughly four steps:

```text
Qwen3 encode × 1
Flux Transformer × 4
VAE decode × 1
```

Speed comes from both model size and distillation that removes solver steps and the second CFG forward.

## 10. Why the field needs so much data

The learned map is not one image-to-noise curve but

$$
(z_t,t,c)\mapsto v
$$

across visual content, all noise regimes, linguistic relations, resolutions, aspect ratios, and reference combinations. Each image yields many states by resampling $\epsilon$ and $t$, while Qwen3 transfers language/world knowledge. Architecture supplies capacity, not missing concepts: data scale, caption quality, aesthetic filtering, text examples, and editing triplets determine where the field is reliable.

## 11. Source-reading order

1. [`Flux2KleinPipeline.__init__`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L155-L205): components and scales.
2. [`Flux2KleinPipeline.__call__`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L614-L919): end-to-end control flow.
3. `_get_qwen3_prompt_embeds()` and `prepare_latents()`: input shapes.
4. [`Flux2Transformer2DModel.__init__`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1059-L1219): config to modules.
5. [`Flux2Transformer2DModel.forward`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py#L1226-L1424): actual data flow.
6. Double-stream block, single-stream block, scheduler `step()`, then VAE decode.

## 12. Mental-model checkpoint

```text
Training:
real image ─VAE→ z₀; sample ε and t
zₜ = (1-t)z₀ + tε
learn vθ(zₜ,t,prompt) ≈ ε-z₀

Inference:
Gaussian z₁ → predict velocity → Euler with negative step
            → repeat → unpack/unpatchify → VAE decode → RGB
```

In one sentence: **FLUX.2 [klein] 4B uses Qwen3 for text conditioning, 5 double-stream MMDiT blocks plus 20 single-stream blocks for the conditional velocity field, and distilled few-step Rectified Flow integration over 2×2-patched tokens from a 32-channel VAE latent.**
