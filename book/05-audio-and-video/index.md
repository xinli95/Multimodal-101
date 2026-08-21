# 05 · Audio and Video — The Other Two Front Ends

**Position in the pipeline**: `audio ──► feature extractor ──► Gemma4AudioModel ──► soft tokens` and `video ──► frame sampling ──► Gemma4VisionModel`

Two modalities, two completely different amounts of new machinery. Video reuses the vision tower entirely — a video is frames, frames are images, and the only new component is a sampler that decides *which* frames. Audio, by contrast, gets a purpose-built encoder that looks nothing like the vision tower: convolutional subsampling, relative position encoding, and chunked attention with an explicit left context. That asymmetry is the lesson of this chapter.

Audio is native to **E2B and E4B only**. The 31B and 26B-A4B checkpoints ship without an audio tower — `Gemma4Model.__init__` simply skips it when `audio_config is None` (chapter 01).

## What you will learn

1. How a waveform becomes model input: 128 mel bins at 16kHz, then 4× temporal downsampling through a stacked convolution projection — which is exactly where "40ms per audio token" comes from
2. Chunked attention: `attention_chunk_size=12`, `attention_context_left=13`, `attention_context_right=0`, and why a streaming-friendly encoder is shaped that way instead of using full attention
3. The Conformer-family building blocks — causal 1D convolution, lightweight conv modules, relative positional encoding, logit capping — and why speech encoders keep convolutions when vision gave them up
4. How video sampling works, what `do_sample_frames` / `num_frames` / `fps` actually control, and what the token cost of a clip really is
5. **Design space**: Gemma 4's audio tower vs. Whisper's encoder-decoder vs. CTC/transducer streaming ASR — and what a general multimodal model gives up against a dedicated ASR model

## Source map

| File | Symbol | Role |
|---|---|---|
| `feature_extraction_gemma4.py` | [`Gemma4AudioFeatureExtractor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/feature_extraction_gemma4.py#L49) | Waveform → 128-bin mel features at 16kHz |
| `video_processing_gemma4.py` | [`Gemma4VideoProcessor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/video_processing_gemma4.py#L158) | Frame sampling (`do_sample_frames`, `num_frames`, `fps`) and per-frame preprocessing |
| `modeling_gemma4.py` | [`Gemma4AudioSubSampleConvProjection`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L385) (+ [`Gemma4AudioSubSampleConvProjectionLayer`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L357)) | The 4× temporal downsample |
| | [`Gemma4AudioAttention`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L249) ([`_convert_to_block`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L278), [`_extract_block_context`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L286), [`_rel_shift`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L296)) | Chunked attention with left context |
| | [`Gemma4AudioCausalConv1d`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L451), [`Gemma4AudioLightConv1d`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L484), [`Gemma4AudioFeedForward`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L415) | Conformer-style blocks |
| | [`Gemma4AudioRelPositionalEncoding`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L218) | Relative position encoding |
| | [`Gemma4AudioModel`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1931), [`_convert_4d_mask_to_blocked_5d`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1955) | The assembled tower and its mask plumbing |

## Walkthrough — audio

`Gemma4AudioModel`'s docstring names its ancestry outright:

> *An audio encoder based on the [Universal Speech Model](https://huggingface.co/papers/2303.01037) architecture.*

This is a USM/Conformer descendant, not a repurposed ViT — and reading it after chapter 04 is instructive precisely because so little transfers.

### 1. Waveform → mel frames

`Gemma4AudioFeatureExtractor`'s shipped settings:

| Field | Value | Meaning |
|---|---|---|
| `sampling_rate` | 16000 | 16kHz mono |
| `feature_size` | 128 | mel bins |
| `frame_length` | 320 | 20ms analysis window |
| `hop_length` | 160 | **10ms per frame** |
| `fft_length` | 512 | |
| `min_frequency` / `max_frequency` | 0 / 8000 | full Nyquist band |
| `preemphasis_htk_flavor` | `true` | HTK-compatible pre-emphasis |

One mel frame every 10ms. Hold onto that number.

### 2. Subsampling: 4× in time, 4× in frequency

```python
class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
    self.norm = nn.LayerNorm(out_channels, ...)
    self.act  = nn.ReLU()
```

Two of these stacked, over the mel spectrogram treated as a **single-channel image** (`input_features.unsqueeze(1)` → `[batch, 1, time, 128]`). Each halves both axes, so after two: time ÷4, mel bins 128 → 32.

**10ms per frame × 4 = 40ms per token.** That is where chapter 02's `audio_ms_per_token = 40` comes from, and why the processor can predict token counts without running the model. A 3.4-second clip: 54,400 samples → 339 mel frames → 170 → **85 audio tokens**, exactly 3400ms ÷ 40.

The channel arithmetic is worth a second look:

```python
proj_input_dim = (config.subsampling_conv_channels[0] // 4) * config.subsampling_conv_channels[1]
self.input_proj_linear = nn.Linear(proj_input_dim, config.hidden_size, bias=False)
```

`(128 // 4) * 32 = 1024 = hidden_size`. The `// 4` is the *frequency* reduction (128 mel bins → 32), and 32 is the second conv's output channels. Flatten 32 frequency positions × 32 channels and you land exactly on the model width. Nothing here is exposed in the config — `subsampling_conv_channels = (128, 32)` is, but the kernel, stride and padding are hard-coded, which is exactly the coupling chapter 02's `_compute_audio_num_tokens` docstring warns about.

The mask rides along, subsampled by the cheapest possible means:

```python
if mask is not None:
    hidden_states = hidden_states * mask[:, None, :, None]   # zero the padding first
...
    mask = mask[:, ::2]                                       # then decimate
```

### 3. Chunked attention with a left context

The heart of the chapter. Three numbers from the config:

```python
self.chunk_size        = config.attention_chunk_size        # 12
self.max_past_horizon  = config.attention_context_left - 1  # 12
self.max_future_horizon = config.attention_context_right    # 0
self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon   # 24
```

Queries are cut into non-overlapping blocks of 12 tokens; keys and values are gathered into **overlapping** windows of 24 that extend 12 tokens into the past and 0 into the future.

```python
query_states = self._convert_to_block(query_states)       # [B, num_blocks, 12, H, D]
key_states   = self._extract_block_context(key_states)    # [B, num_blocks, 24, H, D]
value_states = self._extract_block_context(value_states)
```

`_convert_to_block` is a pad-and-reshape. `_extract_block_context` is a pad-and-`unfold` — the sliding-window trick that materialises overlapping views without copying more than necessary.

In real terms: 12 tokens is 480ms, 24 is 960ms. **The encoder never looks more than about a second back and never looks forward at all.** That is a deliberate streaming-shaped design — with `attention_context_right = 0` this encoder can in principle run causally on a live microphone. Compare Whisper, which attends over a full padded 30-second window bidirectionally and therefore cannot start until the utterance is over.

Attention cost is the other half of the argument. Full attention over a 10-minute recording (15,000 tokens) is 225M pairs per head; chunked attention is `num_blocks × 12 × 24` ≈ 360K. Speech is local — phonemes do not depend on what was said eight minutes ago — so the quadratic term buys almost nothing, and the model spends its long-range budget in the language decoder instead.

**Relative position, Transformer-XL style.** There is no RoPE here. Instead the classic decomposition, with the source citing the paper directly:

```python
relative_key_states = self.relative_k_proj(position_embeddings)
matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)          # content × content
matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)  # content × position
matrix_bd = self._rel_shift(matrix_bd)                            # appendix B of 1901.02860
attn_weights = matrix_ac + matrix_bd
```

`_rel_shift` is the pad-reshape-slice manoeuvre that turns a matrix of relative-distance scores into the correctly aligned per-query offsets. It looks like nonsense until you draw it; it is one of the better exercises in the notebook.

**Two scaling tricks you will not find elsewhere in this model:**

```python
self.q_scale = (self.head_dim**-0.5) / math.log(2)
self.k_scale = math.log(1 + math.e) / math.log(2)
...
query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
```

Dividing both scales by `log 2` converts the subsequent softmax into base-2 — an `exp2` is cheaper than an `exp` on most hardware, and folding the constant into the projections makes it free. And `per_dim_scale` is a learned per-dimension query scale, passed through `softplus` to keep it positive: the model can learn that some head dimensions matter more than others, which a single scalar `1/√d` cannot express.

**Logit softcapping**, the same idea as the decoder's `final_logit_softcapping`:

```python
attn_weights = torch.tanh(attn_weights / self.softcap) * self.softcap   # softcap = 50.0
```

A smooth ceiling instead of a hard clamp, so gradients survive.

**The mask has to be reshaped to match.** The library builds a normal 4D mask and then folds it into block layout:

```python
attention_mask = create_bidirectional_mask(
    config=self.config, inputs_embeds=hidden_states, attention_mask=output_mask,
    and_mask_function=sliding_window_mask_function((self.config.attention_context_left - 1,
                                                    self.config.attention_context_right)))
attention_mask = self._convert_4d_mask_to_blocked_5d(attention_mask)
```

`_convert_4d_mask_to_blocked_5d` gathers `[B, 1, seq, seq]` into `[B, 1, num_blocks, 12, 24]` using explicitly constructed `kv_indices`. Note that `create_bidirectional_mask` is used — within its 24-token window, an audio token sees both directions. "Chunked" is not "causal".

### 4. The layer: a Conformer, rearranged

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

The macaron sandwich — feed-forward, attention, convolution, feed-forward — is Conformer's signature: attention handles global structure, the depthwise convolution (`Gemma4AudioLightConv1d`, with a **causal** `Gemma4AudioCausalConv1d` inside) handles local structure, and the halved feed-forwards wrap both. Note the residual is captured *after* `feed_forward1`, not before it.

And note `torch.clamp` three times per layer, with a bound that is itself clamped to the dtype:

```python
gradient_clipping = min(self.gradient_clipping, torch.finfo(self.norm_pre_attn.weight.dtype).max)
```

Between this, the clippable linears, the logit softcap and the float32 attention, the audio tower is by some distance the most numerically defensive code in the model. Speech encoders operate on inputs with enormous dynamic range, and it shows.

Finally:

```python
self.output_proj = nn.Linear(config.hidden_size, config.output_proj_dims, bias=True)
```

1024 → 1536, and the only projection in the tower **with a bias**. On E2B, `output_proj_dims = 1536` happens to equal the text model's `hidden_size`, so `embed_audio`'s job is a same-width refinement rather than a resize.

### 5. Why the large models have no audio tower

Chapter 01's table: `"audio_config": null` on 31B and 26B-A4B. Everything above is roughly 12 layers × 1024 wide plus the conv stack — small in parameters, but it is a *separately trained encoder* requiring paired speech data, its own evaluation regime, and its own numerical-stability work. The Gemma 4 release put that investment into the sizes meant to run on a phone, where an on-device voice interface is the point. That is a product decision made legible by one `null` in a JSON file.

## Walkthrough — video

Video needs almost no new machinery, and the little it has is revealing.

`Gemma4VideoProcessor`'s shipped settings differ from the image processor in exactly the way you would hope:

| Field | Image | Video |
|---|---|---|
| `max_soft_tokens` | 280 | **70** |
| `num_frames` | — | 32 |
| `do_sample_frames` | — | `true` |
| `do_normalize` | `false` | `true` (with mean 0, std 1 — a no-op) |

**A video frame is worth a quarter of a still image.** 32 frames × 70 soft tokens = **2,240 tokens** for a clip, against 280 for one photo. Eight images' worth of context for a whole video, which is the trade that makes video fit in a prompt at all. It also tells you what to expect: Gemma 4 will read a clip's *narrative* well and its *fine detail* badly. Reading a licence plate in a video is a still-image job.

Frame sampling is controlled by `do_sample_frames`, `num_frames` and `fps` — uniform sampling to a fixed count by default. And then the frames go through the same `Gemma4VisionModel` as any image; there is no temporal attention, no 3D convolution, no video-specific tower.

Time is instead carried in the *prompt*, as chapter 02 showed:

```python
timestamp_str = [f"{int(seconds // 60):02d}:{int(seconds % 60):02d}" for seconds in metadata.timestamps]
video_replacement = " ".join([f"{t} {self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}"
                              for t in timestamp_str])
```

`00:00 <boi>…<eoi> 00:02 <boi>…<eoi> …` — literal text timestamps interleaved with frame tokens. The language model learns to read them the way it reads any other text. If `fps` cannot be inferred the processor warns and assumes 24, and your timestamps silently become wrong, which is worth guarding against in a pipeline.

## Design space

**Audio encoders:**

| | Gemma 4 | Whisper | CTC/transducer (Parakeet, Canary) |
|---|---|---|---|
| Attention | Chunked, 24-token window, no lookahead | Full, over a padded 30s window | Streaming by construction |
| Position | Transformer-XL relative | Sinusoidal absolute | varies |
| Output | Soft tokens into an LLM | Text, via its own decoder | Text, via alignment |
| Streaming | Structurally capable | No — needs the whole window | Yes, its whole reason to exist |
| Long audio | Bounded by LLM context | 30s chunks, stitched | Unbounded |

The interesting comparison is not accuracy but *what the output is for*. Whisper's encoder feeds a decoder that only knows how to emit transcripts. Gemma 4's feeds a general language model, so "transcribe this" and "what is the speaker's mood, and does it match the slide they are describing" are the same code path. You give up a specialist's word-error-rate and gain composition with every other modality. The notebook runs both on the same audio so you can see the size of that trade rather than assume it.

**Video:** the field splits into models that resample frames into a text-like sequence and let the LLM do the temporal reasoning (Gemma 4, most open VLMs), and models with genuine temporal machinery — 3D patches, temporal attention, or time-aware position encoding (Qwen-VL's M-RoPE time axis, video generation models in [chapter 21](../21-video-generation/overview.md)). Frame-sampling is dramatically simpler and turns out to be enough for question answering; it is not enough for anything requiring precise motion.

## Check yourself

1. Derive 40ms per audio token from the feature extractor's settings and the SSCP stack. Which parameters would you have to change to make it 20ms?
2. `attention_context_right = 0`. What does that enable, and what does it cost?
3. Why does the audio tower use Transformer-XL relative position instead of the 2D RoPE from chapter 04?
4. What are `q_scale` and `k_scale` dividing by `log 2` for?
5. A 32-frame clip and a single photo: which costs more context, and by how much? What does that predict about the model's video weaknesses?
6. Where is the timestamp of frame 17 represented in the model's input? (There is exactly one place.)

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_audio_tower_anatomy.ipynb` | Trace a waveform through mel extraction and convolutional subsampling with shapes printed at each step; rebuild the chunked-attention mask by hand and `assert_close` it against the library's | 🟢 CPU |
| `02_asr_and_video.ipynb` | Real E2B weights: transcription, audio question answering, then video QA — with the token cost of a clip measured, not guessed | 🟡 24GB VRAM |
| [`03_whisper_asr_compare.ipynb`](notebooks/03_whisper_asr_compare.ipynb) | The design-space control: Whisper on the same audio. What a specialist gets you, and what it cannot do | 🟢 CPU / 🟡 |
