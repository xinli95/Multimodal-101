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
| `feature_extraction_gemma4.py` | `Gemma4AudioFeatureExtractor` | Waveform → 128-bin mel features at 16kHz |
| `video_processing_gemma4.py` | `Gemma4VideoProcessor` | Frame sampling (`do_sample_frames`, `num_frames`, `fps`) and per-frame preprocessing |
| `modeling_gemma4.py` | `Gemma4AudioSubSampleConvProjection` (+ `...Layer`) | The 4× temporal downsample |
| | `Gemma4AudioAttention` (`_convert_to_block`, `_extract_block_context`, `_rel_shift`) | Chunked attention with left context |
| | `Gemma4AudioCausalConv1d`, `Gemma4AudioLightConv1d`, `Gemma4AudioFeedForward` | Conformer-style blocks |
| | `Gemma4AudioRelPositionalEncoding` | Relative position encoding |
| | `Gemma4AudioModel`, `_convert_4d_mask_to_blocked_5d` | The assembled tower and its mask plumbing |

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_audio_tower_anatomy.ipynb` | Trace a waveform through mel extraction and convolutional subsampling with shapes printed at each step; rebuild the chunked-attention mask by hand and `assert_close` it against the library's | 🟢 CPU |
| `02_asr_and_video.ipynb` | Real E2B weights: transcription, audio question answering, then video QA — with the token cost of a clip measured, not guessed | 🟡 24GB VRAM |
| [`03_whisper_asr_compare.ipynb`](notebooks/03_whisper_asr_compare.ipynb) | The design-space control: Whisper on the same audio. What a specialist gets you, and what it cannot do | 🟢 CPU / 🟡 |
