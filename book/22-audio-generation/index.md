# 22 · Audio Generation — From Waveforms to Audio Language Models

A 24 kHz recording contains 24,000 amplitude values every second. A language model cannot afford 24,000 autoregressive decisions for one second of speech, so audio generation begins by changing the representation of the problem.

The chapter follows one continuous path:

```text
waveform
  → compact acoustic representation
  → generative model
  → reconstructed waveform
```

We will start with the signal itself, build a neural codec from first principles, derive residual vector quantization, compare codec language models with flow matching, and then trace Higgs TTS 3 all the way from text and reference speech to eight code streams and a playable waveform. The goal is not to memorize a model list; it is to be able to look at any new audio generator and identify what it predicts, where it is sequential, and what finally renders sound.

[Chapter 05](../05-audio-and-video/index.md) followed audio into a model for understanding. This chapter runs the arrow in the opposite direction. An understanding encoder is rewarded for discarding speaker and channel details once the words are recognizable; a generation codec is rewarded for preserving enough of those details to reconstruct convincing sound.

## 1. What must a TTS model decide?

For the sentence “The meeting starts at nine,” a synthesizer must produce more than the correct words. It must also choose:

- pronunciation and duration;
- pauses and rhythm;
- pitch contour and energy;
- speaker identity and vocal texture;
- emotion, speaking style, and recording ambience; and
- non-verbal events such as breathing, laughter, or a cough.

The actual problem is therefore

$$
P(\text{audio}\mid\text{text, speaker, style, context}).
$$

The difficulty is the output density. Ten seconds of 24 kHz mono audio contains 240,000 continuous samples. Predicting them one at a time would create an impractically long sequence and force the model to learn both high-level speech planning and low-level waveform rendering in the same loop.

Modern systems introduce an intermediate acoustic representation. The choice of representation largely determines the architecture that follows.

## 2. The representation ladder

| Representation | Shape for one second | What it is good for | What it may lose |
|---|---:|---|---|
| Waveform | 24,000 samples at 24 kHz | Playback; preserves the signal | Nothing by definition, but it is very dense |
| Mel spectrogram | Tens to hundreds of continuous frames | Speech recognition; classical TTS and vocoders | Phase and some fine waveform detail |
| Continuous latent | Model-dependent low-rate vectors | Diffusion / flow generation; learned compression | Whatever its training objective does not preserve |
| Discrete codec tokens | Low-rate integer IDs, often several per frame | Language modeling, prompting, storage, streaming | Quantization and codec reconstruction error |

Chapter 05 used mel features for understanding. A speech recognizer is rewarded for invariance: the same sentence should map to the same words across microphones and speakers. A generation codec is rewarded for reconstruction: microphone texture, pitch, timbre, and background sound may all matter.

That is why an understanding encoder and an audio codec are not interchangeable even though both compress a waveform.

## 3. Codec, codebook, and audio code

These three terms sound similar but refer to different objects.

**Codec** means coder + decoder. A neural audio codec is a learned, lossy compression system:

$$
x
\xrightarrow{\text{encoder}}
h
\xrightarrow{\text{quantizer}}
z
\xrightarrow{\text{decoder}}
\hat{x},
\qquad \hat{x}\approx x.
$$

- $x$: input waveform;
- $h$: a low-rate sequence of continuous latent vectors;
- $z$: discrete integer IDs;
- $\hat{x}$: reconstructed waveform.

**Codebook** is a learned vector dictionary. If a codebook contains 1,024 vectors of dimension 64, then

$$
C\in\mathbb{R}^{1024\times64}.
$$

The quantizer replaces a continuous vector with the ID of a nearby codebook entry:

$$
z=\arg\min_k\lVert h-C_k\rVert^2.
$$

**Audio code** is that integer ID. If entry 137 is selected, the stored or predicted code is `137`. It is not a miniature WAV clip and it need not correspond to one phoneme. It names a learned prototype in latent space.

The text analogy is useful but imperfect:

| Text model | Codec model |
|---|---|
| Vocabulary | Codebook |
| Text token ID | Audio code ID |
| Token embedding | Codebook vector |
| A token often has an interpretable spelling | An audio code usually has no standalone human meaning |

## 4. Why several codebooks? Residual vector quantization

A single 1,024-entry codebook is often too coarse for rich audio. Residual vector quantization (RVQ) approximates one latent vector in stages.

The first codebook chooses $q_1$ to approximate $h$. The next codebook quantizes the remaining error:

$$
r_1=h-q_1,\qquad q_2\approx r_1.
$$

The process continues so that

$$
h\approx q_1+q_2+\cdots+q_K.
$$

The early codebooks establish a coarse approximation; later ones refine the residual. This is a hierarchy of approximation, not a clean semantic division such as “codebook 1 is pronunciation” and “codebook 2 is speaker identity.” Those factors can remain entangled.

A two-dimensional toy example makes the mechanism concrete:

$$
h=[0.72,-0.10],\quad
q_1=[0.60,-0.20],\quad
r_1=[0.12,0.10].
$$

If the second codebook selects $q_2=[0.10,0.08]$, then

$$
q_1+q_2=[0.70,-0.12],
$$

which is already close to $h$. The codec stores two IDs rather than the original floating-point vector.

[SoundStream](https://arxiv.org/abs/2107.03312) established the now-familiar fully convolutional encoder / decoder plus RVQ design. [EnCodec](https://arxiv.org/abs/2210.13438) developed the same family into a practical high-fidelity neural codec.

### A useful bitrate calculation

Higgs represents each 40 ms frame with eight codebooks of 1,024 real entries. Ignoring framing overhead,

$$
25\ \text{frames/s}
\times 8\ \text{codes/frame}
\times \log_2(1024)
=2{,}000\ \text{bits/s}.
$$

By comparison, 24 kHz, 16-bit mono PCM is

$$
24{,}000\times16=384{,}000\ \text{bits/s}.
$$

The token payload is therefore about 192 times smaller. This comparison is intuition, not a file-format benchmark: a real stream also needs metadata and entropy coding may change the final bitrate.

## 5. How codes become a waveform again

Suppose one aligned Higgs frame contains

```text
[137, 921, 44, 818, 31, 402, 9, 615]
```

The codec decoder performs three conceptual operations:

1. Look up each ID in its own RVQ codebook.
2. Sum the eight retrieved vectors to reconstruct a quantized latent $\hat{h}_t$.
3. Run the low-rate latent sequence through learned convolutional upsampling and residual blocks to produce waveform samples.

At 25 frames/s and 24 kHz, the temporal expansion is

$$
24{,}000/25=960
$$

samples per codec frame. The decoder does not copy a latent 960 times. It has learned from paired latents and waveforms how pitch periodicity, transients, phase continuity, timbre, and noise should be rendered.

The reconstruction is lossy:

$$
\hat{x}\ne x,
\qquad
\operatorname{PerceptualSimilarity}(\hat{x},x)\ \text{should be high}.
$$

The decoder can plausibly fill in high-frequency details because its weights contain an audio prior, just as an image decoder can expand a compressed latent into millions of pixels. The codec's reconstruction quality places a ceiling on the downstream generator: even perfect code prediction cannot recover detail the codec itself discards.

### Codec decoder, vocoder, and hardware DAC

| Term | Input | Output |
|---|---|---|
| Traditional neural vocoder | Usually a mel spectrogram | Waveform |
| Neural codec decoder | Quantized codec latents / audio codes | Waveform |
| Hardware DAC | Digital waveform samples | Analog voltage for a speaker |

“DAC” can also name the Descript Audio Codec model family. That neural model is unrelated to the digital-to-analog converter in a sound card.

## 6. The two major modern TTS routes

It is tempting to summarize modern TTS as “an LLM over codec tokens,” but that would erase a second major family.

| | Discrete codec language model | Diffusion / flow-matching TTS |
|---|---|---|
| Typical path | text → discrete audio codes → codec decoder | text + reference → continuous mel / latent → vocoder or decoder |
| Generation | Often autoregressive over codec frames | Usually non-autoregressive over the full acoustic sequence, with several solver steps |
| Natural fit | Decoder-only Transformers, next-token loss, prompting, KV cache | Continuous acoustic modeling, editing, infilling, parallel synthesis |
| Strengths | Streaming; in-context voice prompts; paralinguistic events; text/audio unification | No discrete quantization bottleneck; no AR error accumulation; fast batch generation |
| Costs | Codec ceiling; sampling drift; multi-codebook dependencies; sequential steps | Iterative denoising / ODE solve; duration handling; streaming is less native |
| Examples | VALL-E, Bark, VoiceCraft, Fish Speech, Higgs | Voicebox, E2-TTS, F5-TTS, Matcha-TTS |

[F5-TTS](https://aclanthology.org/2025.acl-long.313/) is a useful counterexample to “all LLM TTS is codec TTS”: it uses a diffusion Transformer with flow matching to generate a continuous mel representation non-autoregressively.

Many systems are hybrids. An LLM may plan semantics or coarse audio tokens while a flow model or neural decoder fills in acoustic detail. The design question is not which family wins universally, but where each system places its discrete bottleneck and sequential loop.

## 7. Where the codec-LM idea came from

Higgs is a mature representative of the paradigm, not its inventor.

| Year | Work | Contribution to the lineage |
|---:|---|---|
| 2021 | [SoundStream](https://arxiv.org/abs/2107.03312) | End-to-end neural audio codec with a convolutional encoder / decoder and RVQ |
| 2022 | [AudioLM](https://arxiv.org/abs/2209.03143) | Recast general audio generation as language modeling over discrete semantic and acoustic tokens |
| 2022 | [EnCodec](https://arxiv.org/abs/2210.13438) | High-fidelity, real-time neural compression for speech, music, and general audio |
| 2023 | [VALL-E](https://arxiv.org/abs/2301.02111) | Recast TTS as conditional neural-codec language modeling and demonstrated 3-second zero-shot voice prompting |
| 2023 | [Bark](https://github.com/suno-ai/bark) | Popularized expressive text-to-audio generation with speech, non-verbal sounds, and community-accessible code |
| 2026 | [Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b) | A clean Qwen3-based, low-frame-rate, parallel multi-codebook implementation aimed at multilingual conversational speech |

AudioLM was primarily an audio-continuation system, not a transcription-conditioned TTS model. VALL-E made the TTS formulation explicit:

$$
P(\text{codec codes}\mid\text{phonemes, acoustic prompt}).
$$

That distinction matters when assigning credit: SoundStream and EnCodec supplied the compression substrate; AudioLM established audio-token language modeling; VALL-E made codec-LM TTS prominent; Higgs packages the mature pattern into a controllable, streamable voice model.

## 8. Higgs TTS 3 as a worked architecture

The shortest correct description is:

> Higgs TTS 3 is a Qwen3-style autoregressive model that continues interleaved text and audio context by predicting delayed, discrete codec codes; a separate neural codec converts the generated codes to a 24 kHz waveform.

```text
target text + control tokens ─────────────────────────────┐
                                                         v
reference audio → Higgs audio tokenizer → reference codes → Qwen3 AR decoder
                                                               │
                                                        8 code streams
                                                               │
                                                        reverse delay
                                                               │
                                                        codec decoder
                                                               │
                                                        24 kHz waveform
```

The [official model card](https://huggingface.co/bosonai/higgs-tts-3-4b) and [configuration](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/config.json) expose the key dimensions:

| Component | Higgs TTS 3 |
|---|---|
| Backbone | Qwen3-style decoder-only Transformer, roughly 4B |
| Transformer | 36 layers, hidden size 2,560, 32 query heads / 8 KV heads |
| Training sequence length | 8,192 tokens |
| Codec | 8 codebooks, vocabulary 1,026 per stream including boundary codes |
| Codec frame rate | 25 fps = 40 ms/frame |
| Waveform sample rate | 24 kHz |

### 8.1 The audio tokenizer

The tokenizer solves

$$
\text{waveform}
\leftrightarrow
[T,8]\ \text{codec IDs}.
$$

Higgs reuses the Higgs Audio V2 tokenizer family. Its documented design combines an acoustic configuration based on DAC with a semantic configuration based on HuBERT, then quantizes the learned representation. This matters because a codec used by an audio language model must balance two objectives:

- preserve acoustic detail so the decoder can reconstruct convincing sound; and
- expose enough linguistic structure that predicting the next code is learnable.

The [Transformers tokenizer documentation](https://huggingface.co/docs/transformers/model_doc/higgs_audio_v2_tokenizer) records the 24 kHz sample rate, 25 fps design goal, 1,024-entry codebooks, 64-dimensional codebook vectors, DAC acoustic branch, and HuBERT semantic branch.

Do not confuse the codec's codebook vectors with Qwen's input embeddings. The codec owns the quantization space used to reconstruct audio. The language model owns the hidden-space embeddings used to reason over IDs.

### 8.2 What Qwen predicts

A normal language model predicts

$$
P(x_t\mid x_{<t}).
$$

During Higgs audio generation, the target is approximately

$$
P\!\left(z_t^{(1)},\ldots,z_t^{(8)}
\mid \text{text, reference audio, previous code rows}\right).
$$

One Transformer hidden state $h_t\in\mathbb{R}^{2560}$ feeds a fused audio head that produces eight categorical distributions:

$$
h_t\longrightarrow\text{logits}_t\in\mathbb{R}^{8\times1026}.
$$

The eight streams are sampled in parallel. The same backbone still consumes ordinary text and control tokens; the prediction target switches to audio codes after the audio boundary.

### 8.3 How eight codes enter one Transformer position

Conceptually, each code stream has its own embedding lookup. The eight vectors for a generation row are fused, typically by summation, into one 2,560-dimensional input:

$$
e_t=\sum_{c=1}^{8}E_c\!\left(z_t^{(c)}\right).
$$

The implementation stores the multi-codebook embedding and output head as fused tensors for efficient lookup and projection. This is what lets a mostly standard Qwen3 backbone process an unusual eight-stream audio alphabet without expanding every frame into eight serial Transformer positions.

### 8.4 The delay pattern

Parallel output creates a dependency problem. In RVQ, later codebooks refine earlier ones, but eight codes sampled simultaneously cannot see one another. A MusicGen-style delay pattern staggers codebook $c$ by $c$ autoregressive rows.

For three codebooks and three true audio frames:

| AR row | Codebook 0 | Codebook 1 | Codebook 2 |
|---:|---|---|---|
| 0 | $z_0^0$ | BOC | BOC |
| 1 | $z_1^0$ | $z_0^1$ | BOC |
| 2 | $z_2^0$ | $z_1^1$ | $z_0^2$ |
| 3 | EOC | $z_2^1$ | $z_1^2$ |
| 4 | EOC | EOC | $z_2^2$ |

When the model predicts $z_0^2$, the history already contains $z_0^0$ and $z_0^1$. Tokens in one AR row are still produced in parallel, while tokens belonging to one real codec frame appear on different diagonals. After generation, reverse delay realigns the matrix to $[T,8]$.

Higgs reserves IDs 1,024 and 1,025 for beginning / end boundary codes. The lowest stream is not delayed; the eighth stream is delayed by seven rows. After the short pipeline fill, the model still advances at roughly one AR step per 40 ms frame rather than eight serial steps per frame.

### 8.5 Text, reference audio, and voice cloning

A simplified zero-shot prompt is:

```text
<tts>
<text> Have a nice day.
<audio> [generation begins here]
```

Voice cloning adds a paired reference:

```text
<reference_text> Hey, Adam here ...
<reference_audio> [codec embeddings for the reference waveform]
<text> Have a nice day.
<audio> [generate new codec codes]
```

The full reference code sequence can carry timbre, pitch range, speed, accent, rhythm, emotion, and recording conditions. This is richer than compressing the speaker into one fixed embedding, but it consumes context and prefill compute and can also copy unwanted room noise.

Supplying the reference transcript helps the model separate **what was said** from **how it sounded**; the [official serving recipe](https://huggingface.co/bosonai/higgs-tts-3-4b#voice-cloning) explicitly recommends the pair.

Voice cloning is therefore a form of in-context learning:

$$
P(\text{new audio codes}\mid
\text{reference text, reference audio codes, target text}).
$$

### 8.6 Inline control tokens

Higgs accepts special text-side tags such as:

```text
<|emotion:amusement|><|prosody:expressive_high|>
Wait, that was hilarious. <|sfx:laughter|>Hehe
```

These tags change the probability of the audio codes during generation. They are not post-processing filters applied to a finished waveform. Consequently they can affect duration, pitch contour, energy, rhythm, voice quality, and the presence of a non-verbal event.

Control accuracy still depends on the labeled relationships learned during training. The public release documents the control vocabulary and inference architecture, but not every detail of the data construction and training recipe.

### 8.7 One complete inference trace

For a target sentence with optional reference audio:

1. Tokenize target text and control tags into text IDs.
2. Encode the reference waveform, if present, into $[T_{\text{ref}},8]$ codec IDs.
3. Apply the delay pattern to the reference codes and fuse their embeddings with the text context.
4. Prefill the Qwen3 KV cache.
5. Project the final hidden state to $[8,1026]$ logits.
6. Sample eight code streams and feed their fused embedding back for the next AR row.
7. Repeat until the end codes complete all streams.
8. Reverse the delay pattern to recover aligned $[T,8]$ codes.
9. Look up and sum the RVQ vectors.
10. Decode the latent sequence into a 24 kHz waveform.
11. In streaming mode, emit waveform windows once the codec decoder has enough context rather than waiting for the complete utterance.

The high-level training decomposition is similarly clean:

$$
\underbrace{\text{waveform}\rightarrow\text{codes}\rightarrow\hat{\text{waveform}}}_{\text{train the codec}}
\qquad
\underbrace{\text{text / context}\rightarrow\text{codes}}_{\text{train the autoregressive model}}.
$$

With teacher forcing, the audio-model loss is a sum of categorical cross-entropies across rows and codebooks:

$$
\mathcal{L}_{\text{audio}}
=-
\sum_t\sum_{c=1}^{8}
\log P\!\left(z_t^c\mid\text{text},z_{<t}^{1:8}\right).
$$

This equation describes the exposed architecture; it should not be mistaken for a complete, officially published Higgs training recipe.

## 9. Why codec tokens fit LLM infrastructure

Once audio is discrete, much of the mature language-model stack transfers directly:

- decoder-only causal attention;
- next-token cross-entropy;
- prompt conditioning and in-context learning;
- sampling controls;
- KV caching and prefix reuse;
- continuous batching; and
- streaming generation.

The decisive simplification is the division of labor:

$$
\boxed{\text{The language model decides what should happen acoustically.}}
$$

$$
\boxed{\text{The codec decoder renders that decision into waveform samples.}}
$$

The trade-off is equally important. Autoregressive sampling can repeat, omit, or drift; temperature trades stability for expressiveness; long-form speech needs segmentation and style continuity; and the codec introduces irreversible quantization loss.

## 10. From TTS to spoken dialogue

A production voice assistant can be cascaded:

```text
microphone → ASR → text LLM → TTS → speaker
```

or end to end:

```text
microphone → speech-native model → speech
```

The cascade is easier to inspect, moderate, cache, and replace component by component. End-to-end speech models can preserve prosody and reduce modality hand-offs, but are harder to control and evaluate.

Full duplex adds another axis. A useful system must listen while speaking, distinguish the user's interruption from its own echo, stop or revise output quickly, and produce backchannels without stealing the turn. These are scheduling and interaction problems as much as synthesis problems; chapter 23 returns to them through Thinker–Talker architectures.

For streaming systems, do not report “latency” as one number. Separate:

- time to first audio;
- autoregressive model step time;
- codec / vocoder window delay;
- real-time factor (generation time divided by audio duration); and
- end-to-end interruption response time.

## 11. Music and general sound

The same representation choices extend beyond speech. Codec-token LMs naturally mix speech, music, ambience, and sound events in one alphabet; continuous diffusion / flow models remain strong for globally structured audio. Text-to-music systems must additionally model minute-scale form, rhythm, harmony, vocals, and stereo production. Video-to-audio systems add temporal alignment: a footstep must sound at the frame where the foot lands.

The important boundary is no longer simply TTS versus music. It is whether a model's tokenizer, context window, training data, and generation objective preserve the semantic and acoustic structure required by the task.

## 12. Evaluation: every metric sees only one slice

| Axis | Typical measure | Blind spot |
|---|---|---|
| Intelligibility | WER / CER from an external ASR model | A clear but unnatural voice may score well |
| Naturalness | Human MOS or pairwise preference | Expensive; protocol and listeners matter |
| Speaker similarity | Speaker-embedding cosine similarity; human judgment | Can reward imitation of noise or channel artifacts |
| Prosody / emotion | Labeled preference or task-specific classifier | Labels are culturally and context dependent |
| Codec fidelity | Reconstruction tests, perceptual audio metrics | Does not measure text-conditioned generation |
| Serving | Time to first audio, RTF, throughput at fixed concurrency | Hardware, batch size, and utterance length can dominate |
| Safety | Consent, impersonation resistance, watermark / disclosure checks | Cannot be reduced to audio quality |

Evaluate the codec and the generative model separately. First compare $x$ with codec reconstruction $\hat{x}$; then evaluate whether text → predicted codes produces the intended content, speaker, and style. Otherwise a single bad sample cannot tell you whether the failure came from compression or generation.

## 13. How to choose an audio-generation architecture

Model names change quickly; the underlying design choices are more stable. Start from the task and representation rather than a leaderboard.

| Requirement | Architecture to evaluate first | Why | Measure before committing |
|---|---|---|---|
| CPU or edge narration | Compact mel / vocoder TTS such as Kokoro | Small footprint and simple serving | RTF on the target CPU, pronunciation, voice coverage |
| Zero-shot voice cloning | One codec LM and one flow-matching system | Both major families can clone through different mechanisms | Consent workflow, speaker similarity, transcript fidelity, noise copying |
| Real-time voice agent | Streaming codec LM or low-latency hybrid | Incremental codes naturally feed waveform windows | Time to first audio, interruption latency, RTF at realistic concurrency |
| Audiobook or batch synthesis | Flow model or stable compact TTS | Parallel generation and long-form stability may matter more than first-chunk latency | Paragraph continuity, drift, throughput, editing cost |
| Expressive dialogue and sound events | Audio LM with explicit controls | Speech, laughter, breaths, and style tokens can share one context | Control success, intelligibility, repetition / omission rate |
| Speech editing / infilling | Flow-matching or masked acoustic model | Conditioning on known audio on both sides is natural | Boundary smoothness and preservation outside the edit |

A useful teaching progression is [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) → [F5-TTS](https://github.com/SWivid/F5-TTS) → [Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b). It moves from a compact working synthesizer, to a continuous non-autoregressive route, to a discrete autoregressive audio language model. License and language claims are deployment inputs, not architectural properties; verify each model card at the time of use.

## 14. Hands-on path

The existing [Kokoro notebook](notebooks/01_kokoro_tts.ipynb) provides the smallest local synthesis loop and a TTS → ASR round-trip test. The next useful practical sequence is to inspect codec reconstruction and RVQ, run Higgs voice cloning and inline controls, then measure a complete ASR → LLM → TTS voice pipeline.

## 15. What to remember

1. A codec is the complete encoder–quantizer–decoder; a codebook is a learned vector dictionary; an audio code is an integer index into that dictionary.
2. RVQ stores one frame as several successive corrections. Later codebooks refine residual error rather than owning one interpretable attribute.
3. Higgs compresses 24 kHz audio to 25 frames/s with eight code streams and uses a Qwen3-style model to predict delayed code rows.
4. Reverse delay restores aligned codec frames; the codec decoder then upsamples the latents into waveform samples.
5. VALL-E made neural-codec language modeling prominent for TTS. Higgs is a polished, controllable, streaming-oriented descendant—not the origin.
6. Codec LMs are one major modern route; continuous diffusion / flow-matching TTS is the other.

## 16. Key papers and technical references

| Work | Year | Why read it |
|---|---:|---|
| [SoundStream](https://arxiv.org/abs/2107.03312) | 2021 | Convolutional neural codec + RVQ foundation |
| [AudioLM](https://arxiv.org/abs/2209.03143) | 2022 | Audio generation as language modeling over discrete tokens |
| [EnCodec](https://arxiv.org/abs/2210.13438) | 2022 | Practical high-fidelity neural audio compression |
| [VALL-E](https://arxiv.org/abs/2301.02111) | 2023 | Conditional codec LM and 3-second zero-shot TTS |
| [Voicebox](https://arxiv.org/abs/2306.15687) | 2023 | Non-autoregressive flow matching for speech generation and editing |
| [F5-TTS](https://aclanthology.org/2025.acl-long.313/) | 2025 | Minimal, fully non-autoregressive flow-matching TTS |
| [Higgs TTS 3 model card](https://huggingface.co/bosonai/higgs-tts-3-4b) | 2026 | Exact multi-codebook architecture, controls, cloning, and serving examples |
| [Higgs Audio V2 tokenizer docs](https://huggingface.co/docs/transformers/model_doc/higgs_audio_v2_tokenizer) | 2026 | The semantic + acoustic codec used by the Higgs family |
