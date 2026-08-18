# Theory · Audio Generation

## 1. Audio representations

- Waveform → mel spectrogram (standard input on the understanding side)
- **Neural codecs**: EnCodec / SoundStream / DAC — RVQ (residual vector quantization) turns audio into multi-layer discrete tokens; ~1.5kbps still sounds faithful
- The codec is the "VAE of audio": the entire generation stack is built on top of it

## 2. The understanding side, in one paragraph

Speech *recognition* is covered in [chapter 05](../05-audio-and-video/index.md), where Gemma 4's own chunked-attention audio encoder is the subject and Whisper is the design-space comparison; the model list is in the [understanding landscape](../landscape.md). Two facts from there matter here. Whisper's robustness came from 680k hours of weak supervision rather than from architecture — data, not design. And ASR is being absorbed into general models: Gemma 4 E2B hears natively, and Qwen3.5-Omni recognises 113 languages, so "run an ASR model" is increasingly a latency decision rather than a capability one.

What does *not* transfer is the representation. An encoder built for understanding throws away everything not needed to identify words; a codec built for generation must keep enough to reconstruct the waveform. Same input, opposite objectives — which is why §1's codecs look nothing like chapter 05's mel front end.

## 3. TTS: paradigm evolution

- Old paradigm: acoustic model + vocoder (Tacotron + HiFi-GAN), per-speaker training
- **Modern paradigm: codec LMs** (opened by VALL-E, 2023):
  - text tokens + reference-audio tokens → autoregressively generate codec tokens → codec decodes to waveform
  - zero-shot cloning = in-context learning with the reference audio as the prompt
- Variants: flow-matching acoustic routes (non-autoregressive, fast), hybrids
- The capability split: streaming low latency (conversational agents) vs. expressiveness (emotion/prosody control via inline tokens)

## 4. Spoken dialogue and full duplex

- Cascaded (ASR→LLM→TTS) vs. end-to-end speech dialogue (GPT-4o style)
- Full duplex: interruptions, backchannels — why this is hard (echoed by the Thinker-Talker architecture in chapter 23)

## 5. Music and sound effects

- Music generation: Suno/Udio (closed), YuE, ACE-Step (open)
- SFX/Foley: video-to-audio, echoing chapter 21's joint audio-video generation

## 6. Evaluation

- ASR: WER/CER (LibriSpeech, Common Voice, multilingual FLEURS)
- TTS: EmergentTTS-Eval, Audio Turing Test, speaker similarity + naturalness MOS

## Key papers

| Paper | Year | Why read it |
|---|---|---|
| Whisper | 2022 | Data-driven robust ASR |
| EnCodec | 2022 | The cornerstone of audio discretization |
| VALL-E | 2023 | Opened the codec-LM paradigm |
| Higgs Audio v3 technical material | 2026 | The latest chat-native, streaming TTS practice |
