# Theory · Speech & Audio

## 1. Audio representations

- Waveform → mel spectrogram (standard input on the understanding side)
- **Neural codecs**: EnCodec / SoundStream / DAC — RVQ (residual vector quantization) turns audio into multi-layer discrete tokens; ~1.5kbps still sounds faithful
- The codec is the "VAE of audio": the entire generation stack is built on top of it

## 2. ASR (speech recognition)

- **Whisper (2022)**: encoder-decoder + 680k hours of weak supervision; robustness comes from data, not architecture
- Streaming families: CTC / RNN-T / TDT (NVIDIA Parakeet) — required for low-latency scenarios
- Distillation and speedups: Distil-Whisper, faster-whisper (CTranslate2)
- New trend: ASR being absorbed into omni models (Qwen3.5-Omni recognizes 113 languages)

## 3. TTS: paradigm evolution

- Old paradigm: acoustic model + vocoder (Tacotron + HiFi-GAN), per-speaker training
- **Modern paradigm: codec LMs** (opened by VALL-E, 2023):
  - text tokens + reference-audio tokens → autoregressively generate codec tokens → codec decodes to waveform
  - zero-shot cloning = in-context learning with the reference audio as the prompt
- Variants: flow-matching acoustic routes (non-autoregressive, fast), hybrids
- The capability split: streaming low latency (conversational agents) vs. expressiveness (emotion/prosody control via inline tokens)

## 4. Spoken dialogue and full duplex

- Cascaded (ASR→LLM→TTS) vs. end-to-end speech dialogue (GPT-4o style)
- Full duplex: interruptions, backchannels — why this is hard (echoed by the Thinker-Talker architecture in chapter 07)

## 5. Music and sound effects

- Music generation: Suno/Udio (closed), YuE, ACE-Step (open)
- SFX/Foley: video-to-audio, echoing chapter 05's joint audio-video generation

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
