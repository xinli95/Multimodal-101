# 22 · Audio Generation — Speech and Sound

**How this connects to Part I.** Chapter 05 took audio *in*: a waveform became mel features, then soft tokens, then part of a prompt. This chapter runs the arrow the other way — text in, waveform out — and the symmetry is closer than the image case. Modern TTS is a language model over **codec tokens**: discretise audio with a neural codec, then predict the next audio token exactly as an LLM predicts the next word. Gemma 4's audio tower and a codec are two answers to the same question (how do you turn continuous sound into a finite alphabet), tuned for opposite purposes: one for understanding, one for reconstruction.

ASR itself has moved to [chapter 05](../05-audio-and-video/index.md), where Whisper serves as the design-space comparison against Gemma 4's own audio encoder. What remains here is generation.

## Learning goals

- Understand neural audio codecs (the EnCodec family): how audio becomes discrete tokens, and what RVQ is doing
- Understand that modern TTS = "a language model over codec tokens", and how zero-shot voice cloning falls out of that
- Understand what a codec optimised for *reconstruction* keeps that an encoder optimised for *understanding* throws away
- Build a local STT + TTS loop into a working voice-chat pipeline, and measure where the latency actually goes

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
