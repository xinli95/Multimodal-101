# 06 · Audio — Speech & Sound

On the understanding side, ASR with the Whisper architecture as the through-line; on the generation side, the codec-token LM paradigm for TTS — which is also the prerequisite for understanding how the omni models of chapter 07 "speak". Practice covers Whisper, Kokoro, and Higgs Audio v3.

## Learning goals

- Understand the two ASR families: encoder-decoder (Whisper) vs. CTC/transducer (streaming)
- Understand neural audio codecs (the EnCodec family): how audio becomes discrete tokens
- Understand that modern TTS = "a language model over codec tokens", and how zero-shot voice cloning works
- Build a local STT + TTS loop into a working voice-chat pipeline

## Contents

- [theory.md](theory.md) — principles and key papers
- [landscape.md](landscape.md) — current state of play (with last-verified date)
- [notebooks](notebooks/README.md) — hands-on code
