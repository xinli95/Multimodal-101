# Landscape · Audio Generation

> **Last verified: 2026-08-02** — re-check if this is more than 6 months old.

> ASR models live in the [understanding landscape](../landscape.md) alongside Gemma 4's audio tower; this page is the generation side.

## TTS (open / open-weight)

| Model | Org | Size | License | Highlights |
|---|---|---|---|---|
| **Kokoro-82M** | community | 82M | Apache 2.0 | Runs on CPU, top of TTS Arena — **the teaching pick** |
| **Higgs Audio v3** | Boson AI | 4B | ⚠️ research/non-commercial (commercial by separate license) | Released 2026-06; chat-native, streaming (speaks mid-sentence), 100+ language cloning |
| **Fish Audio S2 Pro** | Fish Audio | - | open weights (check current terms) | #1 on EmergentTTS-Eval, above ElevenLabs |
| Chatterbox-Turbo | Resemble | - | MIT | Production-grade low latency |
| Orpheus / Dia2 | Canopy / Nari | 3B / - | Apache 2.0 | Emotion control / dialogue-style |

## Closed

- ElevenLabs (broadest commercial ecosystem), OpenAI TTS, Google/Gemini TTS, MiniMax-Speech, Seed-TTS (ByteDance)

## Music generation

- Closed: Suno, Udio; open: YuE (vocals + accompaniment), ACE-Step

## Quick chooser

| Scenario | Pick |
|---|---|
| Free local quick start | Whisper + Kokoro (all 🟢 CPU) |
| Voice agents (streaming, interruption) | Higgs Audio v3 (mind the commercial license) / Chatterbox-Turbo |
| Best multilingual cloning quality | Fish Audio S2 Pro / ElevenLabs |

## Where things are heading

1. Open TTS now beats some closed incumbents in blind tests — audio is the modality where open source has caught up most completely.
2. "TTS models" are becoming "LLMs that can talk" (chat-native, e.g. Higgs v3 with paralinguistic control and context-aware tone).
3. Standalone ASR/TTS will eventually be absorbed into omni models, but cascaded stacks still dominate production for controllability.
