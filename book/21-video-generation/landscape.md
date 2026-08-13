# Landscape · Video Generation

> **Last verified: 2026-08-02** — this area reshuffles fastest; re-check after 3 months.

## Open / open-weight

| Model | Org | License | Highlights |
|---|---|---|---|
| **Wan 2.2** | Alibaba | Apache 2.0 | The cleanest license among high-quality options; ⚠️ Wan 2.5/2.6 are **API-only, not open**, and 2.7's weights/license status is murky |
| **HunyuanVideo 1.5** | Tencent | open weights | 8.3B lightweight; runs on consumer GPUs |
| **LTX-2.3** | Lightricks | tiered commercial (free under $10M ARR) | The only open option with native joint audio-video; 4K |
| Mochi / CogVideoX | Genmo / Zhipu | Apache 2.0 | Previous generation; good teaching references |

## Closed frontier

| Model | Org | Highlights |
|---|---|---|
| **Veo 3.1** | Google | Overall quality leader: native 48kHz synced audio + 4K (upgraded 2026-01) |
| **Seedance 2.5** | ByteDance | Released 2026-06: 30s single pass, 50 multimodal reference inputs, 3D previz |
| **Kling 3.0** | Kuaishou | Released 2026-02: phoneme-level multi-character lip sync, motion control, price/performance |

⚠️ Retired: **Sora 2 consumer product shut down 2026-04; API closes 2026-09-24** — do not build tutorials on the Sora API.

## Quick chooser

| Scenario | Pick |
|---|---|
| Local experiments / teaching | Wan 2.2 (🔴 high VRAM) or HunyuanVideo 1.5 (🟡 barely) |
| Commercial on-prem | Wan 2.2 (Apache 2.0) |
| Highest-quality hero clips | Veo 3.1 API |
| Long narrative / heavy reference control | Seedance 2.5 |
| Production consensus | Multi-model routing: switch between 2–3 models by shot type |

## Where things are heading

1. Open video still trails closed by roughly a generation, and leaders show a "new versions go closed" tendency (Wan 2.5+).
2. Native joint audio-video generation is the new baseline.
3. Real-time / interactive generation (the world-model direction) is the next frontier.
