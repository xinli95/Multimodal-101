# 05 · Audio and Video — 另外两个前端

**在数据流中的位置**：`audio ──► 特征提取 ──► Gemma4AudioModel ──► soft tokens`，以及 `video ──► 抽帧 ──► Gemma4VisionModel`

两个模态，新增机件的量却完全不同。视频完全复用视觉塔——视频就是帧，帧就是图像，唯一的新部件是决定抽**哪些**帧的采样器。音频则拿到了一座专门设计的编码器，长得跟视觉塔毫无关系：卷积下采样、相对位置编码、带显式左上下文的分块注意力。这个不对称就是本章的主题。

音频**仅 E2B / E4B 原生支持**。31B 和 26B-A4B 的 checkpoint 里没有音频塔——`Gemma4Model.__init__` 在 `audio_config is None` 时直接跳过（01 章）。

## 你会学到

1. 波形如何变成模型输入：16kHz 下 128 个 mel 频带，再经堆叠卷积投影做 4× 时间下采样——"每个音频 token 40ms"正是从这里来的
2. 分块注意力：`attention_chunk_size=12`、`attention_context_left=13`、`attention_context_right=0`，以及流式友好的编码器为什么长这样而不用全注意力
3. Conformer 家族的积木——因果 1D 卷积、轻量卷积模块、相对位置编码、logit 截断——以及语音编码器为什么保留了视觉早已抛弃的卷积
4. 视频采样怎么工作，`do_sample_frames` / `num_frames` / `fps` 到底控制什么，一段视频真实的 token 成本是多少
5. **设计空间**：Gemma 4 音频塔 vs Whisper 的 encoder-decoder vs CTC/transducer 流式 ASR——通用多模态模型相对专用 ASR 让出了什么

## 源码地图

| 文件 | 符号 | 作用 |
|---|---|---|
| `feature_extraction_gemma4.py` | `Gemma4AudioFeatureExtractor` | 波形 → 16kHz 下 128 维 mel 特征 |
| `video_processing_gemma4.py` | `Gemma4VideoProcessor` | 抽帧与逐帧预处理 |
| `modeling_gemma4.py` | `Gemma4AudioSubSampleConvProjection` | 4× 时间下采样 |
| | `Gemma4AudioAttention` | 带左上下文的分块注意力 |
| | `Gemma4AudioCausalConv1d`、`Gemma4AudioLightConv1d` | Conformer 式积木 |
| | `Gemma4AudioModel` | 组装好的音频塔与它的 mask 管线 |

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_audio_tower_anatomy.ipynb` | 追踪一段波形经 mel 提取与卷积下采样的形状变化；手写重建分块注意力 mask 并与库 `assert_close` | 🟢 CPU |
| `02_asr_and_video.ipynb` | 真权重 E2B：转写、音频问答、视频问答——视频的 token 成本是测出来的，不是猜的 | 🟡 24GB 显存 |
| <a href="../../05-audio-and-video/notebooks/03_whisper_asr_compare.html"><code>03_whisper_asr_compare.ipynb</code></a> | 设计空间对照组：同一段音频交给 Whisper。专用模型强在哪，又做不到什么 | 🟢 CPU / 🟡 |

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/05-audio-and-video/index.html)。
