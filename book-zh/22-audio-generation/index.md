# 22 · Audio Generation — 从波形到 Audio Language Model

一段 24 kHz 音频每秒包含 24,000 个振幅值。语言模型不可能为了生成一秒语音就做 24,000 次自回归决策，所以音频生成的第一步不是换一个更大的 Transformer，而是先换一种表示问题的方式。

本章沿着一条连续路径展开：

```text
waveform
  → 紧凑的声学表示
  → generative model
  → 重建 waveform
```

我们会从数字声音本身出发，从零搭起 neural codec 的直觉，推导 residual vector quantization，再比较 codec language model 与 flow matching，最后把 Higgs TTS 3 从文本和参考语音一路追踪到 8 路 code stream 和可播放的波形。目标不是记住一串模型名，而是以后看到任何新的 audio generator，都能判断它预测什么、哪里必须串行、最后由谁把表示渲染成声音。

[第 05 章](../05-audio-and-video/index.md)追踪音频怎样进入模型完成理解；这一章把箭头反过来。理解侧 encoder 只要能辨认文字，就可以丢掉说话人和录音通道的细节；生成侧 codec 必须保留足够多的细节，才能重建可信的声音。

## 1. TTS 模型究竟要决定什么？

对于“会议九点开始”这句话，合成器要生成的不只是正确的文字内容，还要决定：

- 每个字如何发音、持续多久；
- 在哪里停顿，节奏如何；
- 音高曲线与能量变化；
- 说话人身份和声音质感；
- 情绪、说话风格与录音环境；
- 呼吸、笑声、咳嗽等非语言事件。

所以真正的问题是

$$
P(\text{audio}\mid\text{text, speaker, style, context}).
$$

难点是输出太密集。10 秒 24 kHz 单声道音频包含 240,000 个连续采样值。逐个预测会产生极长的序列，还会迫使同一个模型同时学习高层语音规划和底层波形渲染。

现代系统会在中间引入一种声学表示。选哪种表示，几乎决定了后面采用哪种架构。

## 2. 音频表示的阶梯

| 表示 | 一秒的大致形态 | 适合做什么 | 可能丢掉什么 |
|---|---:|---|---|
| Waveform | 24 kHz 下 24,000 个采样点 | 播放；完整保留数字信号 | 定义上没有丢失，但过于密集 |
| Mel spectrogram | 数十到数百个连续 frame | 语音识别；经典 TTS 与 vocoder | 相位和一部分细粒度波形信息 |
| Continuous latent | 模型自定义的低帧率向量 | Diffusion / flow 生成；学习式压缩 | 训练目标认为不重要的信息 |
| Discrete codec token | 低帧率整数 ID，通常每帧多个 | 语言建模、prompt、存储、流式生成 | 量化误差与 codec 重建误差 |

第 05 章为了理解音频使用 mel 特征。语音识别器追求“不变性”：不同人、不同麦克风说同一句话，最终都应该映射成同一串文字。生成 codec 追求“可重建性”：麦克风质感、音高、音色和背景声都可能需要保留。

因此，理解侧 encoder 和生成侧 codec 虽然都在压缩波形，却不能互换。

## 3. Codec、codebook 与 audio code

这三个词很像，但指的是三个不同层次的对象。

**Codec** 来自 coder + decoder。Neural audio codec 是一个学习出来的有损压缩系统：

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

- $x$：输入 waveform；
- $h$：低帧率连续 latent 序列；
- $z$：离散整数 ID；
- $\hat{x}$：重建 waveform。

**Codebook** 是模型学出来的一张向量字典。如果它包含 1,024 个 64 维向量，那么

$$
C\in\mathbb{R}^{1024\times64}.
$$

Quantizer 用最接近的 codebook entry 代替连续向量：

$$
z=\arg\min_k\lVert h-C_k\rVert^2.
$$

**Audio code** 是被选中 entry 的整数 ID。如果选中第 137 行，保存或预测的 code 就是 `137`。它不是一小段 WAV，也不一定对应某个音素；它只是 latent space 里一个学习到的原型向量的编号。

与文本的类比很有用，但并不完全等价：

| 文本模型 | Codec 模型 |
|---|---|
| Vocabulary | Codebook |
| Text token ID | Audio code ID |
| Token embedding | Codebook vector |
| 一个 token 往往能看到可解释的拼写 | 一个 audio code 单独拿出来通常没有人类可解释含义 |

## 4. 为什么需要多个 codebook？残差向量量化

只有一个包含 1,024 个向量的 codebook，通常不足以精细表示复杂声音。Residual Vector Quantization（RVQ）会分多轮逼近一个 latent vector。

第一层 codebook 选择 $q_1$ 粗略逼近 $h$，下一层量化剩余误差：

$$
r_1=h-q_1,\qquad q_2\approx r_1.
$$

继续迭代，最后得到

$$
h\approx q_1+q_2+\cdots+q_K.
$$

前面的 codebook 建立粗略轮廓，后面的 codebook 修正残差。这是一种“逐级逼近”，不要机械理解成“第一层只管发音、第二层只管音色”。这些因素在不同层里仍可能纠缠在一起。

一个二维玩具例子可以把机制讲清楚：

$$
h=[0.72,-0.10],\quad
q_1=[0.60,-0.20],\quad
r_1=[0.12,0.10].
$$

如果第二个 codebook 选出 $q_2=[0.10,0.08]$，那么

$$
q_1+q_2=[0.70,-0.12],
$$

已经很接近 $h$。Codec 最终只需存两个 ID，而不必保存原始浮点向量。

[SoundStream](https://arxiv.org/abs/2107.03312) 奠定了今天熟悉的“全卷积 encoder / decoder + RVQ”设计；[EnCodec](https://arxiv.org/abs/2210.13438) 则把同一家族发展成实用的高保真 neural codec。

### 一个有用的码率计算

Higgs 用 8 个 codebook 表示每个 40 ms frame，每个 codebook 有 1,024 个真实 entry。暂时忽略封装开销：

$$
25\ \text{frames/s}
\times 8\ \text{codes/frame}
\times \log_2(1024)
=2{,}000\ \text{bits/s}.
$$

而 24 kHz、16-bit 单声道 PCM 是

$$
24{,}000\times16=384{,}000\ \text{bits/s}.
$$

所以 token payload 大约小 192 倍。这个数字是帮助建立直觉的估算，不是文件格式 benchmark：真实传输还需要 metadata，entropy coding 也会改变最终码率。

## 5. Audio codes 怎样还原成 waveform？

假设对齐后的一个 Higgs frame 是

```text
[137, 921, 44, 818, 31, 402, 9, 615]
```

Codec decoder 在概念上做三件事：

1. 每个 ID 去各自的 RVQ codebook 中取回向量。
2. 将 8 个向量相加，重建量化后的 latent $\hat{h}_t$。
3. 让低帧率 latent sequence 经过学习到的卷积上采样和 residual block，输出 waveform samples。

帧率为 25 fps、采样率为 24 kHz 时，时间维需要扩展

$$
24{,}000/25=960
$$

倍。Decoder 不是把一个 latent 复制 960 次；它从 latent–waveform 配对数据中学会了怎样渲染音高周期、瞬态、相位连续性、音色和噪声。

这是有损重建：

$$
\hat{x}\ne x,
\qquad
\operatorname{PerceptualSimilarity}(\hat{x},x)\ \text{应当很高}.
$$

Decoder 权重本身包含 audio prior，因此能补出合理的高频细节，就像图像 decoder 能把压缩 latent 展开成数百万像素一样。但 codec 的重建质量也构成生成模型的音质上限：即使 audio code 预测完全正确，也无法恢复 codec 已经丢掉的信息。

### Codec decoder、vocoder 与硬件 DAC

| 名称 | 输入 | 输出 |
|---|---|---|
| 传统 neural vocoder | 通常是 mel spectrogram | Waveform |
| Neural codec decoder | 量化 codec latent / audio code | Waveform |
| 硬件 DAC | 数字 waveform samples | 驱动扬声器的模拟电压 |

“DAC”也可能指 Descript Audio Codec 模型家族。这个 neural model 与声卡里的 digital-to-analog converter 不是一回事。

## 6. 现代 TTS 的两条主要路线

把现代 TTS 简写成“跑在 codec token 上的 LLM”很诱人，但这样会抹掉另一个同样重要的家族。

| | Discrete codec language model | Diffusion / flow-matching TTS |
|---|---|---|
| 典型路径 | text → discrete audio codes → codec decoder | text + reference → continuous mel / latent → vocoder 或 decoder |
| 生成方式 | 通常按 codec frame 自回归 | 通常对完整声学序列并行建模，再运行若干 solver step |
| 天然适配 | Decoder-only Transformer、next-token loss、prompt、KV cache | 连续声学建模、editing、infilling、并行合成 |
| 优势 | 流式；in-context 声音 prompt；副语言事件；text/audio 统一 | 无离散量化瓶颈；无 AR 误差累积；批量生成快 |
| 代价 | Codec 音质上限；sampling 漂移；多 codebook 依赖；串行 step | 迭代 denoising / ODE 求解；时长处理；streaming 不够天然 |
| 代表 | VALL-E、Bark、VoiceCraft、Fish Speech、Higgs | Voicebox、E2-TTS、F5-TTS、Matcha-TTS |

[F5-TTS](https://aclanthology.org/2025.acl-long.313/) 是反驳“所有 LLM TTS 都是 codec TTS”的好例子：它使用 flow matching 和 Diffusion Transformer，以非自回归方式生成连续 mel 表示。

不少系统其实是 hybrid：LLM 负责语义规划或粗粒度 audio token，flow model 或 neural decoder 再补充声学细节。真正的问题不是哪条路线永远胜出，而是系统把“离散瓶颈”和“串行循环”放在了哪里。

## 7. Codec-LM 路线从哪里来？

Higgs 是这条路线成熟阶段的代表，不是发明者。

| 年份 | 工作 | 在谱系中的贡献 |
|---:|---|---|
| 2021 | [SoundStream](https://arxiv.org/abs/2107.03312) | 端到端 neural audio codec：卷积 encoder / decoder + RVQ |
| 2022 | [AudioLM](https://arxiv.org/abs/2209.03143) | 把通用音频生成改写成离散 semantic / acoustic token 上的语言建模 |
| 2022 | [EnCodec](https://arxiv.org/abs/2210.13438) | 面向语音、音乐和通用声音的高保真、实时 neural compression |
| 2023 | [VALL-E](https://arxiv.org/abs/2301.02111) | 把 TTS 改写为 conditional neural-codec language modeling，并展示 3 秒 zero-shot voice prompt |
| 2023 | [Bark](https://github.com/suno-ai/bark) | 通过社区可用代码推广包含语音、非语言声音在内的 expressive text-to-audio |
| 2026 | [Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b) | 面向多语言对话语音的 Qwen3、低帧率、并行多 codebook 实现 |

AudioLM 主要是 audio continuation，并不是以 transcript 为条件的纯 TTS。VALL-E 才明确写出了 TTS 形式：

$$
P(\text{codec codes}\mid\text{phonemes, acoustic prompt}).
$$

因此归功时应当区分：SoundStream 和 EnCodec 提供压缩基础；AudioLM 建立 audio-token language modeling；VALL-E 让 codec-LM TTS 进入主流视野；Higgs 把成熟范式整理成一个可控制、可流式部署的 voice model。

## 8. 把 Higgs TTS 3 当作完整案例

最短而准确的描述是：

> Higgs TTS 3 是一个 Qwen3 风格的自回归模型。它读取交错的文本与音频上下文，预测经过 delay 的离散 codec codes，再由独立 neural codec 将 codes 转成 24 kHz waveform。

```text
目标文本 + 控制 token ───────────────────────────────────┐
                                                        v
参考音频 → Higgs audio tokenizer → 参考 codes → Qwen3 AR decoder
                                                          │
                                                     8 路 code stream
                                                          │
                                                    reverse delay
                                                          │
                                                    codec decoder
                                                          │
                                                    24 kHz waveform
```

[官方模型卡](https://huggingface.co/bosonai/higgs-tts-3-4b)和[模型配置](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/config.json)公开了关键尺寸：

| 组件 | Higgs TTS 3 |
|---|---|
| Backbone | Qwen3 风格 decoder-only Transformer，约 4B |
| Transformer | 36 层，hidden size 2,560，32 个 query head / 8 个 KV head |
| 训练序列长度 | 8,192 tokens |
| Codec | 8 个 codebook，每路 vocabulary 1,026，包含边界 code |
| Codec 帧率 | 25 fps = 40 ms/frame |
| Waveform 采样率 | 24 kHz |

### 8.1 Audio tokenizer

Tokenizer 解决的是

$$
\text{waveform}
\leftrightarrow
[T,8]\ \text{codec IDs}.
$$

Higgs 复用了 Higgs Audio V2 tokenizer 家族。公开设计把基于 DAC 的 acoustic configuration 与基于 HuBERT 的 semantic configuration 结合，再量化学习到的表示。这样做很重要，因为服务于 audio language model 的 codec 必须平衡两个目标：

- 保留足够声学细节，让 decoder 能重建可信声音；
- 暴露足够语言结构，让“预测下一个 code”成为可学的问题。

[Transformers tokenizer 文档](https://huggingface.co/docs/transformers/model_doc/higgs_audio_v2_tokenizer)记录了 24 kHz 采样率、25 fps 设计目标、1,024-entry codebook、64 维 codebook vector，以及 DAC acoustic branch 和 HuBERT semantic branch。

不要混淆 codec codebook vector 与 Qwen input embedding。Codec 拥有用于重建音频的量化空间；语言模型拥有用于在 ID 之上推理的 hidden-space embedding。

### 8.2 Qwen 预测的到底是什么？

普通语言模型预测

$$
P(x_t\mid x_{<t}).
$$

Higgs 进入音频生成阶段后，目标近似为

$$
P\!\left(z_t^{(1)},\ldots,z_t^{(8)}
\mid \text{text, reference audio, previous code rows}\right).
$$

一个 Transformer hidden state $h_t\in\mathbb{R}^{2560}$ 进入 fused audio head，得到 8 个 categorical distribution：

$$
h_t\longrightarrow\text{logits}_t\in\mathbb{R}^{8\times1026}.
$$

8 路可以并行 sampling。同一个 backbone 仍然处理普通 text token 与控制 token；只是在越过 audio boundary 后，主要预测目标切换为 audio codes。

### 8.3 8 个 code 如何进入一个 Transformer position？

概念上，每个 code stream 有自己的 embedding lookup。一个生成 row 的 8 个向量融合（通常可以理解为相加）成单个 2,560 维输入：

$$
e_t=\sum_{c=1}^{8}E_c\!\left(z_t^{(c)}\right).
$$

实现中把 multi-codebook embedding 与 output head 存成 fused tensor，以便高效 lookup 和 projection。这样，一个基本标准的 Qwen3 backbone 就能处理 8 路音频字母表，而不必把每帧展开成 8 个串行 Transformer position。

### 8.4 最容易困惑的 delay pattern

并行输出带来一个依赖问题。RVQ 的后层 codebook 在修正前层，但同一时刻并行 sampled 的 8 个 code 无法互相看见。MusicGen 风格的 delay pattern 会把 codebook $c$ 延迟 $c$ 个 autoregressive row。

只看 3 个 codebook、3 个真实音频 frame：

| AR row | Codebook 0 | Codebook 1 | Codebook 2 |
|---:|---|---|---|
| 0 | $z_0^0$ | BOC | BOC |
| 1 | $z_1^0$ | $z_0^1$ | BOC |
| 2 | $z_2^0$ | $z_1^1$ | $z_0^2$ |
| 3 | EOC | $z_2^1$ | $z_1^2$ |
| 4 | EOC | EOC | $z_2^2$ |

模型预测 $z_0^2$ 时，历史里已经有 $z_0^0$ 和 $z_0^1$。一个 AR row 内的 token 仍然并行生成，但属于同一个真实 codec frame 的 token 被放在不同对角线上。生成结束后，reverse delay 再把矩阵恢复成 $[T,8]$。

Higgs 把 ID 1,024 和 1,025 保留给 begin / end boundary code。第一路不延迟，第八路延迟 7 个 row。短暂填满 pipeline 后，模型依然大致每 40 ms frame 前进一个 AR step，而不是每帧串行运行 8 次。

### 8.5 文本、参考音频与 voice cloning

简化的 zero-shot prompt 是：

```text
<tts>
<text> Have a nice day.
<audio> [从这里开始生成]
```

Voice cloning 再加入一组配对 reference：

```text
<reference_text> Hey, Adam here ...
<reference_audio> [参考波形的 codec embeddings]
<text> Have a nice day.
<audio> [生成新的 codec codes]
```

完整 reference code sequence 可以携带音色、音高范围、语速、口音、节奏、情绪和录音环境。它比把 speaker 压缩成一个固定 embedding 保留的信息更多，但也会占用 context 和 prefill 计算，并可能复制不想要的房间噪声。

同时提供 reference transcript，可以帮助模型区分**说了什么**与**怎样说**；[官方 serving recipe](https://huggingface.co/bosonai/higgs-tts-3-4b#voice-cloning)也明确建议音频与 transcript 成对提供。

因此，voice cloning 可以看作一种 in-context learning：

$$
P(\text{new audio codes}\mid
\text{reference text, reference audio codes, target text}).
$$

### 8.6 Inline control token

Higgs 接受这类特殊 text-side tag：

```text
<|emotion:amusement|><|prosody:expressive_high|>
Wait, that was hilarious. <|sfx:laughter|>Hehe
```

这些 tag 在生成过程中改变 audio code 的概率，而不是对完成的 waveform 做后处理。因此它们能影响 duration、pitch contour、energy、rhythm、voice quality，以及是否出现某个非语言声音事件。

控制是否准确，仍取决于训练时学到的“标签–声音”关系。公开 release 描述了 control vocabulary 和推理架构，但没有公开数据构建与训练 recipe 的每个细节。

### 8.7 一次完整推理发生了什么？

给定目标句子和可选参考音频：

1. 将目标文本与控制 tag tokenize 成 text IDs。
2. 若有参考音频，将波形编码成 $[T_{\text{ref}},8]$ codec IDs。
3. 对 reference codes 应用 delay pattern，并把 fused audio embeddings 与文本上下文放在一起。
4. Prefill Qwen3 的 KV cache。
5. 把最后一个 hidden state 投影成 $[8,1026]$ logits。
6. Sample 8 路 code，再把 fused embedding 作为下一个 AR row 的输入。
7. 重复，直到 end codes 完成所有 stream。
8. Reverse delay，恢复对齐的 $[T,8]$ codes。
9. Lookup 并相加 RVQ vectors。
10. 把 latent sequence 解码成 24 kHz waveform。
11. 流式模式下，只要 codec decoder 积累了足够上下文，就开始输出 waveform window，不必等待整句结束。

高层训练分工也很清楚：

$$
\underbrace{\text{waveform}\rightarrow\text{codes}\rightarrow\hat{\text{waveform}}}_{\text{训练 codec}}
\qquad
\underbrace{\text{text / context}\rightarrow\text{codes}}_{\text{训练自回归模型}}.
$$

使用 teacher forcing 时，audio-model loss 是跨时间 row 与 codebook 的 categorical cross-entropy 之和：

$$
\mathcal{L}_{\text{audio}}
=-
\sum_t\sum_{c=1}^{8}
\log P\!\left(z_t^c\mid\text{text},z_{<t}^{1:8}\right).
$$

这个公式描述的是公开架构所对应的高层训练形式，不应误当成官方完整公开的 Higgs training recipe。

## 9. 为什么 codec token 特别适合 LLM 基础设施？

音频一旦离散化，成熟语言模型栈中的很多能力都能直接迁移：

- decoder-only causal attention；
- next-token cross-entropy；
- prompt conditioning 与 in-context learning；
- sampling control；
- KV cache 与 prefix reuse；
- continuous batching；
- streaming generation。

最关键的简化来自分工：

$$
\boxed{\text{语言模型决定声学上应该发生什么。}}
$$

$$
\boxed{\text{Codec decoder 把这个决定渲染成 waveform samples。}}
$$

代价也同样重要。自回归 sampling 可能重复、漏词或漂移；temperature 需要在稳定性与表现力之间取舍；长语音需要分段并维持 style continuity；codec 还会引入不可逆的量化损失。

## 10. 从 TTS 到语音对话

生产 voice assistant 可以采用级联：

```text
麦克风 → ASR → text LLM → TTS → 扬声器
```

也可以端到端：

```text
麦克风 → speech-native model → speech
```

级联方案更容易检查、审核、缓存，也能逐个替换组件；端到端 speech model 能保留 prosody、减少模态交接，但更难控制和评测。

Full duplex 又增加一个维度。实用系统必须边说边听，区分用户打断与自身回声，快速停止或修改输出，并在不抢话的情况下生成 backchannel。这既是合成问题，也是 scheduling 与 interaction 问题；第 23 章会从 Thinker–Talker 架构再次讨论。

评估流式系统时，不要把“延迟”压成一个数字。至少拆开：

- time to first audio；
- autoregressive model step time；
- codec / vocoder window delay；
- real-time factor（生成耗时 / 音频时长）；
- 端到端 interruption response time。

## 11. 音乐与通用声音

同样的表示选择也适用于语音之外。Codec-token LM 很自然地把语音、音乐、环境声和 sound event 放进一套字母表；continuous diffusion / flow model 则擅长全局结构化音频。Text-to-music 还必须建模分钟级结构、节奏、和声、人声与立体声制作；video-to-audio 又增加时间对齐——脚落地的那一帧必须响起脚步声。

今天最重要的边界已不只是 TTS 与音乐，而是模型的 tokenizer、context window、训练数据和生成目标，是否保留了任务所需的 semantic 与 acoustic structure。

## 12. 评测：每个指标只能看见一个切面

| 维度 | 常见指标 | 盲点 |
|---|---|---|
| 可懂度 | 外部 ASR 的 WER / CER | 清楚但不自然的声音也可能得高分 |
| 自然度 | 人类 MOS 或 pairwise preference | 成本高；协议和听众会影响结果 |
| 说话人相似度 | Speaker-embedding cosine similarity；人类判断 | 可能奖励对噪声或录音通道的模仿 |
| Prosody / emotion | 标注偏好或任务专用 classifier | 标签受文化和上下文影响 |
| Codec fidelity | 重建测试、感知音频指标 | 不评估 text-conditioned generation |
| Serving | 首包时间、RTF、固定并发下 throughput | 硬件、batch size、句长都可能主导结果 |
| Safety | 同意流程、反冒充、水印 / disclosure 检查 | 不能被音质指标替代 |

Codec 与 generative model 要分开评测。先比较 $x$ 与 codec reconstruction $\hat{x}$，再检查 text → predicted codes 是否产生了目标内容、speaker 和 style。否则遇到一个坏样本时，无法判断失败来自压缩还是生成。

## 13. 怎样选择音频生成架构？

模型名字变化很快，底层设计选择更稳定。先从任务与表示出发，而不是先看排行榜。

| 需求 | 优先评估的架构 | 原因 | 决策前必须测什么 |
|---|---|---|---|
| CPU 或边缘端旁白 | Kokoro 一类小型 mel / vocoder TTS | 体积小、serving 简单 | 目标 CPU 上的 RTF、发音、voice 覆盖 |
| Zero-shot voice cloning | 各选一个 codec LM 和 flow-matching system | 两大家族用不同机制都能克隆 | 同意流程、speaker similarity、文本准确度、噪声复制 |
| 实时 voice agent | Streaming codec LM 或低延迟 hybrid | 增量 code 可以自然进入 waveform window | 首包时间、打断延迟、真实并发下 RTF |
| 有声书或批量合成 | Flow model 或稳定的小型 TTS | 并行生成与长文稳定性可能比首包更重要 | 段落连续性、漂移、throughput、编辑成本 |
| 表现力强的对话与音效 | 有显式控制的 audio LM | 语音、笑声、呼吸与 style token 可以共享上下文 | 控制成功率、可懂度、重复 / 漏词率 |
| Speech editing / infilling | Flow matching 或 masked acoustic model | 在已知音频左右两侧条件化很自然 | 边界平滑度、未编辑区域保真度 |

一条很好的学习路线是 [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) → [F5-TTS](https://github.com/SWivid/F5-TTS) → [Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b)：从能本地运行的小型合成器，到连续非自回归路线，再到离散自回归 audio language model。License 和语言覆盖是部署输入，不是架构属性；实际使用时必须重新核对模型卡。

## 14. 实践路径

现有的 <a href="../../../22-audio-generation/notebooks/01_kokoro_tts.html">Kokoro notebook</a> 提供最小本地合成循环和 TTS → ASR 回环测试。下一条合适的实践路径是：先听 codec reconstruction、拆 RVQ，再运行 Higgs voice cloning 与 inline control，最后测量完整 ASR → LLM → TTS voice pipeline。

## 15. 最后要记住什么？

1. Codec 是完整的 encoder–quantizer–decoder；codebook 是学习到的向量字典；audio code 是这张字典里的整数索引。
2. RVQ 用多次残差修正表示一个 frame；后层 codebook 细化误差，并不独占某个可解释属性。
3. Higgs 把 24 kHz 音频压缩到 25 frames/s、8 路 code stream，再让 Qwen3 风格模型预测 delayed code rows。
4. Reverse delay 恢复对齐 codec frames，codec decoder 再把 latents 上采样成 waveform samples。
5. VALL-E 让 neural-codec language modeling 在 TTS 中进入主流视野；Higgs 是精炼、可控、面向 streaming 的后继者，不是起点。
6. Codec LM 是现代 TTS 的一条主要路线；continuous diffusion / flow-matching TTS 是另一条。

## 16. 关键论文与技术资料

| 工作 | 年份 | 为什么读 |
|---|---:|---|
| [SoundStream](https://arxiv.org/abs/2107.03312) | 2021 | 卷积 neural codec + RVQ 基础 |
| [AudioLM](https://arxiv.org/abs/2209.03143) | 2022 | 把音频生成改写成离散 token 上的语言建模 |
| [EnCodec](https://arxiv.org/abs/2210.13438) | 2022 | 实用、高保真的 neural audio compression |
| [VALL-E](https://arxiv.org/abs/2301.02111) | 2023 | Conditional codec LM 与 3 秒 zero-shot TTS |
| [Voicebox](https://arxiv.org/abs/2306.15687) | 2023 | 面向语音生成与编辑的非自回归 flow matching |
| [F5-TTS](https://aclanthology.org/2025.acl-long.313/) | 2025 | 极简、完全非自回归的 flow-matching TTS |
| [Higgs TTS 3 模型卡](https://huggingface.co/bosonai/higgs-tts-3-4b) | 2026 | 多 codebook 架构、控制、克隆与 serving 示例 |
| [Higgs Audio V2 tokenizer 文档](https://huggingface.co/docs/transformers/model_doc/higgs_audio_v2_tokenizer) | 2026 | Higgs 家族使用的 semantic + acoustic codec |
