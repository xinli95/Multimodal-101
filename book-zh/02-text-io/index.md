# 02 · Text I/O — Tokenizer、Chat Template 与占位符

**在数据流中的位置**：`messages ──► chat template ──► tokenizer ──► input_ids`

多模态模型里"多模态"的部分，反直觉地从**文本**管线开始。在任何一个像素被处理之前，chat template 就已经决定了图像放在哪：它把一条 `{"type": "image"}` 展开成 280 个完全相同的占位符 token id，外面裹上 begin/end-of-image 标记。08 章会用真正的视觉特征覆盖掉这些占位符的 embedding。中间的一切，都是在一串扁平整数上做记账。

## 你会学到

1. `Gemma4Processor` 如何把 tokenizer、图像处理器、视频处理器、音频特征提取器合并到一个 `__call__` 背后
2. `apply_chat_template` 如何把 `messages` 变成字符串，占位符段怎么算出来——包括 token 数取决于时长的**动态**音频情形
3. 特殊 token 表，以及模型为什么必须在 embedding 查表前把占位符 id 临时改写成 `pad_token_id`
4. Gemma 4 原生的 `system` 角色、可配置 thinking 模式与 function calling 格式
5. 为什么批量生成时 `padding_side="left"` 是强制的

## 特殊 token

| Token | ID | 用途 |
|---|---|---|
| `boi_token_id` | 255999 | 图像开始 |
| `boa_token_id` | 256000 | 音频开始 |
| `image_token_id` | 258880 | 图像占位符——重复 `image_seq_length`（默认 **280**）次 |
| `audio_token_id` | 258881 | 音频占位符——重复 `ceil(时长ms / 40)` 次，上限 750 |
| `eoi_token_id` | 258882 | 图像结束 |
| `eoa_token_index` | 258883 | 音频结束 |
| `video_token_id` | 258884 | 视频占位符 |

每个音频 token 对应 40ms 不是随便定的：它来自音频塔对 10ms 帧做的 4× 时间下采样（05 章）。

## 源码地图

| 文件 | 符号 | 作用 |
|---|---|---|
| `processing_gemma4.py` | `Gemma4Processor.__call__` | 四个模态共用的入口 |
| | `prepare_inputs_layout`、`validate_inputs` | 交错输入的顺序与一致性 |
| | `replace_image_token` / `replace_audio_token` / `replace_video_token` | 占位符段展开 |
| | `_get_num_multimodal_tokens`、`_compute_audio_num_tokens` | 每份输入值多少个占位符 |
| `modeling_gemma4.py` | `Gemma4Model.get_placeholder_mask` | 消费端：把这些占位符再找回来 |
| | `Gemma4TextScaledWordEmbedding` | 乘以 `√hidden_size` 的 embedding 查表 |

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_tokens_and_template.ipynb` | 把一条多模态 `messages` 一步步展开——模板原始输出、token id、逐段解码——肉眼定位每一段占位符；再走一遍 tool calling 与 thinking 模式对比 | 🟢 CPU，只需 tokenizer |
| <a href="../../02-text-io/notebooks/02_tokenize_everything.html"><code>02_tokenize_everything.ipynb</code></a> | 更宽的视角：文本/图像/音频/视频在整个领域里各自怎么变成 token，压缩比是多少 | 🟢 CPU |

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/02-text-io/index.html)。
