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
| `processing_utils.py` / `processing_gemma4.py` | [`Gemma4Processor.__call__`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/processing_utils.py#L648) | 四个模态共用的继承入口 |
| `processing_gemma4.py` | [`prepare_inputs_layout`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L110)、[`validate_inputs`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L134) | 交错输入的顺序与一致性 |
| | [`replace_image_token`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L169) / [`replace_audio_token`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L192) / [`replace_video_token`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L173) | 占位符段展开 |
| | [`_get_num_multimodal_tokens`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L205)、[`_compute_audio_num_tokens`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L260) | 每份输入值多少个占位符 |
| `modeling_gemma4.py` | [`Gemma4Model.get_placeholder_mask`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2231) | 消费端：把这些占位符再找回来 |
| | [`Gemma4TextScaledWordEmbedding`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1465) | 乘以 `√hidden_size` 的 embedding 查表 |

## 源码走读

### 1. `Gemma4Processor` 是四个处理器套一件风衣

```python
class Gemma4Processor(ProcessorMixin):
    def __init__(self, feature_extractor, image_processor, tokenizer, video_processor,
                 chat_template=None, image_seq_length: int = 280,
                 audio_seq_length: int = 750, audio_ms_per_token: int = 40, **kwargs):
```

四个子处理器，一个对象。`AutoProcessor.from_pretrained` 从单个 `processor_config.json` 把四个都建出来，而 `processor(text=..., images=..., videos=..., audio=...)` 把每个参数路由给它的主人，再把结果合并成一个 `BatchFeature`。这是库里每个多模态模型的通用模式；在这里学会一次就够。

构造函数的两个默认值是本章的全部枢纽：

- `image_seq_length = 280` —— 一张图值多少个占位 token
- `audio_ms_per_token = 40` —— 一个音频 token 值多少毫秒

源码里的注释解释了 40 从哪来，这也是 05 章的承重事实：*"the SSCP convolution's 4× time reduction on 10ms frames."*

### 2. chat template 只发出**一个** `<|image|>`，不是 280 个

这是关于多模态 chat template 最常见的误解，值得说精确。把模板拉下来读：

```python
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
print(proc.chat_template)   # 386 行 Jinja
```

在消息循环里，一条 `image` 类型的内容项只产生这个：

```jinja
{%- elif item.get('type') in ['image', 'image_url'] -%}
    {{- '<|image|>' -}}
{%- elif item.get('type') in ['audio', 'input_audio'] -%}
    {{- '<|audio|>' -}}
{%- elif item.get('type') == 'video' -%}
    {{- '<|video|>' -}}
```

每个附件一个标记。展开成 280 个的事发生在更晚的 **processor** 里，等图像处理器真正看过这张图、决定它值多少 soft token 之后：

```python
def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
    num_soft_tokens = image_inputs["num_soft_tokens_per_image"][image_idx]
    return f"{self.boi_token}{self.image_token * num_soft_tokens}{self.eoi_token}"
```

所以链条是 **模板 → 一个标记 → processor → `<boi>` + N×`<image>` + `<eoi>`**，而 N 依赖数据。这个顺序正是 `apply_chat_template(..., tokenize=True)` 必须拿到图像的原因；你没法在不处理附件的情况下 tokenize 一个多模态 prompt。

音频做同样的事，但 N 是靠**模拟编码器**从波形算出来的：

```python
def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
    mask = audio_inputs["input_features_mask"][audio_idx]
    # Simulate two stride-2 conv blocks on the mask
    t = len(mask)
    for _ in range(2):
        t_out = (t + 2 - 3) // 2 + 1
        mask = mask[::2][:t_out]
        t = len(mask)
    return f"{self.boa_token}{self.audio_token * int(mask.sum())}{self.eoa_token}"
```

两次 stride-2 卷积、kernel 3、padding 1 —— 作用在 **mask** 而不是特征上，纯粹为了计数。为什么 processor 要复制编码器的算术而不是去问它？因为占位符必须在模型运行**之前**就存在于 `input_ids` 里。这个数必须被预测，而且必须分毫不差。`_compute_audio_num_tokens` 的 docstring 说明了不准会怎样：

> *Must match `audio_mask.sum()` from the audio tower or vLLM's `_merge_multimodal_embeddings` will raise on a length mismatch.*

同一段 docstring 还标出一个读 config 的人一定会踩的坑：

> *note: `config.conv_kernel_size=5` is the **conformer** depthwise conv, NOT this one*

下采样卷积是 kernel 3 / stride 2 / padding 1，而且**在 `Gemma4AudioConfig` 里根本没有暴露** —— 它们是硬编码在编码器和这个计数函数里的架构常量。你若要改音频栈，得改两处。这类耦合只有读源码才看得见。

视频是个异类，它的处理方式本身是一堂设计课：

```python
timestamp_str = [f"{int(seconds // 60):02d}:{int(seconds % 60):02d}" for seconds in metadata.timestamps]
video_replacement = " ".join(
    [f"{t} {self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}" for t in timestamp_str]
)
```

每一个被采样的帧先变成 `MM:SS` 的**字面文本**，后面跟着该帧的 soft token 段。时间信息不是编码进 embedding 或位置下标的 —— 它作为字符串写进 prompt，被语言模型当作普通文本读取。对比 Qwen-VL 的 M-RoPE，后者把时间编码成一个位置维度（见 [landscape](../landscape.md)）。另外注意视频段被 `<boi>`/`<eoi>` 这对**图像**标记包裹，里面装的是 `<|video|>`。

### 3. token 表，以及 `pad_token_id` 的腾挪

| Token | ID | 说明 |
|---|---|---|
| `<\|image\|>` | 258880 | 重复 `num_soft_tokens_per_image` 次 |
| `<\|audio\|>` | 258881 | 重复 `ceil(duration_ms / 40)` 次，上限 750 |
| `<\|video\|>` | 258884 | 由 processor 在运行时加入 —— 见下 |
| `<boi>` / `<eoi>` | 255999 / 258882 | 同时包裹图像**和**视频段 |
| `<boa>` / `<eoa>` | 256000 / 258883 | 包裹音频段 |

构造函数里一行极其坦率的注释：

```python
# FIXME: add the token to config and ask Ryan to re-upload
tokenizer.add_special_tokens({"additional_special_tokens": ["<|video|>"]})
```

视频 token 不在发布的 tokenizer 里；processor 每次构造时现打补丁。如果你比较过构建 processor 前后的 `len(tokenizer)`，就知道这条值得记住。

在某些配置下占位符 id 超出模型可用的 embedding 范围，所以模型在查表前会做这件事（08 章详述）：

```python
llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, llm_input_ids)
inputs_embeds = self.get_input_embeddings()(llm_input_ids)
```

每个占位符临时变成 pad token，被 embedding（产出一堆马上要被覆盖的垃圾），然后 `masked_scatter` 把真正的 soft token 写上去。另一种选择 —— 用越界 id 去索引 embedding 表 —— 是崩溃。

### 4. `validate_inputs`：你真会撞上的那个错误

```python
n_images_in_text = [sample.count(self.image_token) for sample in text]
n_images_in_images = [len(sublist) for sublist in images]
if n_images_in_text != n_images_in_images:
    raise ValueError("The total number of <|image|> tokens in the prompts should be the same as the number of images passed. ...")
```

是**逐样本**比对，不是整批。一个"样本 0 有两张图、样本 1 没有图"的 batch，必须以嵌套列表 `[[img, img], []]` 传入，而不是两个元素的扁平列表。`prepare_inputs_layout` 调用 `make_nested_list_of_images` 来强制这个结构，而且在你只传图不传文本时还会**替你编**一个 prompt：

```python
if images and not text:
    text = [" ".join([self.image_token] * len(image_list)) for image_list in images]
```

### 5. 轮次语法、thinking 与 tools

Gemma 4 的模板不是 ChatML。轮次由成对的尖括号 token 分隔：

```
<bos><|turn>system
...系统文本与工具声明...
<turn|>
<|turn>user
<|image|>What is shown in this image?<turn|>
<|turn>model
```

有三处值得去读 Jinja 原文。

**system 轮是被合成出来的，不只是照抄。** 三个条件**任一**成立它就会出现 —— 有 system 消息、有 tools、或者开了 thinking：

```jinja
{%- if enable_thinking or tools or (messages and messages[0]['role'] in ['system', 'developer']) -%}
    {{- '<|turn>system\n' -}}
    {%- if enable_thinking -%}{{- '<|think|>\n' -}}{%- endif -%}
```

所以 `enable_thinking=True` 不是生成参数，而是往第一个 system 轮顶部注入一个 `<|think|>` token。`developer` 被当作 `system` 的别名接受。

**thinking 会从历史里被剥掉。** assistant 内容会经过 `strip_thinking` 宏，移除 `<|channel>` 与 `<channel|>` 之间的一切；而 reasoning 只对**最后一条 user 消息之后**的消息重新渲染：

```jinja
{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls')) -%}
```

模型三轮之前的推理是刻意不喂回去的 —— 那是草稿纸，不是上下文。`preserve_thinking=True` 会在工具调用链里覆盖这个行为，因为导向某次调用的推理值得保留。

**工具用一套紧凑 DSL 声明，不是 JSON。** 一条工具声明渲染成 `<|tool>declaration:name{description:<|"|>...<|"|>,parameters:{...},type:<|"|>OBJECT<|"|>}<tool|>`，其中 `<|"|>` 是一个专门的字符串引号 **token**。用单个 token 而不是字面 `"` 来做结构标点，相对 JSON 只**略微**便宜 —— 在上面那个 weather 工具上实测约省 6% —— 但它让模型**难以破坏格式**：`<|"|>` 是一个 token，不存在半开或未转义的引号，解析器可以按 token id 切分，而不必对模型输出做 JSON 容错。模板也拒绝替一个常见的客户端 bug 遮丑：

```jinja
{{- raise_exception("chat_template: tool_calls[].function.arguments must be a JSON object (mapping), not a string. Deserialize arguments before passing to the template.") -}}
```

### 6. `padding_side="left"`，以及它为什么不可选

Gemma 4 文档里每个例子构造 processor 的方式都一样：

```python
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it", padding_side="left")
```

用右侧 padding 时，一批长度不同的 prompt 会把 pad token 放在真实内容**之后**，于是每条序列最后一个真实 token 落在不同下标上 —— 而 `generate` 是在张量末尾追加新 token，也就是追加在 padding 后面。左侧 padding 让每条序列的末尾 token 对齐到同一位置，这正是解码循环所假设的。搞错不会抛异常；你只会安静地得到更差的输出，这更糟。

对应的出口习惯：

```python
input_len = inputs["input_ids"].shape[-1]
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0][input_len:], skip_special_tokens=True))
```

`generate` 返回 prompt + 补全。在 `input_len` 处切片才是拿到模型真正说了什么的方式。

## 设计空间

- **占位段展开**（Gemma 4、LLaVA、Qwen-VL）：预留 N 个文本位置，覆盖它们的 embedding。LLM 的代码路径完全不变 —— 它只见过一串 embedding。代价是每张图吃掉 N 个真实上下文槽位，这也是 N 的大小（03 章）如此要命的原因。
- **交叉注意力注入**（Flamingo、Llama 3.2 Vision）：视觉特征活在序列之外，通过专门的层被 attend。不消耗上下文，但 LLM 需要新参数和改过的前向。
- **固定 query 压缩**（BLIP-2）：Q-Former 把任意图像压成恰好 32 个 token。便宜且恒定，但对密集文字来说这个瓶颈毫不留情。

Gemma 4 走第一条路，然后把力气花在让 N **可控**上 —— 也就是 03 章那张菜单。

关于时间戳，分野同样干净：Gemma 4 把 `MM:SS` 当文本写进 prompt；Qwen-VL 把时间编码成一个 RoPE 维度。文本花 token 但极易解释；位置维度免费，但要求整条技术栈就此达成一致。

## 自测

1. 为什么不能对一个含图像的消息列表直接调 `apply_chat_template(messages, tokenize=True)` 而不同时传入图像本身？
2. 一段 3.4 秒、16kHz 的音频。大约多少个 `<|audio|>` 占位符？上限由什么决定？
3. `<|video|>` 从哪来？它为什么不在 `tokenizer.json` 里？
4. 你传了两条 prompt，第一条带两张图、第二条没有图，然后收到一个关于数量的 `ValueError`。`images` 应该是什么形状？
5. `enable_thinking=True` 改变了渲染后的 prompt。改变具体出现在哪里？长什么样？
6. 你的批量生成结果比单条生成明显更差。第一个该检查的是什么？

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_tokens_and_template.ipynb` | 把一条多模态 `messages` 一步步展开——模板原始输出、token id、逐段解码——肉眼定位每一段占位符；再走一遍 tool calling 与 thinking 模式对比 | 🟢 CPU，只需 tokenizer |
| <a href="../../02-text-io/notebooks/02_tokenize_everything.html"><code>02_tokenize_everything.ipynb</code></a> | 更宽的视角：文本/图像/音频/视频在整个领域里各自怎么变成 token，压缩比是多少 | 🟢 CPU |
