# 02 · Text I/O — 从 Messages 到模型输入

这一章可以独立阅读。它只回答一个问题：**Python 里的聊天记录，怎样变成 Gemma 4 收到的整数序列？** 如果消息里带了一张图片，会有两条并行路径：

```text
messages ── chat template ──► 含一个 <|image|> 的 prompt 文本
image    ── image processor ─► pixel_values + 数量 N
                                      │
prompt + N ── 展开图片标记 ──► 最终 prompt ── tokenizer ──► input_ids
```

本章沿着上面一条路走，并解释两条路在哪里汇合。[03 章](../03-image-processor/index.md)会解释图像处理器如何得到 `pixel_values` 和 `N`；[08 章](../08-fusion-and-masks/index.md)会解释模型怎样把预留的 `N` 个位置换成图像特征。现在不需要预先懂那两部分的实现。

## 你会学到

1. chat template、tokenizer 与多模态 processor 各自负责什么
2. 怎样从一轮最简单的对话开始读 Gemma 4 的 template
3. 为什么一个图片标记最后会变成连续 `N` 个 placeholder ID
4. 组 batch 时，为什么图片要按 sample 嵌套
5. left padding 什么时候适用，又有什么代价

## 三个对象，三份工作

这几个名字很容易混在一起，先把边界划清楚：

| 对象 | 输入 | 输出 | 工作 |
|---|---|---|---|
| chat template | 结构化的 `messages` | 一条 prompt 字符串 | 按 Gemma 4 训练时的语法序列化角色与内容 |
| tokenizer | 字符串 | 整数 `input_ids` | 把字符串切成词表片段，再查出各自的 ID |
| `Gemma4Processor` | 文本，以及可选的图片、音频、视频 | 一份可直接交给模型的 batch | 协调 tokenizer 和三个模态各自的预处理器 |

Chat template 不是模型，也不要求你预先懂 ChatML。它就是一个**序列化器**。不同 chat model 训练时使用了不同的分隔符，因此同一段对话必须先写成各自的“方言”，然后才能 tokenize。

## 从一轮纯文本对话开始

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
messages = [
    {"role": "user", "content": "用一句话解释 tokenization。"}
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(prompt)
```

结果里最重要的结构是：

```text
<bos><|turn>user
用一句话解释 tokenization。<turn|>
<|turn>model
```

按照字面读就可以：

- `<bos>` 表示整条序列开始。
- `<|turn>user\n` 打开 user turn，`<turn|>` 把它关上。
- `<|turn>model\n` 是一个内容为空的 model turn 头。`add_generation_prompt=True` 添加的就是它，用来告诉生成过程接下来轮到哪个角色说话。

完成整条字符串之后，tokenizer 才把其中的普通文字和 turn 分隔符一起转成 ID：

```python
encoded = processor.tokenizer(prompt, return_tensors="pt")
print(encoded["input_ids"].shape)
```

这就是最基本的管线。Jinja 只是实现 template 的语言：它循环 `messages`，按条件拼出这条字符串。

## 从内到外读 template

不需要一口气看懂 386 行 Jinja。先看所有 chat template 都必须做的两个决定。

### 1. 写出角色，关上每一个 turn

概念上，message loop 做的就是：

```jinja
<|turn>{{ message['role'] }}
{{ message['content'] }}<turn|>
```

真实 Jinja 还会做校验、处理 content list，但语法仍然是“打开角色、写入内容、关闭 turn”。历史里的 assistant 回复也没有特殊之处：它会成为一个已经闭合的 `model` turn，排在下一轮 `user` 前面。

### 2. Content 可以是字符串，也可以是 list

纯文本可以直接写字符串。多模态消息用 list，是为了明确文本和附件的先后顺序：

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "path": "cat.jpg"},
        {"type": "text", "text": "这只猫在做什么？"},
    ],
}]
```

对这份 list，template 的内层循环把每一项还原成文本：

```jinja
{%- if item.get('type') == 'text' -%}
    {{- item['text'] -}}
{%- elif item.get('type') in ['image', 'image_url'] -%}
    {{- '<|image|>' -}}
{%- endif -%}
```

此时一张图片只变成**一个** `<|image|>` 标记。Template 不检查像素，因此现在还不知道图片最终需要占几个模型位置。

### 3. System message 只是可选的第一轮

```python
messages = [
    {"role": "system", "content": "像一位耐心的老师那样回答。"},
    {"role": "user", "content": "什么是 token？"},
]
```

Gemma 4 先把第一条渲染成 `system` turn，再渲染 `user` turn。它也接受 `developer` 作为第一个角色的别名。这是 Gemma 4 自己的语法，不是所有 chat model 共用的标准。

掌握这三条，已经足够阅读普通的 Gemma 4 prompt。Thinking 和 tools 是可选扩展，放在本章后面。

## 从一个图片标记到 N 个预留位置

回到图片例子。这里必须区分两种表示：

```text
template 之后：   <|image|> 这只猫在做什么？
processor 之后：  <|image> <|image|> × N <image|> 这只猫在做什么？
```

这些名字确实很像。外面一对在代码里叫 `boi_token` 和 `eoi_token`，标记图片块的边界。中间重复的 `<|image|>` 才是 **placeholder**：它们是一排空座位，稍后每个位置都会放入一个图像向量。

`N` 并不总是 280。默认图像预算下，280 是上限；较小或长宽比特殊的图片可能用得更少。`Gemma4Processor` 先运行 image processor，读取 `num_soft_tokens_per_image`，再展开标记：

```python
def replace_image_token(self, image_inputs, image_idx):
    n = image_inputs["num_soft_tokens_per_image"][image_idx]
    return f"{self.boi_token}{self.image_token * n}{self.eoi_token}"
```

这个顺序解释了一个看起来有点反直觉的 API 规则：

- `apply_chat_template(..., tokenize=False)` 不读取图片，也能展示带一个图片标记的字符串。
- 要得到可直接给模型的多模态输入，processor 必须拿到图片本身（或路径/URL），才能算出 `N`、展开标记，再 tokenize 最终字符串。

因此完整的便捷调用是：

```python
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
)
```

这里图片路径已经写在 message content 里。如果先单独渲染文本，再调用 `processor(text=..., images=...)`，只是手动完成同一套步骤。

## Token map：先分清语法 token 与 placeholder

先按功能分类，原始 ID 表才有意义：

| 类别 | 例子 | 这些 ID 对模型意味着什么 |
|---|---|---|
| turn 语法 | `<bos>`、`<\|turn>`、`<turn\|>` | 描述对话结构、拥有正常 learned embedding 的 token |
| 模态边界 | `boi_token` / `eoi_token`、`boa_token` / `eoa_token` | 标记一个模态块从哪里开始、到哪里结束，拥有正常 learned embedding |
| 模态 placeholder | `<\|image\|>`、`<\|audio\|>`、`<\|video\|>` | 预留的位置；它们的文本 embedding 会被丢弃，换成对应模态的特征 |

Gemma 4 接口使用的具体 ID 是：

| Config / processor 字段 | ID | 用途 |
|---|---:|---|
| `boi_token_id` | 255999 | 图像或视频块开始 |
| `boa_token_id` | 256000 | 音频块开始 |
| `image_token_id` | 258880 | 每个图像特征向量重复一次 |
| `audio_token_id` | 258881 | 每个音频特征向量重复一次 |
| `eoi_token_id` | 258882 | 图像或视频块结束 |
| `eoa_token_index` | 258883 | 音频块结束；`_index` 是 config 里的真实命名 |
| `video_token_id` | 258884 | 每个视频特征向量重复一次 |

### 为什么查 embedding 前仍要改写 placeholder

先纠正一个很容易产生的误读：**官方发布的 Gemma 4 checkpoint 目前并不存在 placeholder ID 越界。** 它们的文本 `vocab_size` 是 262,144，而 placeholder ID 是 258,880–258,884。`<|video|>` 的确是在 processor 构造时加入 tokenizer，但选中的 ID 仍然落在模型 embedding 表以内。

源码注释写的是“如果 image token 是 OOV，就先换成 PAD”，但实际实现会无条件替换。这样做能保护 tokenizer 与模型词表大小不一致的自定义或 resize 配置；同时也避免依赖一份注定没有意义的文本 embedding：placeholder 是 processor 与模型约定的**接口哨兵**，不是 decoder 需要保留的文字。

模型因此按下面的顺序处理：

1. 先比较 ID，找出全部 placeholder 位置。
2. 暂时把这些 ID 换成合法的 `pad_token_id`。
3. 正常查询文本 embedding。
4. 用真实的图像、音频或视频向量覆盖 placeholder 位置。

[08 章会完整展示这四步，以及第 4 步使用的 `masked_scatter`](../08-fusion-and-masks/index.md)。目前只要记住：这里产生的 pad embedding 是一次性的。对官方 checkpoint 来说，这次替换是无害的防御性处理；如果自定义配置里的哨兵 ID 真的越界，它还能避免 index error。

`<|video|>` 在运行时加入 tokenizer，确实很容易让人怀疑这里发生了越界：

```python
# FIXME: add the token to config and ask Ryan to re-upload
tokenizer.add_special_tokens({"additional_special_tokens": ["<|video|>"]})
```

一般来说，调用 `add_special_tokens` 并不会自动 resize 模型的 embedding matrix。但这个 checkpoint 已经预留了足够多的行，所以“video token 后加，因此查表越界”**不是**这里真实发生的事。图像、音频、视频三类 placeholder 都走同一条安全替换路径，因为它们的文本 embedding 本来就不应该保留下来。

## 为什么图片 batch 是嵌套 list

这段校验放在这里，是因为 placeholder 展开是一份逐 sample 的契约：每条 prompt 里图片标记的数量，必须等于这条 sample 实际附带的图片数量。

例如一个 batch 有两条 prompt：

| Batch sample | 自己的 prompt 中有几个图片标记 | 对应图片 |
|---|---:|---|
| sample 0 | 2 | `[img_a, img_b]` |
| sample 1 | 0 | `[]` |

手动调用 processor 时应写成：

```python
texts = [prompt_with_two_image_markers, prompt_with_no_image_markers]
images = [[img_a, img_b], []]
inputs = processor(text=texts, images=images, return_tensors="pt")
```

外层 list 是 batch；每个内层 list 装一条 sample 的图片。它**不是**重复 placeholder 的数量——内层的每张图片都会各自展开成一段长度为 `N` 的 placeholder run。`validate_inputs` 对比的正是这两个逐 sample 计数，从而在图像特征被接错 prompt 之前直接报错。

如果图片路径或 URL 已经写在 `messages` 里，并且使用 `apply_chat_template(..., tokenize=True)`，processor 会自动替你建立这份逐 sample 分组。

## 音频和视频复用同一份契约

另外两个模态不需要学习新的 template 概念：

| Template 发出 | Processor 随后预留 | `N` 来自哪里 |
|---|---|---|
| 一个 `<\|image\|>` | `N` 个图像位置 | 图片尺寸与所选 token budget |
| 一个 `<\|audio\|>` | `N` 个音频位置 | audio front end 的有效输出长度 |
| 一个 `<\|video\|>` | 每个采样帧 `N` 个位置 | 帧采样与图像处理 |

本章只需要记住这份契约：**placeholder 的数量必须等于对应 tower 最终返回的特征向量数量。** 音频大约每 40 ms 对应一个 token，上限为 750，但推导这个数字需要先理解 audio tower。[05 章](../05-audio-and-video/index.md)会从两层 stride-2 convolution 推出它。<a href="../../02-text-io/notebooks/01_tokens_and_template.html">动手 notebook</a> 里的可选音频部分则直接对不同波形长度调用 `_compute_audio_num_tokens`，不把 encoder 细节设成本章前置知识。

视频还会在采样帧块前写入人能读懂的 `MM:SS` 时间戳；具体采样逻辑同样留给 [05 章](../05-audio-and-video/index.md)。

## 可选 template 功能：thinking 与 tools

普通 turn 已经理解之后，其余 Jinja 就容易定位了。

### Thinking 会改变渲染后的 prompt

`enable_thinking=True` 是 chat template 参数，不是 `generate()` 参数。Template 会在需要时合成 system turn，并在其中插入 `<|think|>`。默认情况下，它还会从历史对话中移除旧 reasoning span，避免草稿推理随着轮数不断撑大 prompt。Notebook 会把同一段对话在 thinking 开、关时分别渲染，直接对比差异。

### Tools 只是再多一层序列化

传入 `tools=[...]` 后，system turn 会包含工具声明。Gemma 4 用自己的紧凑 token DSL 序列化这些声明，模型的 tool call 也使用配套语法。你不需要懂 ChatML，也不需要背这套 DSL：把普通 JSON Schema 字典交给 `apply_chat_template`，让 template 负责序列化；只有调试工具调用时才需要查看渲染结果。

如果 `tool_calls[].function.arguments` 还是一条 JSON 字符串，template 会刻意拒绝它；调用方要先把它反序列化成字典。<a href="../../02-text-io/notebooks/01_tokens_and_template.html">Notebook</a> 保留了一份完整的渲染实例。

## Left padding：生成阶段的惯例，不是普遍规则

使用 decoder-only 模型做 batch generation 时，应这样加载：

```python
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-E2B-it",
    padding_side="left",
)
```

把两条不同长度的序列排在一起，原因就很直观：

```text
left padding                 right padding
[PAD PAD A B C]              [A B C PAD PAD]
[D   E   F G H]              [D E F G   H  ]
             ↑ 下一 token 的 logits 从最后一列读取
```

标准 decoder-only generation loop 会统一读取每一行最后一个 tensor 位置的 next-token logits。Left padding 保证这个位置对每条 sample 都是真实 prompt token。Right padding 会让短 sample 停在 padding 位置；attention mask 能阻止这个位置关注其他 padding，却不能把它的 hidden state 变成“最后一个真实 token”的 hidden state。

Left padding 常见于 **decoder-only 的 batch inference**，不是所有任务：

- 单条未 padding 的 prompt 根本没有左右之分。
- Encoder-only 模型和 causal LM training 通常仍用 right padding；训练会屏蔽 pad label，也不会要求每一行都从最后一列预测 next token。
- Left padding 不会减少 padding 浪费的算力或显存。要解决这个问题，需要 length bucketing、packing 或 continuous batching。
- 自定义 position ID 逻辑或 attention kernel 可能假设有效 token 是左对齐的连续前缀。Transformers 支持 Gemma 4 的标准生成路径；自定义管线仍需正确保留 attention mask 和 position IDs。

因此更准确、也更实用的规则是：**交给 Gemma 4 `generate()` 的变长 batch 使用 left padding；不要把它当成 tokenizer 的普遍设置。** [09 章](../09-generation-and-serving/index.md)会继续讲 batch、cache 与从生成序列中切掉 prompt。

## 源码地图

| 文件 | 符号 | 为什么读它 |
|---|---|---|
| `processing_utils.py` | [`ProcessorMixin.apply_chat_template`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/processing_utils.py#L1805)、[`ProcessorMixin.__call__`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/processing_utils.py#L603) | 渲染与处理这两个阶段 |
| `processing_gemma4.py` | [`Gemma4Processor.prepare_inputs_layout`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L100)、[`validate_inputs`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L122) | 逐 sample 的图片分组与校验 |
| `processing_gemma4.py` | [`replace_image_token`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L153) | 一个图片标记变成一段由边界包裹的 placeholder |
| `processing_gemma4.py` | [`replace_audio_token`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L173)、[`replace_video_token`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/processing_gemma4.py#L157) | 可选细节，建议读完 05 章再看 |
| `modeling_gemma4.py` | [`Gemma4Model.forward`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2098) | placeholder 的消费端；在 08 章阅读 |

## 自测

1. 原始 user text 里没有的哪些信息，是 chat template 补进去的？
2. 为什么 template 只发出一个图片标记，而 `input_ids` 里有许多图片 placeholder ID？
3. `images=[[img_a, img_b], []]` 中，外层和内层 list 各代表什么？
4. 运行时加入的 `<|video|>` 是否超出了官方模型的 embedding 表？模型为什么仍把它换成 `pad_token_id`？
5. `add_generation_prompt=True` 到底添加了什么？
6. Left padding 在什么情况下有用？再说出一个 right padding 仍然常见的场景。

<details>
<summary>展开答案</summary>

<ol>
<li>Template 会补上角色名、turn 边界、序列/控制 token、模态标记，以及按需加入的空 model turn 头；这些信息都不在原始 user text 里。</li>
<li>Template 只知道“这里有一张图片”，所以先发出一个标记。Processor 检查图片后算出 <code>N</code>，再把这个标记展开成恰好 <code>N</code> 个预留位置。</li>
<li>外层 list 表示 batch；每个内层 list 装属于一条 sample 的图片。因此 sample 0 有两张图，sample 1 没有图。</li>
<li>没有。官方模型有 262,144 行 embedding，而 <code>video_token_id=258884</code>，仍在范围内。模型依然换成 <code>pad_token_id</code>，是因为 placeholder embedding 本来就会被丢弃，同时这条路径也能保护 ID 真正越界的自定义配置。</li>
<li>它会追加一个内容为空的 <code>&lt;|turn&gt;model\n</code> 头，表示接下来生成的内容属于 model 角色。</li>
<li>把不同长度的 prompt 组成 batch，交给 decoder-only 模型生成时适合 left padding，因为每行最后一个位置都是真实 prompt token。Causal-LM 训练和 encoder-only 任务仍常用 right padding；单条未 padding prompt 则没有左右之分。</li>
</ol>

</details>

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| <a href="../../02-text-io/notebooks/01_tokens_and_template.html"><code>01_tokens_and_template.ipynb</code></a> | 先构造纯文本 prompt，再加入一张图片；查看渲染字符串、展开后的 placeholder run 与 ID。音频计数、thinking、tools 是可选进阶 | 🟢 CPU，只需 tokenizer/processor |
| <a href="../../02-text-io/notebooks/02_tokenize_everything.html"><code>02_tokenize_everything.ipynb</code></a> | 横向比较不同模型家族怎样离散化文本、图像与音频；不是 Gemma 4 主线的必读内容 | 🟢 CPU |
