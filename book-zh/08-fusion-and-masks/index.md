# 08 · Fusion and Masks — 模态真正相遇的地方

**在数据流中的位置**：塔与解码器之间——四个模态汇成一条序列的唯一那个点。

这是前半本书一直在铺垫的一章，而它比你想象的要小。04、05 两章产出的 soft token 已经在文本嵌入空间里了；02 章埋下的占位符段已经躺在 `input_ids` 里了。融合就是：找到占位符，把 soft token 覆盖上去，然后修好注意力 mask 让模型正确对待结果。

有意思的是 mask。句子是因果的——第 n 个 token 不能看见第 n+1 个。但图像不是序列，左上角的 patch 没有任何理由不许看右下角。于是 Gemma 4 支持 `use_bidirectional_attention="vision"`：**视觉 token 在自己的块内双向注意力，文本仍然因果。** 生成的 4D mask 有非常独特的形状，把它画成热力图，胜过任何文字描述。

## 你会学到

1. `get_placeholder_mask` 如何从 `input_ids` 或 `inputs_embeds` 找出图像/视频/音频位置，以及 `inputs_embeds` 那条路为什么必须跟一个嵌入后的 token 比较
2. 占位符 id 为什么要在 embedding 查表前改写成 `pad_token_id`——避开了什么索引错误
3. `masked_scatter` 如何把 soft token 写进 `inputs_embeds`，占位符数量与 soft token 数量不一致时错误长什么样
4. `mm_token_type_ids` 携带了什么，模型为什么除了 token id 之外还要它
5. 双向视觉 mask 如何构造（`create_masks_for_vision_model`、`get_block_sequence_ids_for_mask`），"块"是什么，一个 prompt 里有两张图会发生什么
6. 把最终 4D mask 画出来，直接从图上读出架构

## 源码地图

| `modeling_gemma4.py` 中的符号 | 作用 |
|---|---|
| [`Gemma4Model.get_placeholder_mask`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2231) | 定位图像 / 视频 / 音频位置 |
| [`Gemma4Model.forward`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2279) | `masked_scatter` 融合与 `pad_token_id` 改写 |
| [`create_masks_for_vision_model`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2112) | 双向视觉 mask |
| [`get_block_sequence_ids_for_mask`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2167) | 把连续的多模态段分组成块 |

## 源码走读

### 1. 找到占位符

```python
def get_placeholder_mask(self, input_ids=None, inputs_embeds=None):
    if input_ids is not None:
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        special_audio_mask = input_ids == self.config.audio_token_id
    else:
        special_image_mask = (inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device))).all(-1)
        ...
    return special_image_mask, special_video_mask, special_audio_mask
```

`input_ids` 这条路是整数比较。`inputs_embeds` 那条 —— 调用方直接给 embedding 时用 —— 必须*把占位符 id 嵌入后再比较向量*，这与 07 章 PLE 遇到的别扭是同一种，原因也一样：一旦你接受 embedding 作为入口，原本住在 id 里的信息就都得被恢复出来。

### 2. `pad_token_id` 的调包

```python
image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
multimodal_mask = image_mask | video_mask | audio_mask

llm_input_ids = None
if inputs_embeds is None:
    llm_input_ids = input_ids.clone()
    # Replace image id with PAD if the image token if OOV, to avoid index-errors
    llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, llm_input_ids)
    inputs_embeds = self.get_input_embeddings()(llm_input_ids)
```

占位符 id（258880–258884）可能超出 embedding 表的行数，所以查表前被改写成 `pad_token_id`（0）。这些位置得到的 embedding 毫无意义 —— 马上就要被覆盖 —— 但它们必须**合法**，因为越界索引是崩溃而不是警告。

PLE 也享受同等待遇，只是把 pad embedding 显式顶了上去：

```python
pad_embedding = self.language_model.embed_tokens.weight[self.config.text_config.pad_token_id, :]
llm_inputs_embeds = torch.where(multimodal_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds)
per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)
```

### 3. 融合就是一行，外加一个守卫

图像、视频、音频跑的是同一套三步：

```python
image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

n_image_tokens = image_mask.sum()
image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
torch_compilable_check(
    inputs_embeds[image_mask].numel() == image_features.numel(),
    f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}")

inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
```

**那个 `masked_scatter` 就是融合。** 03–05 章的一切存在都是为了产出 `image_features`；02 章的一切存在都是为了产出 `image_mask`；这一行把它们合在一起。此后世上再没有"图像"这种东西 —— 只有一串 embedding，其中一些来自像素。

`torch_compilable_check` 是你手工构造输入时会遇到的那个错误。当占位符数量与 soft token 数量不一致时它会触发，而这正发生在 processor 的预测与塔的实际输出出现分歧的时候 —— 恰是 02 章 `_compute_audio_num_tokens` 的 docstring 所防的那件事。用 `torch_compilable_check` 而不是普通 `assert`，是因为对张量值做 Python `assert` 会在 `torch.compile` 下打断图。

音频多一步，因为音频塔吐出的是补齐过的 batch：

```python
audio_features = audio_features[audio_mask_from_encoder.to(audio_features.device)]
```

编码器自己的有效性 mask 在散射前剥掉 padding —— 与 04 章里 `pooler_mask` 对视觉做的是同一件事。三个模态、三套 padding 约定，最后汇成一条扁平序列。

### 4. 块：把连续的多模态段分组

```python
def get_block_sequence_ids_for_mask(mm_token_type_ids, device):
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    return torch.where(is_vision, vision_group_ids, -1)
```

`mm_token_type_ids` 由 processor 产出（这正是 02 章提到 `model_input_names` 会追加它的原因），给每个位置标注模态。这个函数把标注变成**块 id**：每一段极大的连续视觉 token 拿到一个不同的整数，文本拿到 `-1`。

因此一个 prompt 里的两张图会得到块 id 0 和 1，其余一律 `-1`。块 0 里的 token 可以互相看见；块 1 里的可以互相看见；块 0 里的 token 不能双向看见块 1。没有这个，带两张图的 prompt 会让第一张图向前偷看第二张。

注意 `is_vision` 同时涵盖图像（1）和视频（2），但**不含音频** —— 音频 token 永远不享受双向待遇。

### 5. mask，以及所有人都会搞错的那件事

这里是论文插图永远不会告诉你的发现。`create_masks_for_vision_model` 的 docstring 直说了：

> For global (full attention) layers: causal only (no bidirectional)
> For local (sliding window) layers: AND(sliding_window, OR(causal, blockwise))
>
> Unlike Gemma 3 (which applies bidirectional attention on all layers), Gemma 4 explicitly disables bidirectional attention on global attention layers.

```python
full_mask = create_causal_mask(**mask_kwargs)

sliding_mask = create_causal_mask(
    **mask_kwargs,
    or_mask_function=blockwise_overlay(padded_block_sequence_ids),
    and_mask_function=sliding_window_overlay(config.sliding_window),
)
return {"full_attention": full_mask, "sliding_attention": sliding_mask}
```

**双向视觉注意力只发生在滑窗层上。** 全局层 —— 那 6 取 1 的、看得见整条序列的层 —— 保持严格因果，对图像 token 和对文本一视同仁。

一旦看见理由就很显然。全局层的 KV cache 是昂贵的那个，也是 `generate` 在解码步之间复用的那个；严格因果的 mask 才让这份 cache 有效，因为第 *n* 项永远不依赖 *n* 之后的任何东西。在那里放开双向，每个图像 token 缓存的 key 都会依赖更靠后的 token，增量解码就错了。滑窗层付得起：它们窗口小，而 blockwise overlay 被限制在图像跨度内，那些跨度在 prefill 时是完整在场的。

所以 Gemma 4 拿到了双向图像注意力的大部分好处 —— patch 在 6 层里的 5 层可以自由互看 —— 却完全不承担 cache 失效。Gemma 3 是全网施加；这是一次刻意的、有文档的修正。

仔细读这个组合式，因为顺序是承重的：

```
sliding = AND( sliding_window , OR( causal , blockwise ) )
```

`OR(causal, blockwise)` 说的是*"你可以向后看，**或者**看你自己图像块里的任何东西"*。`AND(sliding_window, …)` 再把结果裁到窗口内。因此一个大的图像块**并非**完全双向 —— 512 窗口里的 280 token 图像是，但两张 1120 token 的图像会被裁。组合顺序也解释了 `maybe_pad_block_sequence_ids` 那段舞蹈：`or_mask_function` 绕过了 `create_causal_mask` 通常施加的内部 padding，所以块 id 必须先被手工补齐到 `kv_length`。

以及决定走哪条路的分支：

```python
use_bidir = text_config.use_bidirectional_attention == "vision"
if use_bidir and mm_token_type_ids is not None:
    block_sequence_ids = get_block_sequence_ids_for_mask(mm_token_type_ids, device=inputs_embeds.device)
    causal_mask_mapping = create_masks_for_vision_model(block_sequence_ids=block_sequence_ids, **mask_kwargs)
else:
    # Smaller Gemma models (use_bidirectional_attention=None) or text-only inputs use standard causal masking
    causal_mask_mapping = create_masks_for_generate(**mask_kwargs)
```

两个条件。在 **E2B 和 E4B** 上第一个为假（01 章 §5），所以它们的图像 token 是完全因果的 —— 左上角的 patch 永远看不见右下角那个。而在任何尺寸上，纯文本 prompt 都走朴素路径。**同一份代码配两个 checkpoint 会产出结构上不同的 mask**，这让本章的 notebook 成为一个对照实验而不是一次演示。

另外注意结果是一个**按层型索引的 dict**，不是张量。解码器逐层查 `attention_mask[self.layer_type]`。这就是一个模型同时跑两套 mask 机制的机制。

## 设计空间

| 模型 | 图像 token 的注意力 | 施加范围 |
|---|---|---|
| **LLaVA、多数开源 VLM** | 因果 | 全网 |
| **Gemma 3** | 双向 | 所有层 |
| **Gemma 4（31B、26B-A4B）** | 双向 | 仅滑窗层 |
| **Gemma 4（E2B、E4B）** | 因果 | 全网 |
| **PaliGemma** | 双向（prefix-LM） | 所有层，图像作为前缀 |
| **Flamingo / Llama 3.2 V** | 不适用 —— 特征活在序列之外 | 交叉注意力层 |

底层问题是：那个因为文本有阅读顺序才存在的因果约束，该不该施加在没有阅读顺序的数据上。纯粹主义的答案是不该；务实的答案是双向与 KV 缓存相冲突。Prefix-LM 设计（PaliGemma）靠把图像做成永远完整在场的固定前缀来化解。Gemma 4 靠**按层型**化解 —— 一个更细粒度的答案，让可缓存的层保持因果，把双向花在便宜的地方。

E2B 干脆不启用这个特性，本身也是一个数据点：在小尺度上，简单性显然比建模收益更值钱。

## 自测

1. 既然占位符的 embedding 马上就会被覆盖，为什么还要先把它们的 id 改写成 `pad_token_id`？
2. 你手工构造输入并收到 "Image features and image tokens do not match, tokens: 280, features: 266"。你做错了什么？
3. 一个带两张图的 prompt。块 id 是什么？没有它们会出什么问题？
4. 为什么偏偏在全局层上禁用双向注意力？给出缓存方面的论证。
5. `sliding_window=1024` 下的两张 1120 token 图像。其中任何一张在块内是完全双向的吗？
6. 同一份代码，用 E2B 和 31B 跑同一个 prompt，会产出不同的 mask。哪一行决定？两种结果分别是什么？
7. `causal_mask_mapping` 是 dict 不是张量。谁消费它？为什么它必须是 dict？

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_mask_heatmap.ipynb` | 构造一个两图一音频的 prompt；把最终 4D 注意力 mask 画成热力图，读出因果文本区、双向图像块与滑动窗口；再 diff `masked_scatter` 前后的 `inputs_embeds`，亲眼看见融合发生 | 🟢 CPU（迷你 config） |
