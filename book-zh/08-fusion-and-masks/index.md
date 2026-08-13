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
| `Gemma4Model.get_placeholder_mask` | 定位图像 / 视频 / 音频位置 |
| `Gemma4Model.forward` | `masked_scatter` 融合与 `pad_token_id` 改写 |
| `create_masks_for_vision_model` | 双向视觉 mask |
| `get_block_sequence_ids_for_mask` | 把连续的多模态段分组成块 |

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_mask_heatmap.ipynb` | 构造一个两图一音频的 prompt；把最终 4D 注意力 mask 画成热力图，读出因果文本区、双向图像块与滑动窗口；再 diff `masked_scatter` 前后的 `inputs_embeds`，亲眼看见融合发生 | 🟢 CPU（迷你 config） |

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/08-fusion-and-masks/index.html)。
