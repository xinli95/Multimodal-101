# 06 · Text Decoder — 付得起的注意力

**在数据流中的位置**：`inputs_embeds ──► Gemma4TextModel ──► hidden states`

在 128K–256K 上下文下，解码器的设计被一个约束支配：KV cache。每层都用全尺寸头做全注意力是负担不起的，所以 Gemma 4 把预算花得极不均匀且非常刻意。多数层只看 512 token 的滑动窗口；少数层拿到全局注意力，而且头**更大**（`global_head_dim=512` 对 `head_dim=256`）。最后几层干脆不拥有自己的 KV 投影——它们复用更早一层的。某些全局层连独立的 value 投影都没有，直接把 key 投影的输出当 value 用。

这些每一条都是质量与显存之间的具体交易，而且都只有几行可读的代码。

## 你会学到

1. `layer_types` 如何把每层指派为 `sliding_attention` 或 `full_attention`，以及怎么从真实 checkpoint 上把这个模式读出来
2. 滑窗层为什么需要跟全局层**不同**的 RoPE，`Gemma4TextRotaryEmbedding` 怎么同时伺候两者
3. QK-norm：对 query、key **和** value 逐头做 RMSNorm（含无缩放的 value norm），它防住了什么不稳定
4. 跨层 KV 共享（`num_kv_shared_layers`）与 K=V 投影（`attention_k_eq_v`）：省了什么，代价是什么
5. `final_logit_softcapping` 与 `Gemma4TextScaledWordEmbedding`——让训练稳住的数值细节
6. 自己量一遍 KV cache，用兆字节而不是形容词来看这些节省

## 源码地图

| `modeling_gemma4.py` 中的符号 | 作用 |
|---|---|
| `Gemma4TextAttention` | GQA + q/k/v RMSNorm、滑窗与全局分支、KV 共享、K=V |
| `Gemma4TextRotaryEmbedding` | 按层型区分的 RoPE |
| `sliding_window_mask_function` | 把窗口写成 `(q_idx, kv_idx)` 上的布尔谓词 |
| `Gemma4TextMLP`、`Gemma4TextDecoderLayer` | 基本块 |
| `Gemma4TextModel.forward` | 主循环、共享 KV 字典与 mask 构造 |
| `repeat_kv`、`eager_attention_forward`、`apply_rotary_pos_emb` | 读融合 kernel 之前先读的参考实现 |

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_decoder_anatomy.ipynb` | 打印真实的 `layer_types` 模式与 head 维度分配；手写 `sliding_window_mask_function` 并与库核对；测量 KV cache 按层型的增长，量化共享究竟省了多少 | 🟢 CPU（迷你 config）/ 🟡 真实测量 |

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/06-text-decoder/index.html)。
