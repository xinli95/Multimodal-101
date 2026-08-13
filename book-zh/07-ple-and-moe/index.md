# 07 · PLE 与 MoE — 花参数的两种方式

**在数据流中的位置**：每个 `Gemma4TextDecoderLayer` 内部。

两套机制，从相反方向回答同一个问题："怎么获得更多能力，又不必为每个 token 都付全价"。

**Per-Layer Embeddings (PLE)** 是不寻常的那个，也是本章存在的理由。与其在输入端放一个 embedding、让它在 30 个 block 的残差流里被小心保存，Gemma 4 给**每一个解码层**单独喂一份辅助 embedding 信号。它由两部分求和后乘以 `1/√2` 得到：

- **token identity** — 用 `input_ids` 查第二张 embedding 表 `embed_tokens_per_layer`，其权重形状为 `[vocab_size_per_layer_input, num_hidden_layers × hidden_size_per_layer_input]`：每层的切片被打包进同一行
- **context aware** — 把真实的 `inputs_embeds` 过一个线性层、乘 `1/√hidden_size`、按层 reshape，再做 RMSNorm

多模态位置上没有 `input_ids` 可查——soft token 不在词表里——所以只有 context-aware 那一半起作用。这个细节是一扇小窗，让你看见整套设计如何为非文本 token 让路。

**MoE** 则是熟悉的那个：在 26B-A4B checkpoint 里，符合条件的层把稠密 MLP 换成路由器加一组专家，每个 token 激活 `top_k_experts` 个。26B 参数，约 4B 计算量。

## 你会学到

1. PLE 两条路径从 `input_ids` 到逐层张量的完整复现，并与库核对
2. 打包成 `[vocab, num_layers × ple_dim]` 的布局为什么存在，又是怎么被 reshape 的
3. 把 PLE 置零后模型输出会发生什么——一分钟就能跑完的消融
4. MoE 路径：路由、专家分发、`top_k_experts`、`moe_intermediate_size`，以及把专家负载画出来长什么样
5. `use_double_wide_mlp` 与融合的 gate/up 投影——提醒你多数"架构"其实是布局

## 源码地图

| `modeling_gemma4.py` 中的符号 | 作用 |
|---|---|
| `Gemma4TextModel.get_per_layer_inputs` | token-identity 那一半 |
| `Gemma4TextModel.project_per_layer_inputs` | context-aware 那一半、`1/√2` 合并与多模态回退 |
| `Gemma4TextRouter`、`Gemma4TextExperts` | MoE 路径 |
| `Gemma4TextDecoderLayer.forward` | 逐层输入真正被消费的地方 |
| `Gemma4PreTrainedModel._resize_per_layer_embeddings` | 扩词表时第二张表怎么办 |

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_ple_ablation.ipynb` | 复现 PLE 两条路径并 `assert_close`；然后把逐层输入置零，看生成如何劣化——相信一个机制有用的最快方式 | 🟡 24GB（E2B） |
| `02_moe_routing.ipynb` | 随机权重的小 MoE config：追踪一个 token 走过路由器 → 专家 → 合并，再画一批数据上的专家负载 | 🟢 CPU |

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/07-ple-and-moe/index.html)。
