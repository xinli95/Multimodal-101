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

## 源码走读 —— PLE

### 1. 它是一个门，不是一次加法

文档把 PLE 描述成"注入每个解码层的辅助残差信号"，没错，但低估了它。下面是一层真正拿它做的事，位于 `Gemma4TextDecoderLayer.forward` 的最末尾：

```python
if self.hidden_size_per_layer_input:
    residual = hidden_states
    hidden_states = self.per_layer_input_gate(hidden_states)   # hidden_size -> 256
    hidden_states = self.act_fn(hidden_states)                 # gelu_pytorch_tanh
    hidden_states = hidden_states * per_layer_input            # <- 逐元素门控
    hidden_states = self.per_layer_projection(hidden_states)   # 256 -> hidden_size
    hidden_states = self.post_per_layer_input_norm(hidden_states)
    hidden_states = residual + hidden_states
```

这一层把自己的隐状态挤压到 256 维、激活、**与逐层 embedding 逐元素相乘**、投影回去、归一化、再相加。逐层 embedding 不是被加进流里的 —— 它*调制的是这条流的一个低秩视图*。

这个区别很重要。加性信号会不管该层算出了什么都注入同样的内容。乘性门则让 embedding 说出*"对这个 token、在这一层，放大这 256 个方向、压制那些方向"*。这是一条逐 token、逐层的学习式条件通道，代价是每层两个瘦投影。

### 2. token-identity 那一半

```python
self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
    config.vocab_size_per_layer_input,                                   # 262144
    config.num_hidden_layers * config.hidden_size_per_layer_input,       # 35 * 256 = 8960
    self.padding_idx,
    embed_scale=config.hidden_size_per_layer_input**0.5,                 # sqrt(256) = 16
)
```

一张**第二份全词表 embedding 表**，其中每个 token 的那一行把它对全部 35 层的贡献首尾相接打包在一起。查一次表、reshape，完事：

```python
return self.embed_tokens_per_layer(input_ids).reshape(
    *input_ids.shape, self.config.num_hidden_layers, self.hidden_size_per_layer_input)
```

这个打包布局不是顺手为之 —— 它把 35 次独立的 embedding 查表变成一次 gather。在 E2B 上这张表是 262144 × 8960 ≈ 23.5 亿参数，比模型其余部分加起来还大（实测占 checkpoint 的 **46%**）。这正是"E2B"里那个*有效* 2B 的由来：表很大，但每个 token 只会碰到其中一行，所以它可以住在更慢的存储里，而计算密集的权重常驻。这也是 PLE 属于小尺寸的原因（01 章 §5）：它买来的逐 token 容量花的是内存而不是 FLOPs，这恰是端侧模型想要的交易，而 31B 服务器模型不想要。

### 3. context-aware 那一半

```python
per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
per_layer_projection = per_layer_projection.reshape(*inputs_embeds.shape[:-1],
                                                    self.config.num_hidden_layers,
                                                    self.hidden_size_per_layer_input)
per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

if per_layer_inputs is None:
    return per_layer_projection

return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale   # 2**-0.5
```

一个 `nn.Linear(hidden_size, num_layers * ple_dim)` 作用在输入 embedding 上，乘 `1/√hidden_size`，按层 reshape，再 RMSNorm。然后两半相加、乘 `1/√2` —— 两个近似单位方差信号的标准方差保持组合。

注意两半各自知道什么。token-identity 那半只取决于*这是哪个 token*。context-aware 那半取决于 `inputs_embeds`，而对一个多模态位置来说那是**来自视觉或音频塔的 soft token**。所以图像位置的门是由图像内容算出来的。

### 4. 多模态分支，以及一个确实古怪的回退

```python
if per_layer_inputs is None:
    return per_layer_projection
```

Soft token 没有 `input_ids` —— 视觉 token 不在词表里 —— 所以 `embed_tokens_per_layer` 里没什么可查。多模态位置只拿到 context-aware 那一半。08 章会展示 `Gemma4Model.forward` 如何用 pad embedding 顶替多模态跨度来拼出逐层输入张量。

然后是这个，出现在你用 `inputs_embeds` 但不带 `input_ids` 调模型时的 `get_per_layer_inputs` 里：

```python
input_ids = ((inputs_embeds[:, :, None, :]
              == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5)
             .all(dim=3).nonzero()[:, 2])
```

它靠**对整个词表做暴力精确匹配来反解 embedding**，从而恢复 token id。一次 `[batch, seq, 262144, hidden]` 的广播比较 —— 外加一个 `try/except`，在你的 embedding 不是逐比特等于表中某行时告诉你哪里错了。`forward` 的 docstring 直接叫你别走这条路：预先算好 `per_layer_inputs` 传进去。这很好地说明了第二条并行 embedding 通路的代价：每一个假设了"embedding 就是模型入口"的 API 都需要一个特例。

两个守卫强制这份契约：

```python
if (input_ids is None) ^ (inputs_embeds is not None): raise ValueError(...)
if input_ids is not None and per_layer_inputs is not None: raise ValueError(...)
```

### 5. 扩词表要动两张表

`Gemma4PreTrainedModel` 在常规的 `resize_token_embeddings` 旁边还带了 `_resize_per_layer_embeddings`。给一个 PLE 模型加特殊 token，你必须同时扩 `embed_tokens` **和** `embed_tokens_per_layer`，后者的行宽是 `num_layers × 256`。10 章里如果你在微调时加 token，这一条就相关。

## 源码走读 —— MoE

### 6. 稠密与稀疏，相加

`enable_moe_block` 只在 26B-A4B 上为 `True`。意外之处在于 MoE 块并没有*取代*稠密 MLP：

```python
residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)                        # 稠密 MLP，始终运行

if self.enable_moe_block:
    hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

    hidden_states_flat = residual.reshape(-1, residual.shape[-1])   # 注意：MLP 之前的状态
    _, top_k_weights, top_k_index = self.router(hidden_states_flat)
    hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
    hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
    hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2.reshape(residual.shape))

    hidden_states = hidden_states_1 + hidden_states_2            # 稠密 + 稀疏
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

稠密 MLP 是一个**共享专家**，每个 token 都能拿到；被路由的专家在其上追加特化。这是 DeepSeekMoE 风格的安排，它解决了一个真实的 MoE 失效模式：纯路由时每个专家都要独立地重新学一遍公共变换，浪费容量。给所有人一条共享通路，被路由的专家才敢真正特化。

管道里有两个细节。路由器读的是 `residual` —— 该层在 pre-FFN norm 和稠密 MLP **之前**的隐状态 —— 所以路由决策基于该层的输入，而不是某个半变换的中间量。而两个分支各有自己的 norm（`post_feedforward_layernorm_1` / `_2`）再相加，因此它们的尺度可以不同。

26B-A4B 的形状：稠密通路 `intermediate_size = 2112`，每个专家 `moe_intermediate_size = 704`，`num_experts = 128`，`top_k_experts = 8`。每个 token 是 8 × 704 = 5632 的被路由宽度，加上 2112 的共享宽度 —— 而可用总量是 128 × 704 = 90,112。这就是 26B/4B 的比值。

### 7. 路由器

```python
hidden_states = self.norm(hidden_states)                       # RMSNorm, with_scale=False
hidden_states = hidden_states * self.scale * self.scalar_root_size   # 学习式逐维缩放，再乘 1/sqrt(hidden)
expert_scores = self.proj(hidden_states)                       # [tokens, 128]
router_probabilities = F.softmax(expert_scores, dim=-1, dtype=torch.float32)
top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.config.top_k_experts, dim=-1)
top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)       # 对留下的 8 个重新归一化
top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
```

softmax → top-k → 重新归一化是标准配方。两处 Gemma 特有的手笔：一个无参数的 RMSNorm 后跟一个**学习式逐维** `scale` 再做投影（路由器可以自己决定哪些特征值得用来路由），以及在重新归一化之后施加的学习式 `per_expert_scale` —— 一个逐专家增益，模型可以用它压低那些被系统性过度选中的专家。注意这里没有辅助负载均衡损失；均衡活在训练里，不在推理里。

### 8. 专家分发

```python
expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

for expert_idx in expert_hit:
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
    current_state = hidden_states[token_idx]
    gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
    current_hidden_states = self.act_fn(gate) * up
    current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
    final_hidden_states.index_add_(0, token_idx, current_hidden_states)
```

专家权重被存成 **3D 张量** —— `gate_up_proj` 是 `[num_experts, 2*intermediate, hidden]` —— 而不是 128 个独立的 `nn.Linear` 模块。正是这一点让分布式后端可以对所有专家一次做 grouped GEMM（见 01 章 `base_model_ep_plan` 里的 `"grouped_gemm"`）。

这份参考实现只在实际被选中的专家上循环（`expert_hit`），聚集它们的 token、跑一次 SwiGLU、按路由权重缩放、再 scatter-add 回去。可读且慢；类上的 `@use_experts_implementation` 会在有融合 kernel 时换掉它。读这一版是为了理解 MoE，别在生产里跑它。

## 设计空间

**PLE** 相当独特。最近的亲戚：

- **Adapter / LoRA** 同样注入低秩的逐层信号，但那个信号是*按任务*学出来的、对每个 token 都相同。PLE 的是逐 token 的、来自一张表、在预训练中学到的。
- **Prefix tuning / prompt tuning** 加逐层的可学向量，同样与 token 无关。
- **Gemma 3n 的 PLE** 是直系祖先；Gemma 4 出于同样理由继承它（端侧推理，内存带宽比 FLOPs 更要紧）。
- **Mixture-of-depths / early-exit** 也让逐 token 计算量变化，但方式是跳过层而不是给层加条件。

思考 PLE 的正确方式是把它看成**一个非常大、非常稀疏的参数仓库，花的是内存而不是计算** —— 因此它的价值完全取决于你的部署环境。在闪存慢、NPU 小的手机上，拿 FLOPs 换一次查表是赢的。在批量服务的 H100 上，那张表只是需要搬运的重量，这也正是 31B 把它设成 0 的原因。

相比之下 **MoE** 是走熟的路。Gemma 4 这版 —— 共享稠密专家加 128 个细粒度被路由专家、top-8 —— 本质上是 DeepSeekMoE 的配方，与 Mixtral 粗粒度的 8 专家 top-2 恰成对照。细粒度路由给每个 token 带来组合意义上多得多的专家*组合*（C(128,8) 比 C(8,2) 大到天文数字），代价是更多路由开销和更难的负载均衡。

放在一起读，两套机制是同一个想法在不同尺度上的体现：**大多数参数不该为大多数 token 运行。** PLE 用一张按 token 身份索引的 embedding 表实现它；MoE 用一个学习出来的路由器实现它。Gemma 4 在内存是瓶颈的地方用前者，在计算是瓶颈的地方用后者。

## 自测

1. PLE 被描述成残差信号。它在层里实际施加在哪？为什么"门"比"加"是更好的词？
2. `embed_tokens_per_layer` 比 E2B 其余部分加起来还大。为什么这可以接受？这对它该存放在哪意味着什么？
3. 一个图像 soft token 抵达某个解码层。它的逐层输入里哪一半在、哪一半缺？
4. 为什么 `get_per_layer_inputs` 里会有反解 embedding 表的代码？怎样避免触发它？
5. MoE 块里路由器读的是 `residual` 而不是 MLP 之后的隐状态。这为什么重要？
6. 26B-A4B：128 专家、top-8、`moe_intermediate_size=704`、稠密 `intermediate_size=2112`。一个 token 实际见到多少 FFN 宽度？占可用宽度的几分之几？
7. 为什么给 E2B 加一个特殊 token 需要扩两张 embedding 表？

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_ple_ablation.ipynb` | 复现 PLE 两条路径并 `assert_close`；然后把逐层输入置零，看生成如何劣化——相信一个机制有用的最快方式 | 🟡 24GB（E2B） |
| `02_moe_routing.ipynb` | 随机权重的小 MoE config：追踪一个 token 走过路由器 → 专家 → 合并，再画一批数据上的专家负载 | 🟢 CPU |
