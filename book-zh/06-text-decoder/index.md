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

## 源码走读

### 1. 注意力排班表

01 章展示了 `layer_types` 是怎么生成的。这里是它在实践中的样子，直接取自 checkpoint：

| | E2B | 31B | 26B-A4B |
|---|---|---|---|
| 层数 | 35 | 60 | 30 |
| 全局层（`full_attention`）位置 | 4, 9, 14, 19, 24, 29, 34 | 5, 11, …, 59 | 5, 11, 17, 23, 29 |
| 比例 | 5 取 1 | 6 取 1 | 6 取 1 |
| `sliding_window` | 512 | 1024 | 1024 |

**每次前向大约六分之五只看得见一个局部窗口。** 全局注意力 —— 那个开销随上下文二次增长的部分 —— 在 E2B 上只发生七次，31B 上十次。这是 256K 上下文根本负担得起的最大单一原因。

窗口本身被定义成一个谓词而不是一个矩阵：

```python
def sliding_window_mask_function(sliding_window: tuple[int, int]) -> Callable:
    def inner_mask(batch_idx, head_idx, q_idx, kv_idx) -> bool:
        left_window_size, right_window_size = sliding_window
        dist = q_idx - kv_idx
        left_mask  = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask
    return inner_mask
```

是 `(q_idx, kv_idx)` 的函数，不是 `[seq, seq]` 张量。掩码工具会组合这些谓词，只在注意力后端确实需要时才物化成张量 —— 而 FlashAttention 从不需要，它只要一个整数窗口大小。注意这个函数在形式上是对称的：它支持右侧窗口，音频塔正是复用它做这件事（05 章），`use_bidirectional_attention` 也需要它（08 章）。

### 2. 两套注意力机制、两套 RoPE、两种 head 尺寸

```python
self.is_sliding = self.layer_type == "sliding_attention"
self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
```

**全局层拿到更大的 head**：512 而不是 256。更稀有，但单个更有表达力 —— 模型在花长程预算的地方，把钱花在质量上。

而且每种机制拿到自己的旋转嵌入。`Gemma4TextRotaryEmbedding` **按层型**各建一套 buffer：

```python
for layer_type in self.layer_types:
    rope_params = self.config.rope_parameters[layer_type]
    ...
    if layer_type == "full_attention" and rope_type == "proportional":
        rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"
    curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, **rope_init_fn_kwargs)
    self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
```

`forward` 随后按名字选取：`inv_freq = getattr(self, f"{layer_type}_inv_freq")`。承 01 章，两套设置是：

- `sliding_attention` → `rope_type="default"`，θ = 10,000
- `full_attention` → `rope_type="proportional"`，θ = 1,000,000，`partial_rotary_factor=0.25`

逻辑值得直说。一个滑窗层可能出现的最大相对距离是 512，所以经典的 θ=10,000 频率完全分辨得了，拉伸它反而浪费分辨率。一个全局层必须区分相距最多 262,144 的位置，所以它需要平坦得多的 θ=1,000,000 —— 并且只旋转它那 512 维里的四分之一，其余保持位置无关。**每一层都拿到与其跨度相称的位置分辨率。** 全网用一套 RoPE 的模型，是在过度拉伸局部层去迁就全局层。

### 3. QK-norm，以及一个没有参数的 value norm

```python
self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
self.scaling = 1.0
```

与视觉塔相同的三件套。并注意 `self.scaling = 1.0` —— 惯例中的 `1/√d` **不见了**，因为归一化 q 和 k 已经把它们的量级固定住了。那个缩放因子当初只是在补偿一个如今不再发生的增长。

`forward` 里顺序有讲究：投影 → norm → RoPE → transpose。先归一化再旋转，让旋转作用在近似单位长度的向量上。

### 4. 跨层 KV 共享

模型里最激进的内存技巧，大约十五行。

```python
first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
prev_layers = config.layer_types[:first_kv_shared_layer_idx]
self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])

if not self.is_kv_shared_layer:
    self.k_norm = ...; self.v_norm = ...
    self.k_proj = nn.Linear(...)
    self.v_proj = nn.Linear(...) if not self.use_alternative_attention else None
```

在 E2B 上，35 层中 `num_kv_shared_layers = 20`，所以第 15–34 层**完全没有 `k_proj`、`v_proj`、`k_norm`、`v_norm`**。它们读取**同类型**最后一个非共享层算出的 key 和 value：

```python
if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.layer_type]
```

`store_full_length_kv` 标记出哪些层是捐赠者 —— 共享边界之前最后一个滑窗层和最后一个全局层。两个捐赠者，每种层型一个，因为滑窗层的 key 与全局层的 key 是用不同 RoPE 旋转的，不可互换。

共享分支上的注释解释了一处否则看起来像是漏掉的优化：

> *We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache once we are past the sliding window — so we always use `shared_kv_states` instead, even when `past_key_values` is not None.*

捐赠者的 cache 可能已经忘了消费者需要的东西。正确性优先于复用。

加载器也被告知要预期这些缺失的权重，而不是对它们发警告：

```python
for i, layer in enumerate(self.layers):
    if layer.self_attn.is_kv_shared_layer:
        self._keys_to_ignore_on_load_unexpected.extend(
            [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")])
```

**K=V** 是大模型版的同一个想法：

```python
self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
...
value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
```

只在全局层上，value 投影被删掉，key 投影的输出同时充当两者。恰好为那些 cache 是全长的层把 KV cache 减半。注意它是在 norm 和 RoPE 分岔**之前**施加的 —— key 走 `k_norm` + 旋转，value 走 `v_norm` 且不旋转，所以尽管共用一个投影，两者在下游并不相同。

### 5. 让整个设计豁然开朗的那个细节

`Gemma4TextMLP.__init__`：

```python
first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
```

读两遍。**双宽 MLP 恰好施加在那些放弃了 KV 投影的层上。** 在 E2B 上，第 0–14 层有 6144 宽的 MLP 和自己的 key/value；第 15–34 层有 **12288 宽**的 MLP 和没有自己的 key/value。

这不是两个碰巧共存的技巧，而是同一个决定：*在网络上半部分，别再为注意力的参数和 cache 付钱，把省下的花在前馈容量上。* 它反映了一个关于 transformer 的真实发现 —— 上层做更多特征变换、更少信息路由 —— 而 Gemma 4 悄悄把它焊进了架构。你从 config 本身看不出这一点；`use_double_wide_mlp: true` 看起来像个全局开关。只有上面那三行才揭示它是有条件的。

### 6. 一些要紧的零碎

**Embedding 缩放。** `Gemma4TextScaledWordEmbedding` 把查表结果乘以 `√hidden_size`，注释里保存了一段机构记忆：

```python
# Gemma4 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5.
```

这个缩放被存成 buffer，以便在与原始实现相同的精度下施加。逐比特复现一个模型，意味着连它的舍入也要复现。

**`layer_scalar`。** 每个解码层以 `hidden_states *= self.layer_scalar` 结束，那是一个初始化为 1.0、从 checkpoint 加载的 `register_buffer` —— 每层一个残差缩放，代价是一次乘法。

**`final_logit_softcapping = 30.0`**，全尺寸通用，在 `Gemma4ForCausalLM` 里施加：`logits = tanh(logits / cap) * cap`。与音频塔的注意力 softcap 是同一个平滑天花板。它约束输出分布的熵，阻止任何单个 token 的 logit 跑飞。

## 设计空间

| 技术 | 谁在用 | 省什么 | 代价 |
|---|---|---|---|
| **GQA** | 几乎所有人 | KV cache ÷（heads/kv_heads） | 高比例时轻微质量损失 |
| **MQA**（`kv_heads=1`） | Gemma 4 E2B、PaLM | GQA 的最大化版本 | 更多；通常需要从头就带着它训练 |
| **滑窗 + 全局混合** | Gemma 2/3/4、Mistral、Character.ai | 多数层变成 O(n·w) | 长程信息必须经由少数全局层路由 |
| **跨层 KV 共享** | Gemma 4、CLA、YOCO | cache ÷（层数/捐赠者数） | 上层无法形成新的注意力模式 |
| **K=V** | Gemma 4 大尺寸 | 那些层的 cache ÷ 2 | key 与 value 不再能各自特化 |
| **MLA** | DeepSeek | 经低秩 latent 降 cache | 一种确实不同的注意力形式 |

Gemma 4 的贡献不是其中任何一行 —— 而是它同时用了其中四个，而且**按尺寸区别对待**。必须塞进手机的 E2B 选了 MQA 加 20 个共享层。有质量预算要保护的大模型保留 16 或 8 个 KV head，只在全局层上拿更温和的 K=V 收益。同一份代码，取舍曲线的两端，完全由 config 选择。

诚实的局限：跨层共享意味着 E2B 顶部 57% 的层无法决定*该看什么* —— 只能决定拿下层看到的东西做什么。这是否有害是个经验问题，而双宽 MLP 就是那份补偿。

## 自测

1. 滑窗层和全局层为什么需要不同的 RoPE θ？如果两者都给 θ=10,000 会坏掉什么？
2. `self.scaling = 1.0` —— `1/√d` 去哪了？
3. 为什么 `shared_kv_states` 里是两个条目而不是一个？
4. E2B：35 层、共享 20 层、1 个 KV head、head_dim 256/512。每个 token 大约多少 KV cache？没有共享的话又是多少？
5. 在 `attention_k_eq_v` 下 key 和 value 来自同一个投影。说出两件在下游仍然不同的事。
6. 哪些层拿到双宽 MLP？支持这个选择的论证是什么？

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_decoder_anatomy.ipynb` | 打印真实的 `layer_types` 模式与 head 维度分配；手写 `sliding_window_mask_function` 并与库核对；测量 KV cache 按层型的增长，量化共享究竟省了多少 | 🟢 CPU（迷你 config）/ 🟡 真实测量 |
