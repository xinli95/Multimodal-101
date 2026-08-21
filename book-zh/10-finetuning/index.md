# 10 · Fine-Tuning — 改变权重

**在数据流中的位置**：全部，反向。

Part I 的最后一章闭环。一旦你知道每个模块在做什么，"这个怎么微调"就不再是一份抄来的配方，而是一个你自己做的决定：视觉塔、音频塔、多模态 embedder、语言模型这四组参数里，为了**你的**任务哪些该动，哪些该冻。

## 你会学到

1. 一个 Gemma 4 checkpoint 的四组参数、各自的规模，以及训练每一组究竟改变了什么
2. 领域为什么收敛到"冻住塔、只调 LLM"，以及这个做法在什么情况下是错的
3. PEFT 下的 LoRA：适配器放哪，rank 与 target modules 如何与 GQA、MoE 路径相互作用
4. 写一个多模态 data collator：所有教程都跳过的部分，也是 02 章的占位符数与 03–05 章的 soft token 数必须对上、否则什么都不工作的地方
5. label 掩码：只在回答上训练，以及这里出错会如何悄悄产出一个专门预测 prompt 的模型
6. 显存账——权重、梯度、优化器状态、激活——在启动之前算清楚，而不是 OOM 之后

## 源码地图

| 符号 | 作用 |
|---|---|
| [`Gemma4PreTrainedModel`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1480) | [`_init_weights`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1501)、[`resize_token_embeddings`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1559)、[`_resize_per_layer_embeddings`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1573)（加 token 也会动 PLE 表——07 章） |
| [`Gemma4Model.vision_tower`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2191) / [`.audio_tower`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2197) / [`.embed_vision`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2198) / [`.embed_audio`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2203) / [`.language_model`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2195) | 你要冻结或解冻的四组 |
| [`Trainer`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/trainer.py#L257)、[`TrainingArguments`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/training_args.py#L179) | `transformers` 的训练循环 |
| [`peft.LoraConfig`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py#L434)、[`get_peft_model`](https://github.com/huggingface/peft/blob/main/src/peft/mapping_func.py#L105) | 适配器 |

## 源码走读

### 1. 五个组，以及各自会动什么

01 章 §4 把它们建了出来；这里说训练每一组实际改变什么：

| 组 | 属性 | 训练它改变…… | 通常结论 |
|---|---|---|---|
| 视觉塔 | `model.vision_tower` | 像素如何被*感知* | 冻结 —— 要安全地动它需要图像级规模的数据 |
| 音频塔 | `model.audio_tower` | 声音如何被感知 | 冻结 —— 同理，且更甚 |
| 视觉 embedder | `model.embed_vision` | 视觉特征*落在*文本空间的哪里 | 有时解冻 |
| 音频 embedder | `model.embed_audio` | 音频同上 | 有时解冻 |
| 语言模型 | `model.language_model` | 模型拿感知到的东西*做什么* | 常规目标 |

默认配方 —— 冻塔、调语言模型 —— 之所以有效，是因为多数下游任务要的不是"换个方式看"，而是"对看到的东西换个方式回应"。改输出格式、遵循某个领域的惯例、用固定文风作答：全在语言侧。

例外值得点名，因为那是默认配方悄悄失效的场合：

- **确实新颖的视觉领域**（医学影像、卫星、工业检测）。塔可能对它根本没有可用特征。解冻塔 —— 或至少解冻 `embed_vision` —— 是站得住的，但学习率要比 LLM 低得多。
- **超出 token 预算的细节。** 如果模型读不清你的小字，那是 03 章的问题，不是训练问题。先调高 `max_soft_tokens` 再考虑动权重。
- **E2B/E4B 上的音频。** 音频塔是模型里数值上最娇气的部分（05 章 —— clipped linear、梯度截断、float32 注意力）。最后才解冻它，并盯紧不稳定性。

### 2. LoRA 的 target modules，以及 Gemma 4 的麻烦

PEFT 常规起手式：

```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

**这个配置在 Gemma 4 上会直接崩。** 按名字匹配会同时命中视觉塔和音频塔里的同名模块，而那些是 `Gemma4ClippableLinear` 包装器（04 章 §3）不是 `nn.Linear`，PEFT 包不了，抛 `ValueError`。修法是用正则把范围钉死在语言模型上：

```python
target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
```

修好之后还有三件事会让你意外。

**很多层没有 `k_proj` 或 `v_proj`。** 在 E2B 上第 15–34 层是 KV 共享的（06 章 §4），根本不拥有那些模块。实测下来 adapter 数是 `q_proj` 35 个、`o_proj` 35 个，而 `k_proj` 和 `v_proj` **各只有 15 个**。这不是 bug —— 你没法适配一个不存在的投影 —— 但 `print_trainable_parameters()` 会报出一个比你预期小的数字，而且不会解释。

**在 `attention_k_eq_v` 下（31B、26B-A4B）全局层也没有 `v_proj`。** 同样的道理。

**MLP 宽度不统一。** 06 章 §5：E2B 上 KV 共享的那些层带双宽 MLP。一个 rank-16 的 adapter 加在 12288 宽的投影上，按比例是比加在 6144 宽上**更小**的干预。想要均匀的适配压力就得自己动脑；多数人不必操心，但值得知道这些层并不等价。

如果你也想让 connector 动起来，按名字加上 `embedding_projection`（`Gemma4MultimodalEmbedder` 唯一的 Linear，04 章 §5）。那个无 bias 的单线性层是让视觉特征重新落到更有用位置的最廉价方式，而当冻结的塔*几乎*够用时，它常常就够了。

**加 token 要付双份。** 如果你的任务需要新特殊 token，`resize_token_embeddings` 在 PLE 尺寸上还必须扩 `embed_tokens_per_layer`（07 章 §5）。新行宽是 `num_layers × 256`、随机初始化，因此需要训练。

### 3. Collator，一切汇合的地方

这是教程都会跳过的部分，而它是前面每一章同时现身的地方。四条要求，每条都能追溯到某一章：

1. **`tokenize=False`，然后让 processor 去 tokenize。** 模板只发出一个 `<|image|>`；只有 processor 在看过图像之后才知道该展开多少（02 章 §2）。在模板阶段就 tokenize，你会得到一个占位符，而模型期待几百个。
2. **嵌套的图像列表。** `validate_inputs` 逐样本比对数量（02 章 §4）。跨 batch 的扁平列表会抛错。
3. **把每个多模态占位符从 label 里屏蔽掉**（设成 `-100`）。训练模型去预测 `<|image|>` token 毫无意义且有害 —— 而且它是静默的，因为 loss 照样会下降。
4. **padding 方向要有意识。** 生成用左 padding；**训练**用右 padding 是惯例也没问题，因为没有解码循环。

```python
labels = batch["input_ids"].clone()
labels[labels == processor.tokenizer.pad_token_id] = -100
for tok_id in (config.image_token_id, config.audio_token_id, config.video_token_id,
               config.boi_token_id, config.eoi_token_id):
    labels[labels == tok_id] = -100
batch["labels"] = labels
```

### 4. 只在回答上训练

指令微调最常见的错误：屏蔽了 pad 却没屏蔽 prompt，于是模型把大部分梯度花在学习如何生成*问题*上。

Gemma 4 的轮次语法让这件事可做：模型自己的内容永远从一个 `<|turn>model\n` 标记之后开始（02 章 §5）。定位最后一个，把它之前的一切设成 `-100`。

启动任何训练之前先做次检查：把 `labels != -100` 的位置解码出来读一遍。它们应当恰好是 assistant 的回复，别无其他。这花三十秒，能挡掉本章里的多数错误。

### 5. 显存账

在启动前算，别在 OOM 之后。E2B、bf16：

| 项 | 估计 |
|---|---|
| 权重 | 约 10GB（checkpoint 是 10.25GB） |
| LoRA adapter | 几十 MB |
| 梯度 | 只有 adapter —— 几十 MB |
| 优化器状态（AdamW，两个 fp32 动量） | 约 adapter 的 8 倍，仍然很小 |
| **激活** | **真正的变量** |

用 LoRA 时冻结权重占主导，所有可训练的东西相比之下都是噪声 —— 这正是重点。激活才是你能控制的：

- **序列长度由图像主导。** 一张图 280 token（03 章）；四张图加一段文字轻松超过 1,000。要紧的数是 batch size × sequence length，不是 batch size。
- **`gradient_checkpointing=True`** 用约 30% 更多计算换取大量激活内存。解码层本来就是 `GradientCheckpointingLayer` 子类，一个开关的事。
- **视觉塔即使被冻结也每步都跑。** 没有梯度，但前向和它的激活是真实的。如果你的数据集重复使用同一批图像，预计算 soft token 是真正的优化 —— 前提是注意 09 章关于同时预计算 `per_layer_inputs` 的提醒。

2×24GB 的配置在短序列下跑 E2B LoRA 很从容。对语言模型做全量微调放不下；那是 40GB+ 的活。

两个 `Trainer` 陷阱，都不会报错：

- **`remove_unused_columns=False` 不是可选项。** `Trainer` 会检查 `forward` 的签名并丢弃它不认识的数据集列；当你的自定义 collator 产出 `pixel_values` 和 `image_position_ids` 时，默认行为会悄悄把它们删掉，于是你在一堆塞满 pad embedding 的 prompt 上训练了一个纯文本模型。
- **`Trainer` 在看到多于一张 GPU 时会自动启用 DataParallel**，把整个模型复制到每张卡上，24GB 的板子直接 OOM。用 `CUDA_VISIBLE_DEVICES=0` 钉死单卡，或显式配置分布式。

## 设计空间

**该调什么**在整个领域收敛得出奇地一致：

| 策略 | 成本 | 何时用 |
|---|---|---|
| Prompt / few-shot | 免费 | 永远先试这个 |
| 对 LLM 做 LoRA | 数小时，单卡 | 行为改变的默认选项 |
| LoRA + connector | 同上 | 冻结的塔已接近，但特征落点不好 |
| 全量微调 LLM | 40GB+ | 领域漂移大、数据多 |
| 微调塔 | 昂贵、不稳定 | 确实新颖的视觉领域 |
| 继续预训练 | 工业级 | 你不会做这个 |

**领域内的分歧**在于多阶段配方。经典的 LLaVA 三阶段序列 —— 对齐 connector、然后全解冻做多任务预训练、再 SFT —— 假设你在*建造*一个 VLM。适配一个已有的是另一个问题，答案通常是"只做最后一阶段"。Gemma 4 自己的指令微调 checkpoint 已经走完了整套流程；你是在编辑它的末端，不是重跑一遍。

一个 Gemma 4 特有的告诫：因为这个模型的参数预算有很大一部分住在不寻常的地方 —— E2B 上 23.5 亿参数的 PLE 表、KV 共享的上层、不统一的 MLP 宽度 —— 从别的模型迁移过来的 LoRA 参数量直觉在这里很不准。把 trainable parameters 打出来看看，别假设你惯用的设置在这里意味着同样的东西。

## 自测

1. 你用 `target_modules=["q_proj","k_proj","v_proj","o_proj"]` 对 E2B 做 LoRA，先是崩了，改成正则之后 adapter 数远少于预期。分别为什么？
2. 你微调后的模型开始生成问题而不是答案。collator 做错了什么？
3. 为什么必须把多模态占位符 token 在 label 里设成 `-100`？为什么忘了很难察觉？
4. `remove_unused_columns` 默认是 `True`。具体什么坏了？为什么没有异常？
5. 你给 E2B 加了三个特殊 token。哪些表会变大？新行宽是多少？
6. 视觉塔是冻结的。为什么它仍然影响峰值内存？你能对此做什么？
7. 模型读不清你扫描件里的小字。给出两个候选修法，并说明你会按什么顺序尝试。

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_lora_sft.ipynb` | 在 E2B 上端到端做一次小规模图像指令 LoRA SFT：数据集、collator、label 掩码、训练过程，以及在留出样本上的前后对比 | 🟡 24GB 显存 |
