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
| `Gemma4PreTrainedModel` | `_init_weights`、`resize_token_embeddings`、`_resize_per_layer_embeddings`（加 token 也会动 PLE 表——07 章） |
| `Gemma4Model.vision_tower` / `.audio_tower` / `.embed_vision` / `.embed_audio` / `.language_model` | 你要冻结或解冻的四组 |
| `Trainer`、`TrainingArguments` | `transformers` 的训练循环 |
| `peft.LoraConfig`、`get_peft_model` | 适配器 |

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_lora_sft.ipynb` | 在 E2B 上端到端做一次小规模图像指令 LoRA SFT：数据集、collator、label 掩码、训练过程，以及在留出样本上的前后对比 | 🟡 24GB 显存 |

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/10-finetuning/index.html)。
