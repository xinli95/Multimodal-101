# 09 · Generation and Serving — 从 `forward` 到一个跑着的服务

**在数据流中的位置**：`hidden ──► lm_head ──► logits ──► generate() ──► 文本`，然后彻底走出 notebook。

到此为止讲的都是一次前向。生成就是把这次前向放进循环，而几乎全部工程都在于**不要重做已经做过的事**：KV cache。06 章解释了 Gemma 4 的 cache 为什么长成那样——滑窗层只留一个窗口，有些层根本不持有自己的 KV。本章是这些设计兑现的地方，也是你把 `transformers` 生成 API 学到能调试的地方。

## 你会学到

1. `Gemma4ForConditionalGeneration` 在 `Gemma4Model` 之上加了什么，以及官方文档三个例子为什么用了三个不同的 `Auto*` 类
2. `Cache` 类族：动态 cache 与 `cache_implementation="static"`，静态买到了什么（可编译的固定形状）、代价是什么
3. 批量多模态生成：为什么必须 `padding_side="left"`，以及怎样正确地切掉输入部分
4. 流式输出，以及在推理模型上控制 thinking 预算
5. 量化选项，以及它对一个挂着视觉塔和音频塔的模型意味着什么
6. 真正把它跑起来：用 vLLM 起一个 OpenAI 兼容端点并调用它

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_generate_and_cache.ipynb` | 在真权重 E2B 上测量 prefill 与 decode、动态 vs 静态 cache、批量大小与吞吐；再加流式示例与 thinking 预算对比 | 🟡 24GB 显存 |

部署以命令行形式写在本章正文里，而不是塞进 notebook——服务不该活在 cell 里。

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/09-generation-and-serving/index.html)。
