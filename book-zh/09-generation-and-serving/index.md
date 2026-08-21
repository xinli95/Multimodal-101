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

## 源码地图

| 符号 | 作用 |
|---|---|
| [`Gemma4ForConditionalGeneration`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2507) | `Gemma4Model` + `lm_head` + `GenerationMixin` |
| [`Gemma4ForCausalLM`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L1825) | 纯文本负载使用的纯文本路径 |
| [`Gemma4CausalLMOutputWithPast`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L109) | logits、cache 与多模态附加输出 |
| [`GenerationMixin.generate`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/generation/utils.py#L2274) | 不属于 Gemma，但值得完整读一次 |

## 源码走读

### 1. 三个 `Auto*` 类，一个 checkpoint

Gemma 4 文档的三个例子各用了一个不同的类，看起来随意，直到你知道它们分别映射到什么：

| 例子 | 类 | 你拿到什么 |
|---|---|---|
| 图像 + 文本 | `AutoModelForImageTextToText` | `Gemma4ForConditionalGeneration` |
| 函数调用（纯文本） | `AutoModelForCausalLM` | `Gemma4ForCausalLM` |
| 音频 | `AutoModelForMultimodalLM` | `Gemma4ForConditionalGeneration` |

两个多模态别名解析到同一个类；它们存在是为了让面向任务的代码表达意图。不同的是 `AutoModelForCausalLM` —— 它给你 `Gemma4ForCausalLM`，那个**纯文本**模型，没有视觉塔也没有音频塔。如果你的负载是纯文本，加载那个类就不会实例化你永远不会调用的塔。

两个 head 的差别只在包了什么：

```python
class Gemma4ForCausalLM(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

class Gemma4ForConditionalGeneration(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
```

注意权重绑定的路径：多模态模型深一层，因为它的 `Gemma4Model` 把语言模型当作子模块持有。`lm_head` 全程与输入 embedding 绑定 —— 262,144 的词表在 1536 宽度下约 4 亿参数，不必付两次。

### 2. Logits、softcap 与 `logits_to_keep`

```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
if self.config.final_logit_softcapping is not None:
    logits = torch.tanh(logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
```

`logits_to_keep` 是生成时最要紧的那个小优化。把一条 128K 的完整序列投影到 262,144 宽的词表，会产出几十 GB 的张量；而解码时你只需要一个位置。`generate` 会替你传 `logits_to_keep=1`。值得知道它的存在，因为如果你在循环里直接调 `forward` 并纳闷显存去哪了，答案就是这个。

Softcap（全尺寸 30.0）把每个 logit 平滑地约束进 ±30，所以没有任何单个 token 的分数能跑飞。它会略微压平分布，这与采样温度相互作用 —— 也是 Gemma 发布的 `generation_config.json` 长成这样的原因之一：

```json
{"do_sample": true, "temperature": 1.0, "top_k": 64, "top_p": 0.95,
 "eos_token_id": [1, 106, 50], "pad_token_id": 0, "bos_token_id": 2}
```

**采样默认是开的**，而且有**三个** EOS id。`generate` 遇到任何一个都会停。想要确定性就显式传 `do_sample=False`，别假设默认是贪心。

### 3. Cache，以及 Gemma 4 的为何不寻常

库的 `Cache` 对象逐层持有 key 和 value。Gemma 4 在两个方面把它复杂化了，而这两点你在 06 章都见过：

- **滑窗层不需要全长 cache。** 一个只向后 attend 512 token 的层可以丢掉更早的一切。
- **KV 共享层什么都不存。** 它们读的是随前向传递的 dict `shared_kv_states`，而不是 cache：

  ```python
  if past_key_values is not None and not self.is_kv_shared_layer:
      key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
  ```

  而 `Gemma4CausalLMOutputWithPast` 正是为此在 `past_key_values` 之外单独携带 `shared_kv_states`。

这两点合起来的效果是可以量化的，本章 notebook 在真实 E2B 权重上测了：**滑窗层的 cache 从 512 token 起饱和在 6.28MB 不再增长**，而全局层线性增长；全局层占比从 128 token 的 33% 涨到 8192 token 的 **89%**。长上下文的账全在那 7 个全局层上。

`cache_implementation="static"` —— 文档第一个例子用的那个 —— 预分配固定大小的 cache。固定形状正是 `torch.compile` 避免重编译所需要的。但要**诚实地**看它的成本：实测首次调用 109 秒（编译），之后 0.75 秒，对比 dynamic 的 5.15 秒 —— 大约**25 次同形状生成才回本**。它是 `torch.compile` 的使能条件，不是一个打开就变快的开关。

### 4. 批量多模态生成

02 章讲过 `padding_side="left"`。实践中还有两件事会咬人。

**自己把 prompt 切掉：**

```python
input_len = inputs["input_ids"].shape[-1]
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0][input_len:], skip_special_tokens=False))
```

`generate` 永远返回 prompt + 补全。

**`skip_special_tokens` 是一个决定，不是默认值。** 设 `True` 你得到干净散文。设 `False` 你能看见 `<|turn>`、`<|channel>thought`、`<|tool_call>` 之类 —— 而这是调试工具调用或检查 thinking 输出的唯一途径。

批量本身几乎免费：实测 batch 1→16 吞吐从 12.6 涨到 193.5 tok/s（**15.4×**），而墙钟时间几乎不动（2.55s→2.65s）。解码是显存带宽瓶颈 —— 你已经在为搬运权重付钱了，多几条序列基本是搭便车。这也是服务引擎拼命做批处理的原因，以及你的单用户延迟为什么是吞吐的糟糕预测指标。

### 5. Thinking

02 章 §5 展示过 `enable_thinking` 是 chat template 的参数，不是生成标志：

```python
inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                       add_generation_prompt=True, enable_thinking=True)
```

它往 system 轮里注入 `<|think|>`，模型则以在答案前发出一段 `<|channel>thought … <channel|>` 作为回应。对服务的影响：

- Thinking token 是**被生成的 token** —— 它们花延迟、花钱，并计入 `max_new_tokens`。设这个预算时要把 thinking 算进去，否则你会得到被截断的推理和没有答案。
- 模板会从历史里剥掉此前的 thinking（`strip_thinking`），所以多轮对话不会累积它。如果你自己管理历史，请照做。
- 想展示"思考/答案"分栏，就用 `skip_special_tokens=False` 生成并按 channel 标记切分。

### 6. 用 vLLM 服务

vLLM 已支持 Gemma 4 —— `Gemma4ForConditionalGeneration` 在它的多模态模型注册表里。标准启动方式：

```bash
vllm serve google/gemma-4-E2B-it \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 4, "audio": 2}'
```

然后像调 OpenAI API 一样调它：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "google/gemma-4-E2B-it",
       "messages": [{"role": "user", "content": [
         {"type": "image_url", "image_url": {"url": "https://…/cat.jpg"}},
         {"type": "text", "text": "What is in this image?"}]}]}'
```

两个 Gemma 4 特有的提醒。`--max-model-len` 比通常更要紧：滑窗/全局的混合让长上下文比全局模型便宜，但全局层依然主导，所以调高它并不免费。而 `--limit-mm-per-prompt` 值得刻意设置 —— 03 章告诉你每张图 280 个 token，05 章告诉你一段视频 2,240 个，所以多模态上限**就是**上下文上限。

如果你在 Vast.ai 的 vLLM 镜像上，服务已经作为 supervisor service 接好了；设 `VLLM_MODEL` 和 `MODEL_NAME` 后重启即可，不要另起一份。

## 设计空间

这里有意思的对比不在推理引擎之间，而在**多模态预处理发生在哪里**。

- **`transformers` + `generate`**：processor 跑在你的 Python 进程里，塔跑在 `forward` 内部。简单、可调试，而且是唯一能做本书这类 hook 的地方。吞吐不是目标。
- **vLLM**：processor 仍按请求运行，但塔跑在引擎内部，soft token 由 `_merge_multimodal_embeddings` 并入序列 —— 正是 02 章 docstring 警告必须与 processor 的 token 计数一致的那个函数。连续批处理和 paged attention 让它成为服务的正确答案。
- **预计算 embedding**：塔只跑一次，缓存 soft token，然后喂 `inputs_embeds`。对同一份文档的重复查询很有吸引力 —— 而在 Gemma 4 上，这也正是你必须**同时**预计算 `per_layer_inputs`（07 章 §4）否则要付一次 embedding 反解的场合。

最后那一行很好地体现了 Part I 的主题：一个为端侧效率做的架构选择（PLE），在两百页之后作为对某项服务优化的约束再次出现。只有读完整条路径，才看得见这样的联系。

## 自测

1. 你用一个 Gemma 4 checkpoint 服务纯文本流量。该加载哪个 `Auto*` 类？省下什么？
2. `logits_to_keep` 防住了什么？没有它的话，32K 上下文下那个张量大约多大？
3. 滑窗层为什么让长上下文服务更便宜？哪些层仍然主导账单？
4. 你没传任何采样参数，输出却每次都不同。为什么？
5. `enable_thinking=True` 且 `max_new_tokens=100`。最可能的失效模式是什么？
6. 你想跨多个问题缓存一份长文档的 soft token。除了 `inputs_embeds`，你还必须缓存什么？

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_generate_and_cache.ipynb` | 在真权重 E2B 上测量 prefill 与 decode、动态 vs 静态 cache、批量大小与吞吐；再加流式示例与 thinking 预算对比 | 🟡 24GB 显存 |

部署以命令行形式写在本章正文里，而不是塞进 notebook——服务不该活在 cell 里。
