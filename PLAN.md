# 重构计划 · 以 Gemma 4 为主线

> 本文件是这次重构的施工图与进度记录，随工作推进更新。

## 背景

重构前这本书是「按模态/任务横向铺开」的九章 survey：每章 `overview.md`(16 行) + `theory.md`(~40 行) + `landscape.md` + 零散 notebook。问题在于——它是一张**清单**而不是一条**路径**：读者读完知道有哪些模型，但不知道一个多模态模型内部到底长什么样、每个部件怎么实现、`transformers` 里这些东西写在哪。theory 页太薄（40 行讲不动架构），notebook 之间也没有连续性。

重构目标：**用 Google Gemma 4 这一个具体模型作为纵向主线**，从 config → tokenizer → image/audio processor → vision tower → audio tower → text decoder(PLE/MoE/混合注意力) → 多模态融合与 mask → generate/部署 → 微调，逐层解剖它在 `transformers` 里的真实实现，顺带把 `transformers` 的用法（Auto* 类、Processor 体系、Config 组装、Cache、Trainer/PEFT）学透。

Gemma 4 是理想教具：一个开源 checkpoint 家族同时覆盖文本/图像/视频/音频输入，且包含固定 token 预算下的可变长宽比、2D RoPE、Per-Layer Embeddings、滑窗+全局混合注意力、跨层 KV 共享、MoE、128K–256K 上下文；而 E2B 尺寸小到单张消费级显卡能跑。

## 四个已定决策

1. **两部分结构**：Part I 新主线（Gemma 4 解剖，11 章），Part II 保留并压缩生成类章节（5 章）。
2. **深度：源码级 + 手写复现**——逐段读本机 `transformers` 的 `models/gemma4/*.py`，notebook 用 hook 打印中间张量，并手写复现关键模块与官方输出 `assert_close` 对拍。
3. **双语：先英文**（`book/` 为唯一源），中文 `book-zh/` 随后补。
4. **硬件基线：E2B 真权重为主 + 随机权重兜底**——纯架构解剖类 notebook 用 `Gemma4Config` 造极小随机权重模型，CPU 可跑、零下载；能力展示类用 `google/gemma-4-E2B-it`（10.25GB，未 gated，单卡 24GB 够）。

## 版本基线

Part I 基于 **transformers 5.14.1** 写成。5.15.0 的 `models/gemma4/` 与之只有表层差异（`register_buffer` → `nn.Buffer`、新增 `@deprecate_kwarg`；`image_processing_gemma4.py` 逐字节相同），架构与书中所有事实结论不变。一个例外：5.15 起 gemma4 config 走 heterogeneous per-layer 路径，`config.text_config.num_key_value_heads` 这类读法会抛 `AmbiguousGlobalPerLayerAttributeError`。因此凡是要陈述某个 checkpoint 的事实，notebook 一律直接读发布的 `config.json`，不经过 `AutoConfig`。

## 章节结构

### Part I · Gemma 4 解剖（11 章）

每章统一为 `index.md`（导读 → 源码走读 → 「设计空间」对比 → 自测题）+ `notebooks/`。取消了旧的 `overview/theory/landscape` 三件套——走读本身就是主体。「设计空间」小节负责装回原 00/01/02 章的谱系知识（CLIP→BLIP-2→LLaVA→Qwen3-VL/InternVL），保证不是只见一棵树。

| 章 | 目录 | 主要源码 |
|---|---|---|
| 00 | `00-orientation/` | 为什么只讲一个模型；家族全景；数据流地图；CLIP→今天的前史 |
| 01 | `01-config/` | `configuration_gemma4.py`、`Gemma4Model.__init__` |
| 02 | `02-text-io/` | `processing_gemma4.py`、`chat_template.jinja` |
| 03 | `03-image-processor/` | `image_processing_gemma4.py` |
| 04 | `04-vision-tower/` | `Gemma4VisionPatchEmbedder/Pooler/RotaryEmbedding`、`Gemma4MultimodalEmbedder` |
| 05 | `05-audio-and-video/` | `Gemma4AudioModel`、`feature_extraction_gemma4.py`、`video_processing_gemma4.py` |
| 06 | `06-text-decoder/` | `Gemma4TextAttention`、`Gemma4TextRotaryEmbedding`、`sliding_window_mask_function` |
| 07 | `07-ple-and-moe/` | `get_per_layer_inputs`、`project_per_layer_inputs`、`Gemma4TextRouter/Experts` |
| 08 | `08-fusion-and-masks/` | `get_placeholder_mask`、`create_masks_for_vision_model` |
| 09 | `09-generation-and-serving/` | `Gemma4ForConditionalGeneration`、Cache、vLLM |
| 10 | `10-finetuning/` | `Trainer`、`peft`、多模态 collator |
| — | `landscape.md` | 理解侧格局总览（合并原 00/01/02 的 landscape） |

### Part II · 生成篇（5 章）

保留 `overview/theory/landscape/notebooks` 三件套，每章开头加「与 Part I 的关系」把术语接上。

| 新目录 | 来源 |
|---|---|
| `20-image-generation/` | 原 `03-image-generation` + `04-image-editing` 合并 |
| `21-video-generation/` | 原 `05-video-generation` |
| `22-audio-generation/` | 原 `06-audio` 的生成部分（ASR 移到 Part I ch05 作对照组） |
| `23-unified-omni/` | 原 `07-unified-omni` |
| `24-applications/` | 原 `08-applications` |

## 进度

- [x] **Batch 1 · 骨架**（`c8f1fc1`）：目录重组、`_toc.yml` / `intro.md` / `README.md` 重写、`landscape.md` 新建、Part II 迁移与合并、中英两版 build 干净。
- [x] **Batch 2 · Part I 走读正文**（`84bb26b`、`b71f880`）：11 章共 ~2300 行（旧版整本 848 行）。
- [ ] **Batch 3 · notebooks**：先 CPU/随机权重批（01/02/03/05-anatomy/06/07-moe/08），再 E2B 真权重批（00/04/05/09/10）。
- [ ] **Batch 4 · Part II 改写**：接到主线术语上。
- [ ] **Batch 5 · 中文版正文**：`book-zh/` Part I 走读翻译（目前只有导读部分的中文，正文指向英文版）。

## 读源码挖到的、与「看 config 想当然」相反的事实

这些都已写进书里，也是这次重构最有价值的产出：

1. **PLE 只在 E2B/E4B 开着。** 31B 与 26B-A4B 的 `hidden_size_per_layer_input` 是 `0`，逐层 embedding 通路整个关掉。PLE 是用内存换容量的端侧技术——`embed_tokens_per_layer` 在 E2B 上约 2.35B 参数，比模型其余部分还大，但每 token 只读一行。大模型宁可把预算给 hidden size。
2. **双向视觉注意力恰好相反，而且只作用于滑窗层。** E2B/E4B 的 `use_bidirectional_attention` 是 `None`（图像 token 也严格因果）；31B/26B 才是 `"vision"`。且 `create_masks_for_vision_model` 的 docstring 明说：全局层刻意保持纯因果，这是相对 Gemma 3 的修正——因为全局层的 KV cache 必须对增量解码有效。
3. **`use_double_wide_mlp` 是有条件的。** `Gemma4TextMLP.__init__` 里 `use_double_wide_mlp and is_kv_shared_layer`：E2B 上放弃了 KV 投影的那 20 层，正是拿到 2× 宽 MLP 的那 20 层。这不是两个 flag，是同一个决定——上层不再为注意力付钱，把省下的给 FFN。
4. **KV 共享边界之前必须至少留一个该类型的「捐赠」层**（搭迷你 config 时撞出来的）。`store_full_length_kv` 标记捐赠层，滑窗层与全局层各需一个，因为两者 RoPE 不同、KV 不可互换。E2B 是 35 层共享 20 层，边界在 15，全局层在 4/9/14，正好留住 14。违反时报 `KeyError: 'full_attention'`。
5. **像素预算是目标而非上限**——100×100 的缩略图在默认档下被放大到 768×768 并收 256 个 soft token。批处理小图时应显式降档。
6. **视频每帧只值 70 soft token**（图片是 280），32 帧共 2240；时间戳以 `MM:SS` 字面文本写进 prompt，而不是位置编码。

## 验证方式

- `teachbooks build book` 与 `teachbooks build book-zh` 必须 0 warning。
- toc 条目全部存在、无孤儿页、相对链接可解析（有校验脚本思路见提交历史）。
- CPU/随机权重类 notebook：`jupyter nbconvert --execute` 全跑通，输出存进 `.ipynb`（读者无卡也能看到结果）。注意 kernelspec 用的是裸 `python`，执行时需确保 PATH 指向目标环境。
- 手写复现类 cell 必须带 `torch.testing.assert_close` 断言——既是教学点也是回归测试。
- 真权重类 notebook 受磁盘约束（见下），未能本机执行的会明确标注。

## 已知约束

**磁盘**：开发这台实例 `/` 只剩约 11GB，而 `google/gemma-4-E2B-it` 权重 10.25GB，下完几乎归零。E2B 真权重那批 notebook 因此可能只写不跑；凡未在本机执行过的，都会在 notebook 里标注，并只做静态检查（`inspect.signature` 核对 API 名与参数）。
