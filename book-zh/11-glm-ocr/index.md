# 11 · OCR as a Multimodal System — GLM-OCR

**在全书中的位置**：方法迁移测试。00–10 章教你如何逐层拆解一个多模态模型；本章把同一套方法迁移到专用 OCR，并把分析边界从 checkpoint 扩展到完整文档 pipeline。

GLM-OCR 的核心是约 0.9B 参数的视觉—语言生成模型：0.4B CogViT 视觉编码器、视觉 token 下采样与连接模块、0.5B GLM 语言解码器。完整 SDK 还会先用 PP-DocLayout-V3 做版面检测，把文本、表格、公式区域并行送入 base model，最后恢复阅读顺序并输出 Markdown 与 layout JSON。

## 你会学到

1. 现代 OCR 的五类输出契约：文本、公式、表格、文档解析、KIE
2. 如何从发布的 `config.json` 反推 CogViT、visual-token merge、GLM decoder 与融合路径
3. 为什么普通 `transformers.generate()` 是自回归 baseline，而官方 MTP 加速需要在 vLLM/SGLang serving 中显式开启
4. 如何区分 base model 错误、layout detector 错误、reading-order 错误和 formatter 错误
5. 如何分别用编辑距离、CDM、TEDS、field F1、JSON/tag validity 评估不同输出契约

## 源码地图

本章面向已发布的 `zai-org/GLM-OCR` checkpoint，以及与 Part I 相同版本线的 `transformers` 5.14.x 实现。

| 来源 | 符号 / 文件 | 作用 |
|---|---|---|
| Hugging Face checkpoint | [`config.json`](https://huggingface.co/zai-org/GLM-OCR/blob/main/config.json)、[`preprocessor_config.json`](https://huggingface.co/zai-org/GLM-OCR/blob/main/preprocessor_config.json) | 已发布权重的真实参数；优先于类默认值 |
| `configuration_glm_ocr.py` | [`GlmOcrConfig`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/configuration_glm_ocr.py#L132)、[`GlmOcrVisionConfig`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/configuration_glm_ocr.py#L30)、[`GlmOcrTextConfig`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/configuration_glm_ocr.py#L72) | 嵌套架构配置 |
| `modeling_glm_ocr.py` | [`GlmOcrVisionPatchEmbed`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L517)、[`GlmOcrVisionBlock`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L469) | patchify 与 CogViT encoder |
| | [`GlmOcrVisionPatchMerger`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L500)、[`GlmOcrVisionModel`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L537) | 2×2 空间下采样与到文本宽度的 gated projection |
| | [`GlmOcrModel.get_image_features`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L1008)、[`get_placeholder_mask`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L1028) | 视觉特征提取与占位符替换 |
| | [`GlmOcrTextModel`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L708)、[`GlmOcrForConditionalGeneration`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm_ocr/modeling_glm_ocr.py#L1202) | decoder 与自回归 LM head |
| `processing_glm46v.py` / `image_processing_glm46v.py` | [`Glm46VProcessor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm46v/processing_glm46v.py#L43)、[`Glm46VImageProcessor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/glm46v/image_processing_glm46v.py#L87) | checkpoint 复用的 GLM-V chat 与图像预处理 |
| GLM-OCR SDK | [`PageLoader`](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/dataloader/page_loader.py#L40)、[`PPDocLayoutDetector`](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/layout/layout_detector.py#L27)、[`OCRClient`](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/ocr_client.py#L24)、[`ResultFormatter`](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/postprocess/result_formatter.py#L40) | base model 外围的生产文档 pipeline |

## Notebooks

notebook 为中英文版共用的英文版本：

| Notebook | 内容 | 硬件 |
|---|---|---|
| <a href="../../11-glm-ocr/notebooks/01_glm_ocr_anatomy.html"><code>01_glm_ocr_anatomy.ipynb</code></a> | 读 checkpoint JSON、构造迷你随机模型、hook 张量、手写验证 2×2 downsample | 🟢 CPU |
| <a href="../../11-glm-ocr/notebooks/02_tasks_and_mtp.html"><code>02_tasks_and_mtp.ipynb</code></a> | 文本/表格/公式/KIE 四种契约；AR 与 MTP server 的严格分离对照 | 🟡 8–12GB+ 或服务端 |
| <a href="../../11-glm-ocr/notebooks/03_base_vs_pipeline.html"><code>03_base_vs_pipeline.ipynb</code></a> | 整页 base model vs layout-aware SDK；故障注入与分阶段评测 | 🟡 GPU 或 MaaS |

> 📝 完整源码走读目前以英文版为准，中文翻译进行中。见 [English edition](https://xinli95.github.io/Multimodal-101/11-glm-ocr/index.html)。
