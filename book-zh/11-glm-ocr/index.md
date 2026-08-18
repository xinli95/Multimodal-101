# 11 · OCR as a Multimodal System — GLM-OCR

**在全书中的位置**：方法迁移测试。00–10 章教你如何逐层拆解一个多模态模型；本章把同一套方法迁移到专用 OCR，并把分析边界从 checkpoint 扩展到完整文档 pipeline。

GLM-OCR 的核心是约 0.9B 参数的视觉—语言生成模型：0.4B CogViT 视觉编码器、视觉 token 下采样与连接模块、0.5B GLM 语言解码器。完整 SDK 还会先用 PP-DocLayout-V3 做版面检测，把文本、表格、公式区域并行送入 base model，最后恢复阅读顺序并输出 Markdown 与 layout JSON。

## 你会学到

1. 现代 OCR 的五类输出契约：文本、公式、表格、文档解析、KIE
2. 如何从发布的 `config.json` 反推 CogViT、visual-token merge、GLM decoder 与融合路径
3. 为什么普通 `transformers.generate()` 是自回归 baseline，而官方 MTP 加速需要在 vLLM/SGLang serving 中显式开启
4. 如何区分 base model 错误、layout detector 错误、reading-order 错误和 formatter 错误
5. 如何分别用编辑距离、CDM、TEDS、field F1、JSON/tag validity 评估不同输出契约

## Notebooks

notebook 为中英文版共用的英文版本：

| Notebook | 内容 | 硬件 |
|---|---|---|
| <a href="../../11-glm-ocr/notebooks/01_glm_ocr_anatomy.html"><code>01_glm_ocr_anatomy.ipynb</code></a> | 读 checkpoint JSON、构造迷你随机模型、hook 张量、手写验证 2×2 downsample | 🟢 CPU |
| <a href="../../11-glm-ocr/notebooks/02_tasks_and_mtp.html"><code>02_tasks_and_mtp.ipynb</code></a> | 文本/表格/公式/KIE 四种契约；AR 与 MTP server 的严格分离对照 | 🟡 8–12GB+ 或服务端 |
| <a href="../../11-glm-ocr/notebooks/03_base_vs_pipeline.html"><code>03_base_vs_pipeline.ipynb</code></a> | 整页 base model vs layout-aware SDK；故障注入与分阶段评测 | 🟡 GPU 或 MaaS |

> 📝 完整源码走读目前以英文版为准，中文翻译进行中。见 [English edition](https://xinli95.github.io/Multimodal-101/11-glm-ocr/index.html)。
