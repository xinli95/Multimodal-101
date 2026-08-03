# Multimodal-101

Multimodal AI explained simply — from CLIP embeddings to unified omni models, theory and hands-on practice side by side.

本书系统走过多模态 AI 的全部主线：**VLM 理解 → 文档 OCR → 图像/视频生成与编辑 → 语音 → 统一 Omni 模型 → RAG/Agent 应用**。每章都是"原理（theory）+ 最新格局（landscape，带核实日期）+ 可运行 notebook"三件套，开源模型本地跑、闭源前沿 API 对照。

📖 **在线阅读**: https://xinli95.github.io/Multimodal-101

Built with the [TeachBooks](https://teachbooks.io) template, same as [AI-101](https://github.com/xinli95/AI-101) and [Speech-101](https://github.com/xinli95/Speech-101).

## Build locally

```bash
pip install -r requirements.txt
teachbooks build book
```

Then open `book/_build/html/index.html`.

## License

教程内容 CC BY 4.0，代码 MIT（见 [LICENSE](LICENSE)）。各章引用的模型 license 见每章 `landscape.md`。
