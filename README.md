# Multimodal-101

Multimodal AI explained simply — from CLIP embeddings to unified omni models, theory and hands-on practice side by side.

The book walks the full modern multimodal stack — **VLM understanding → document OCR → image/video generation & editing → speech → unified omni models → RAG/agent applications**. Every chapter is a three-piece set: theory (principles + key papers), landscape (current state of play with a last-verified date), and runnable notebooks — open models locally, closed frontier APIs for contrast.

📖 **Read online (English, canonical)**: https://xinli95.github.io/Multimodal-101

📖 **中文版**: https://xinli95.github.io/Multimodal-101/zh/ （理论/格局页面双语维护，notebook 为两版共用的英文版）

Built with the [TeachBooks](https://teachbooks.io) template, same as [AI-101](https://github.com/xinli95/AI-101) and [Speech-101](https://github.com/xinli95/Speech-101). Both editions deploy from one repo via [`deploy-books.yml`](.github/workflows/deploy-books.yml).

## Build locally

```bash
pip install -r requirements.txt
teachbooks build book       # English edition
teachbooks build book-zh    # Chinese edition
```

Then open `book/_build/html/index.html` (or `book-zh/_build/html/index.html`).

## Contributing / translation policy

- **English (`book/`) is the source of truth** — content changes land there first.
- The Chinese edition (`book-zh/`) mirrors the same file layout; translate the changed pages after the English side merges.
- Notebooks (`book/*/notebooks/*.ipynb`) exist only in English and are shared by both editions; the Chinese notebook index pages link to them.

## License

Content CC BY 4.0, code MIT (see [LICENSE](LICENSE)). Model licenses vary — see each chapter's `landscape.md`.
