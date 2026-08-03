# Theory · Applications & Evaluation

## 1. Multimodal RAG

- Two routes:
  - **Parse route**: document → OCR/parse to text (ch. 02) → text RAG. Controllable and traceable — the engineering mainstream
  - **Visual retrieval route**: ColPali/ColQwen style — embed pages visually, skip parsing. Less plumbing; strong on long-tail layouts
- Multimodal embedding models: CLIP-family vs. late interaction on VLM backbones
- Choosing: table-dense → parse; scans/chart-heavy → visual retrieval; production often runs both

## 2. Multimodal agents

- **GUI / computer-use agents**: the screenshot → grounding (element localization) → action loop; why grounding precision is the bottleneck
- Visual tool calling: LLMs invoking image generation/editing as tools (agentic content pipelines)
- A glance at embodiment: VLA (vision-language-action) models and robotics

## 3. Evaluation methodology

- The three-layer evaluation pyramid:
  1. Automatic metrics (fast, regression-friendly, but risk drifting from human preference)
  2. **LLM/VLM-as-judge** (the mainstream; mind judge biases: position preference, self-preference)
  3. Human arena blind tests (gold standard; slow and expensive)
- This book's stance: every demo ships with a minimal eval set — "looks good to me" is not a conclusion

## 4. Production engineering

- Visual token economics: one image = hundreds to thousands of tokens; resolution strategy is the cost dial
- Latency: streaming, speculative decoding, routing to distilled small models
- **Model routing**: the production consensus is routing across 2–3 models by task (seen already in the video chapter); build the routing table from evals
- Content safety: watermarking (SynthID-class), deepfake detection, license compliance (this is what the license columns in every landscape.md are for)

## Key papers / projects

| Paper/Project | Year | Why read it |
|---|---|---|
| ColPali | 2024 | Opened the visual retrieval route |
| CogAgent / UI-TARS | 2024–2025 | GUI-agent representatives |
| MMMU / MMMU-Pro | 2023–2024 | How understanding benchmarks are designed |
| VBench series | 2024–2025 | Dimensional evaluation for generation |
