# Theory · Multimodal Foundations

## 1. Modalities and representations

- What a modality is: raw data forms and information density of text / images / video / audio
- Everything becomes tokens:
  - Text → BPE/SentencePiece subwords
  - Images → ViT patch embeddings (understanding side); VAE latents + patchify (generation side)
  - Video → spacetime patches
  - Audio → mel-spectrogram frames (understanding side); discrete neural-codec tokens (generation side, e.g. EnCodec / SoundStream)
- Key papers: ViT (Dosovitskiy 2020), EnCodec (Défossez 2022)

## 2. Cross-modal alignment: contrastive learning

- CLIP (Radford 2021): dual towers + InfoNCE, in-batch negatives
- SigLIP (Zhai 2023): sigmoid loss instead of softmax, decoupling batch size from quality
- Why CLIP-family encoders became the visual backbone of nearly every VLM and text-to-image model
- Limitations: weak fine-grained perception, short text window → motivates the solutions in later chapters

## 3. The three generative paradigms

### 3.1 Autoregressive (AR)
- Model p(x_t | x_<t), sample token by token
- Pros: same shape as an LLM, natively supports interleaved multimodal sequences. Cons: long token sequences for images/audio, error accumulation
- Representatives: the GPT family; VQ-GAN + AR for images; codec LMs for audio

### 3.2 Diffusion
- Forward noising / reverse denoising; DDPM (Ho 2020) → DDIM for fast sampling
- Latent Diffusion (Rombach 2022): run diffusion in a VAE latent space, compute drops dramatically — this is where Stable Diffusion comes from
- Classifier-Free Guidance: conditional control without a classifier

### 3.3 Flow Matching
- Rectified Flow / Flow Matching (Lipman 2022, Liu 2022): learn a straightened velocity field from noise to data
- Why virtually every post-2024 model (SD3, FLUX, Wan, LTX) switched to flow matching: more stable training, fewer sampling steps
- A unifying view with diffusion: both learn a continuous transport from a prior to the data distribution

## 4. Architectural convergence: Transformers everywhere

- U-Net → DiT (Peebles & Xie 2022): the diffusion backbone becomes a Transformer too, and scaling laws start to apply
- Bottom line for the whole book: every model in later chapters (VLM, image, video, audio, omni) is essentially "a Transformer + modality-specific tokenizer/detokenizer"

## Suggested reading order

1. CLIP paper (read closely)
2. DDPM → LDM (understand the loss; derivations can be skipped)
3. DiT + Flow Matching (focus on *why the industry converged here*)
