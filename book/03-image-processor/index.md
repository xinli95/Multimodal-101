# 03 · Image Processor — Pixels Under a Token Budget

**Position in the pipeline**: `image ──► Gemma4ImageProcessor ──► pixel_values + image_position_ids`

Every vision-language model has to answer one question: an image is a 2D grid of arbitrary size, a transformer wants a 1D sequence of bounded length — how do you convert? The answers the field has tried are the whole history of VLM image handling: squash everything to 224×224 (CLIP, lossy for anything text-shaped), tile a big image into fixed crops (InternVL), keep native resolution and let the sequence grow (Qwen2-VL, expensive and unpredictable).

Gemma 4's answer is a **pixel budget**: keep the aspect ratio, scale the image so its total pixel count fits a budget, and force both sides to be divisible by 48. The number of vision tokens is then fixed and known in advance — you choose it from a menu.

## What you will learn

1. Why the divisor is **48**, and why that is `patch_size (16) × pooling_kernel_size (3)`
2. The soft-token menu and its cost/detail trade-off, measured rather than asserted
3. Why Gemma 4 deliberately **does not** apply ImageNet mean/std normalisation, and where the scaling to [-1, 1] actually happens
4. What `image_position_ids` is, why padding patches are marked `(-1, -1)`, and why the model needs coordinates rather than an index
5. Reimplement the resize and patchify steps yourself and assert bit-for-bit agreement with the library

## The soft-token menu

| Soft tokens | Patches before pooling | Approx. image area |
|---|---|---|
| 70 | 630 | ~161K px |
| 140 | 1,260 | ~323K px |
| **280** (default) | **2,520** | **~645K px** |
| 560 | 5,040 | ~1.3M px |
| 1,120 | 10,080 | ~2.6M px |

Nine patches (3×3) pool down to one soft token, hence the 9× ratio between the columns. `max_soft_tokens` is validated against exactly this set — anything else raises.

## Source map

| File | Symbol | Role |
|---|---|---|
| `image_processing_gemma4.py` | `_SUPPORTED_SOFT_TOKENS` | The menu above, as a tuple |
| | `get_aspect_ratio_preserving_size` | The budget solver: target size from image size + patch budget |
| | `convert_image_to_patches` | Grid → sequence of patches |
| | `pad_along_first_dim` | Padding a batch of variable patch counts to a common length |
| | `Gemma4ImageProcessor` (`TorchvisionBackend`) | The fast path; `Gemma4ImageProcessorPil` is the PIL fallback |
| `configuration_gemma4.py` | `Gemma4VisionConfig.patch_size`, `.pooling_kernel_size` | Where 16 and 3 come from |

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_pixels_to_patches.ipynb` | Reimplement `get_aspect_ratio_preserving_size` and `convert_image_to_patches` by hand, `assert_close` against the library, visualise the patch grid over a real photo, and sweep the soft-token menu against a text-heavy image to see where OCR breaks down | 🟢 CPU |
