# 03 · Image Processor — Pixels Under a Token Budget

**Position in the pipeline**: `image ──► Gemma4ImageProcessor ──► pixel_values + image_position_ids`

**Mental-model checkpoint:** this chapter covers the *compute-allocation and geometry* step before the learned vision tower. The processor decides how densely the model will sample the image; it does not yet decide what the image means.

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
| `image_processing_gemma4.py` | [`_SUPPORTED_SOFT_TOKENS`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L29) | The menu above, as a tuple |
| | [`get_aspect_ratio_preserving_size`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L33) | The budget solver: target size from image size + patch budget |
| | [`convert_image_to_patches`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L88) | Grid → sequence of patches |
| | [`pad_along_first_dim`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L103) | Padding a batch of variable patch counts to a common length |
| | [`Gemma4ImageProcessor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L136) ([`TorchvisionBackend`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/image_processing_backends.py#L86)) | The fast path; [`Gemma4ImageProcessorPil`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_pil_gemma4.py#L137) is the PIL fallback |
| `configuration_gemma4.py` | [`Gemma4VisionConfig.patch_size`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L267), [`.pooling_kernel_size`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/configuration_gemma4.py#L266) | Where 16 and 3 come from |

## Walkthrough

You can follow this entire chapter without downloading a checkpoint — the image processor is a standalone class:

```python
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
ip = Gemma4ImageProcessor()   # patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
```

### 1. The budget solver

The whole scheme is nine lines:

```python
def get_aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size):
    total_px  = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width  = factor * width
    side_mult = pooling_kernel_size * patch_size          # 3 * 16 = 48

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width  = int(math.floor(ideal_width  / side_mult)) * side_mult
```

Read it as: *how much do I have to scale this image so its area equals the budget?* — that is `factor` — *then round both sides down to the nearest multiple of 48.*

**Why 48.** Two constraints stack. The image is cut into 16×16 patches, so each side must be a multiple of `patch_size`. Then 3×3 blocks of patches are average-pooled into one soft token (chapter 04), so the *patch grid* must be a multiple of `pooling_kernel_size`. Together: `48 = 16 × 3`. Every "why is this number 48" question about Gemma 4 bottoms out in those two lines of `Gemma4VisionConfig`.

**Rounding down is the safety property.** Because both sides are floored, the result is guaranteed to fit — and the function asserts it rather than trusting itself:

```python
if target_height * target_width > target_px:
    raise ValueError(f"Resizing [{height}x{width}] to [{target_height}x{target_width}] but this exceeds {max_patches} patches...")
```

**The edge cases are where the real thinking is.** A 3000×200 panorama at a small budget will floor one dimension to literally zero. The function catches that and clamps to one 48-pixel band, capped so the other side cannot blow the budget:

```python
max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
if target_height == 0:
    target_height = side_mult
    target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
```

### 2. What it actually does to real images

Run the solver over a spread of shapes — this is the table that makes the design click:

| Input | Budget 70 | Budget 280 | Budget 1120 |
|---|---|---|---|
| 640×480 | 432×336 → 63 tokens | 912×672 → 266 | 1824×1344 → 1064 |
| 1024×1024 | 384×384 → 64 | 768×768 → 256 | 1584×1584 → 1089 |
| 1920×1080 | 528×288 → 66 | 1056×576 → 264 | 2112×1200 → 1100 |
| 3000×200 | 1536×96 → 64 | 3072×192 → 256 | 6192×384 → 1032 |
| **100×100** | **384×384 → 64** | **768×768 → 256** | **1584×1584 → 1089** |

Three observations, one of which costs money:

1. **Aspect ratio really is preserved.** 640×480 and 480×640 produce transposed targets and identical token counts. The 15:1 panorama stays 16:1. No squashing.
2. **You never quite hit the budget.** Flooring to multiples of 48 wastes 2–10% of the allowance. `pixel_values` is padded back up to the full `max_patches` anyway (§4), so that slack is real compute you pay for and do not use.
3. **The budget is a target, not a cap — small images get upscaled.** That 100×100 thumbnail is resized *up* to 768×768 and charged 256 soft tokens. Bicubic upsampling adds no information; it adds cost. If you are batch-processing thumbnails or icons, drop `max_soft_tokens` to 70 explicitly. Nothing in the API warns you about this.

### 3. No ImageNet normalisation, and where the real scaling happens

The class attributes are unusually blunt:

```python
class Gemma4ImageProcessor(TorchvisionBackend):
    image_mean    = [0.0, 0.0, 0.0]
    image_std     = [1.0, 1.0, 1.0]
    do_normalize  = False
    do_rescale    = True        # rescale_factor = 1/255
```

Normalisation is off and the mean/std are identity, so `rescale_and_normalize` is a plain division by 255. Verify it:

```python
out = ip([np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)], return_tensors="pt")
print(out["pixel_values"].min(), out["pixel_values"].max())   # 0.0  1.0
```

Pixels reach the model in **[0, 1]**, not standardised. The shift to [-1, 1] the model wants is folded into the patch-embedding layer itself (chapter 04). Two reasons this matters in practice: if you preprocess images yourself you must **not** apply the ImageNet constants everyone's fingers type automatically, and if you are debugging a fine-tune where images look washed out, this is the first place to look.

There is also a small piece of plumbing worth noticing, because it explains an odd override:

```python
def _validate_preprocess_kwargs(self, **kwargs):
    # Gemma4 uses aspect_ratio_preserving_resize driven by patch_size, max_soft_tokens,
    # and pooling_kernel_size — not the standard `size` parameter.
    kwargs["do_resize"] = False
    super()._validate_preprocess_kwargs(**kwargs)
```

The base class insists that `do_resize=True` implies a `size` dict. Gemma 4 resizes, but its target is computed, not configured — so it lies to the validator. `size = None` on this class is deliberate.

### 4. Patchify, positions, padding

Patchification is inherited wholesale from SigLIP 2 (the source says so in a `# Copied from` comment) and is pure reshaping:

```python
patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
patched_image = patched_image.permute(1, 3, 2, 4, 0)
patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
```

A patch becomes a vector of `16 × 16 × 3 = 768` numbers. Row-major over the grid: patch index runs left to right, then top to bottom.

Positions are built with an explicit `"xy"` indexing convention:

```python
patch_grid = torch.meshgrid(torch.arange(patch_width), torch.arange(patch_height), indexing="xy")
real_positions = torch.stack(patch_grid, dim=-1).reshape(patches.shape[0], 2)
```

so each row is **`(x, y)` — column first, row second**. Get this backwards when you write your own 2D RoPE in chapter 04 and everything will run, produce plausible shapes, and be subtly wrong.

Then everything is padded to the full budget, with padding *positions* marked `-1`:

```python
positions = torch.nn.functional.pad(positions, (0, 0, 0, padding_length), mode="constant", value=-1)
```

Run it on two very different images at once and look at the result:

```python
out = ip([img_480x640, img_100x100], return_tensors="pt")
# pixel_values             torch.Size([2, 2520, 768])
# image_position_ids       torch.Size([2, 2520, 2])
# num_soft_tokens_per_image  tensor([266, 256])
```

**`pixel_values` has a fixed shape regardless of input image.** 2520 is `280 × 3²`, the full patch budget. Variable-resolution input, constant-shape tensor — that is what makes batching and `torch.compile` tractable, and it is why `(-1, -1)` exists: the *positions* tensor is what tells the tower which of those 2520 slots are real. `num_soft_tokens_per_image` is the true count (266 and 256 here, not 280), and it is what chapter 02's `replace_image_token` uses to size the placeholder run.

This trio — fixed-shape values, sentinel-marked positions, a separate true-count — is the same pattern you will meet again for audio in chapter 05.

### 5. `Gemma4ImageProcessorPil`

There are two implementations: `Gemma4ImageProcessor` extends `TorchvisionBackend` (tensor ops, GPU-capable, the default), and `Gemma4ImageProcessorPil` is the PIL fallback. They must agree numerically, which is why the resize specifies `antialias=True` explicitly — PIL's `BICUBIC` antialiases by default and torchvision's does not. Silent train/serve skew lives in exactly this kind of gap.

## Design space

Four answers to "arbitrary grid → bounded sequence", and what each optimises:

| Approach | Token count | Aspect ratio | Cost of a big image |
|---|---|---|---|
| **Fixed square** (CLIP, LLaVA 1.0) | Constant | Destroyed | Constant — detail is simply lost |
| **Tiling** (InternVL, LLaVA-NeXT) | Steps by tile | Preserved per tile | Grows in chunks; tile seams are real |
| **Native resolution** (Qwen2-VL/Qwen3-VL) | Continuous, unbounded | Preserved | Grows without limit — a 4K screenshot can cost thousands of tokens |
| **Pixel budget** (Gemma 4) | Chosen from a menu | Preserved | **Constant by construction** |

Gemma 4's version is the most *predictable*: you know the token cost before you see the image, which is exactly what you need to plan a context budget or price an API call. The trade is that it does not adapt — a dense 4K document and a blurry snapshot both get 280 tokens unless you intervene, whereas Qwen's native-resolution scheme would spend more on the document automatically. The menu is the escape hatch, and using it well is a real engineering decision, not a default to leave alone.

The design also quietly explains Gemma 4's OCR behaviour. At 280 soft tokens a 1024×1024 page is rendered at 768×768 — roughly 0.7 megapixels for a full page. Small print will not survive that. Bumping to 560 or 1120 is the fix, and chapter 04's notebook measures where the cliff is.

## Check yourself

1. Why must both image dimensions be divisible by 48 and not just by 16?
2. A 4000×3000 photo at the default budget. What target size does the solver pick, and how many soft tokens does it produce?
3. You preprocess images yourself with the usual ImageNet mean/std before handing them to the model. What breaks, and why is it not an exception?
4. `pixel_values` is `[2, 2520, 768]` for two images of completely different sizes. Where is the information about their actual sizes?
5. You are OCR-ing scanned invoices and the model misreads small figures. Which single parameter do you change, and what does it cost you?
6. Positions are `(x, y)`. What goes wrong if you assume `(row, col)` when writing your own 2D RoPE?

## Notebooks

| Notebook | What it does | Hardware |
|---|---|---|
| `01_pixels_to_patches.ipynb` | Reimplement `get_aspect_ratio_preserving_size` and `convert_image_to_patches` by hand, `assert_close` against the library, visualise the patch grid over a real photo, and sweep the soft-token menu against a text-heavy image to see where OCR breaks down | 🟢 CPU |
