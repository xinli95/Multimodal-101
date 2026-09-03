# 03 · Image Processor — Pixels Under a Token Budget

This chapter can be read on its own. Its question is: **how does an ordinary image become the fixed-shape tensors that Gemma 4's learned vision tower accepts?**

```text
PIL image / NumPy array / tensor
              │
              ▼
      Gemma4ImageProcessor
      resize → rescale → patchify → pad
              │
              ├─ pixel_values
              ├─ image_position_ids
              └─ num_soft_tokens_per_image
              │
              ▼
      Gemma4VisionModel                (chapter 04)
              │
              ▼
      visual soft-token embeddings
```

This chapter follows the middle box: it allocates image compute and records geometry, but does not contain learned weights and does not decide what the image means. [Chapter 02](../02-text-io/index.md) uses `num_soft_tokens_per_image` to reserve the right number of positions in the text prompt; [chapter 04](../04-vision-tower/index.md) consumes `pixel_values` and `image_position_ids` to produce learned visual features.

Every vision-language model has to answer one question: an image is a 2D grid of arbitrary size, a transformer wants a 1D sequence of bounded length — how do you convert? The answers the field has tried are the whole history of VLM image handling: fit everything into a fixed square (CLIP, lossy for anything text-shaped), tile a big image into fixed crops (InternVL), or let an aspect-ratio-preserving grid grow with the image within configured bounds (Qwen2-VL, adaptive but variable in cost).

Gemma 4's answer is a **pixel budget**: keep the aspect ratio, scale the image so its total pixel count fits a budget, and force both sides to be divisible by 48. You choose a maximum from a five-item menu. The padded patch tensor then has a known fixed shape; the number of real soft tokens can be slightly below that maximum because both image dimensions are rounded down.

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

Nine patches (3×3) pool down to one soft token, hence the 9× ratio between the columns.

Here `max_soft_tokens` is **not an arbitrary integer upper bound**. “Validated against this set” means the processor checks that the value is a member of `(70, 140, 280, 560, 1120)`. “Raises” is Python shorthand for “raises an exception”: for example, `Gemma4ImageProcessor(max_soft_tokens=300)` immediately raises `ValueError` instead of silently rounding 300 to a supported tier. The same check is applied if the value is overridden during preprocessing. These five values are therefore configuration presets, not five examples.

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

1. **Aspect ratio is preserved approximately, then quantised.** 640×480 and 480×640 produce transposed targets and identical token counts. Independent rounding to 48-pixel multiples changes 4:3 to about 1.29:1 at budget 70, and the 15:1 panorama to 16:1. That is a small geometry error, not the wholesale squashing into a square used by some fixed-canvas pipelines.
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

### 4. From a 2D image to a batchable sequence

Four terms are easy to collapse into one, so keep their jobs separate:

| Term | What it is | Why it exists |
|---|---|---|
| **patch** | One 16×16 RGB square, flattened to 768 numbers | Turns the pixel grid into units a transformer can ingest |
| **position ID** | The patch's `(x, y)` coordinate in the patch grid | Preserves 2D layout after patches become a 1D sequence |
| **padding slot** | A zero patch with position `(-1, -1)` | Gives every image in a batch the same tensor shape |
| **soft token** | The learned result after the vision tower pools a 3×3 block of patches | This is what eventually occupies the language-model context |

#### First make patches

Take the first row of the earlier table: a 640×480 input at budget 70 is resized to **432×336** (width × height). With 16×16 patches, it becomes a grid with 27 columns and 21 rows:

```text
432×336 pixels
    │ split into 16×16 squares
    ▼
27×21 patch grid = 567 patches
    │ later, pool each 3×3 patch block
    ▼
9×7 grid = 63 soft tokens
```

The processor performs only the first conversion. The 3×3 pooling happens in the learned vision tower in chapter 04; `num_soft_tokens_per_image` merely records the result it will have: `567 / 9 = 63`.

Patchification is inherited wholesale from SigLIP 2 (the source says so in a `# Copied from` comment) and is pure reshaping:

```python
patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
patched_image = patched_image.permute(1, 3, 2, 4, 0)
patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
```

A patch becomes a vector of `16 × 16 × 3 = 768` numbers. The 567 patches above therefore become a tensor with shape `[567, 768]`. They are flattened row by row: left to right across the first row, then left to right across the second row, and so on.

#### Then remember where each patch came from

Positions are built with an explicit `"xy"` indexing convention:

```python
patch_grid = torch.meshgrid(torch.arange(patch_width), torch.arange(patch_height), indexing="xy")
real_positions = torch.stack(patch_grid, dim=-1).reshape(patches.shape[0], 2)
```

so each row is **`(x, y)` — column first, row second**. For the 27×21 example, the sequence starts `(0,0), (1,0), …, (26,0), (0,1), …` and ends at `(26,20)`. A one-dimensional sequence index is not enough: index 27 could mean “start of the second row” only if the model also knew that this particular image was 27 patches wide. Explicit `(x, y)` coordinates keep that fact after flattening. Get the order backwards when you write your own 2D RoPE in chapter 04 and every shape still looks valid, but horizontal and vertical position information is exchanged.

#### Finally pad to one common shape

Different aspect ratios use different numbers of real patches. At budget 70, for example, a batch tensor cannot directly stack the rectangular image's `[567, 768]` result above beside a square image's `[576, 768]`, so each image is extended to the tier's full 630-patch budget. Pixel rows are padded with zeros and their matching positions are marked `(-1, -1)`:

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

Read the three outputs together:

- `pixel_values[i]` always has 2,520 rows at the default tier: real patch vectors first, zero padding after them.
- `image_position_ids[i]` gives every real row its `(x, y)` coordinate and marks every padded row `(-1, -1)`. The vision tower derives its attention mask from this sentinel, so padding is not treated as image content.
- `num_soft_tokens_per_image[i]` is the number of real 3×3 groups after pooling. It is 266 and 256 here, not 280; chapter 02 uses it to size the placeholder run in the text sequence.

Variable-resolution input has therefore become a constant-shape patch tensor without losing the boundary between real data and padding. This makes batching and compilation tractable. Notice that padding happens **before** the vision tower, whereas soft-token pooling happens **inside** it.

This trio — fixed-shape values, sentinel-marked positions, a separate true-count — is the same pattern you will meet again for audio in chapter 05.

### 5. Why is there a PIL fallback?

“Fallback” here means **an alternative preprocessing backend**, not “retry with PIL if an image is corrupt” or “recover from a GPU error.” `Gemma4ImageProcessor` uses torchvision tensor operations and is the default when torchvision is installed; it can keep tensor inputs on an accelerator and fits compiled tensor pipelines. `Gemma4ImageProcessorPil` implements the same resize → rescale → patchify → pad contract with PIL and NumPy, so the processor also works in a lightweight CPU environment without torchvision. Hugging Face's [`AutoImageProcessor`](https://huggingface.co/docs/transformers/main/en/image_processors) can select PIL when torchvision is unavailable, or the caller can request a supported backend explicitly.

Is this common practice? **PIL preprocessing is common; providing two interchangeable backends is common compatibility engineering, but not a requirement of the model and not universal across all modern processors.** PIL was the conventional Python image path for years and remains useful for portability and reproducing older pipelines. Torchvision is now the performance path. Gemma 4 carries both because the preprocessing contains no learned computation and can be expressed either way.

The maintenance cost is numerical parity. Resize libraries can differ at pixel boundaries, which can create silent train/serve skew. Gemma 4 therefore requests `antialias=True` on the torchvision bicubic resize to match PIL's antialiased bicubic behaviour as closely as possible. The two backends should implement the same semantics, but floating-point or interpolation details can still cause tiny numerical differences; “same backend everywhere” is the safest rule when exact reproducibility matters.

## Design space

The approaches below solve the same interface problem—turn arbitrary `H×W` pixels into transformer tokens—but make different choices about *where resolution is lost* and *whether cost follows the input*.

| Approach | Mechanism | Token-cost behaviour | Strength | Characteristic failure mode |
|---|---|---|---|---|
| **Fixed canvas** ([CLIP](https://arxiv.org/abs/2103.00020), early LLaVA) | Resize every image to one training resolution; obtain a square via cropping, padding, or sometimes distortion | Exactly constant | Simple, dense batches, predictable latency | Downsampling removes fine detail; cropping may remove content; stretching changes geometry |
| **Any-resolution tiling** ([LLaVA-NeXT](https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py), [InternVL 1.5](https://arxiv.org/abs/2404.16821)) | Choose a canvas/grid, split it into fixed-size crops, and usually add a global thumbnail | Rises in discrete tile-sized steps, up to a configured tile limit | Local tiles retain small text and objects while the thumbnail supplies global context | Repeated encoding and padding cost; the model must reconcile tiles, boundaries, and global coordinates |
| **Dynamic/native grid** ([Qwen2-VL](https://arxiv.org/abs/2409.12191)) | Keep one aspect-ratio-preserving grid and let its patch count follow image area, within configured minimum and maximum pixel bounds | Variable and approximately proportional to retained area | A small image can stay cheap while a detailed large image receives more tokens | Variable sequence lengths complicate batching and latency; an overly generous maximum makes large inputs expensive |
| **Selected pixel budget** (Gemma 4) | Scale every image toward one chosen area tier, round both sides to pooling-compatible multiples, then pad patches to the tier ceiling | Fixed padded patch shape; real soft-token count is near but at most the chosen tier | Predictable memory shape and a direct quality/cost knob | Content density is ignored; small inputs are upscaled and dense large inputs are downsampled |

### Fixed canvas: spend the same, whatever arrives

Classic CLIP-style pipelines present the vision encoder with one square resolution. “Fixed square” does not imply one universal resize rule: a pipeline may resize and centre-crop, letterbox with padding, or stretch. All three create a constant patch grid, which makes training and batching easy. The unavoidable trade is that an ultrawide screenshot and a portrait document must both fit the same canvas. Cropping sacrifices coverage, padding wastes part of the grid, and stretching sacrifices geometry; heavy downsampling sacrifices fine detail in every variant.

### Tiling: buy detail in whole crops

Tiling avoids shrinking the entire image to a small square. LLaVA-NeXT's AnyRes selects a candidate canvas, fits and pads the image, then divides it into vision-encoder-sized crops; InternVL-style dynamic high resolution similarly chooses a tile grid and commonly adds a thumbnail of the full image. This gives the model both local high-resolution views and global context. Cost increases one tile at a time rather than one patch at a time. It works well for documents and large scenes, but some pixels may be encoded twice, padding inside the chosen canvas may be wasted, and the model needs machinery or training to understand how separate crop coordinates fit together.

### Dynamic/native grid: let the image choose the bill

“Native resolution” is convenient shorthand, not a claim that raw dimensions pass through unchanged. Qwen2-VL rounds/resizes the image into a patch-compatible grid and constrains it with [`min_pixels` and `max_pixels`](https://huggingface.co/docs/transformers/model_doc/qwen2_vl#usage-tips); within that interval, a larger retained area produces more visual tokens. This is adaptive in a way Gemma 4 is not: a small icon can use few tokens while a large document can use many. The price is input-dependent memory and latency, plus less uniform batches. It is also not literally unbounded when `max_pixels` is configured—the bound is continuous and user-set rather than one of Gemma 4's five discrete tiers.

### Gemma 4's selected budget: choose the bill first

Gemma 4 asks the caller to choose a tier, then scales both a dense 4K document and a blurry snapshot toward that same target area. This is the most operationally predictable option: before seeing the image, you know the padded patch shape and the maximum number of context tokens. You do **not** know the exact real-token count until the aspect-ratio rounding is done. The trade is lack of content awareness: the processor cannot know that one image contains tiny invoice lines and the other does not. The five-tier menu is the manual escape hatch, and selecting it is a real engineering decision rather than a default to leave alone.

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
