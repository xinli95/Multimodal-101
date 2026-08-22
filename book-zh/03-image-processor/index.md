# 03 · Image Processor — token 预算下的像素

**在数据流中的位置**：`image ──► Gemma4ImageProcessor ──► pixel_values + image_position_ids`

**Mental-model checkpoint：**本章只讲 learned vision tower 之前的「计算分配与几何」步骤。Processor 决定模型用多密的采样去看这张图；它还没有在判断图像表达什么。

每个视觉语言模型都要回答同一个问题：图像是任意尺寸的二维网格，Transformer 要的是长度有界的一维序列，怎么转？领域里试过的答案就是 VLM 图像处理的全部历史：全部压成 224×224（CLIP，对带文字的东西损失惨重）、把大图切成固定 crop（InternVL）、保持原生分辨率让序列自由生长（Qwen2-VL，昂贵且不可预测）。

Gemma 4 的答案是**像素预算**：保持长宽比，把图缩放到总像素数落进预算内，并强制两边都能被 48 整除。于是视觉 token 数固定且事先已知——你从一张菜单上挑。

## 你会学到

1. 为什么除数是 **48**，以及它为什么等于 `patch_size (16) × pooling_kernel_size (3)`
2. soft token 菜单及其成本/细节取舍——实测而不是断言
3. 为什么 Gemma 4 刻意**不做** ImageNet 均值/方差归一化，以及缩放到 [-1, 1] 究竟发生在哪
4. `image_position_ids` 是什么，padding patch 为什么标成 `(-1, -1)`，模型为什么要坐标而不是下标
5. 自己重写 resize 与 patchify，并断言与库的结果逐元素一致

## soft token 菜单

| soft token | 池化前 patch 数 | 大致图像面积 |
|---|---|---|
| 70 | 630 | ~161K 像素 |
| 140 | 1,260 | ~323K 像素 |
| **280**（默认） | **2,520** | **~645K 像素** |
| 560 | 5,040 | ~1.3M 像素 |
| 1,120 | 10,080 | ~2.6M 像素 |

9 个 patch（3×3）池化成 1 个 soft token，这就是两列之间 9 倍关系的由来。`max_soft_tokens` 会被严格校验只能取这几个值。

## 源码地图

| 文件 | 符号 | 作用 |
|---|---|---|
| `image_processing_gemma4.py` | [`_SUPPORTED_SOFT_TOKENS`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L29) | 上面那张菜单 |
| | [`get_aspect_ratio_preserving_size`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L33) | 预算求解：由图像尺寸 + patch 预算算出目标尺寸 |
| | [`convert_image_to_patches`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L88) | 网格 → patch 序列 |
| | [`pad_along_first_dim`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L103) | 把 patch 数不等的一批补齐 |
| | [`Gemma4ImageProcessor`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_gemma4.py#L136) | 快速路径；[`Gemma4ImageProcessorPil`](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/image_processing_pil_gemma4.py#L137) 是 PIL 回退实现 |

## 源码走读

不下载任何 checkpoint 就能跟完本章 —— 图像处理器是一个独立的类：

```python
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
ip = Gemma4ImageProcessor()   # patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
```

### 1. 预算求解器

整个方案就九行：

```python
def get_aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size):
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width  = factor * width
    side_mult = pooling_kernel_size * patch_size          # 3 * 16 = 48

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width  = int(math.floor(ideal_width  / side_mult)) * side_mult
```

这样读：*我要把这张图缩放多少，才能让它的面积等于预算？* —— 那就是 `factor` —— *然后把两边都向下取整到 48 的倍数。*

**为什么是 48。** 两个约束叠加。图像被切成 16×16 的 patch，所以每边必须是 `patch_size` 的倍数。然后 3×3 的 patch 块会被平均池化成一个 soft token（04 章），所以 **patch 网格**必须是 `pooling_kernel_size` 的倍数。合起来：`48 = 16 × 3`。所有"这个数为什么是 48"的问题，最后都落到 `Gemma4VisionConfig` 的那两行上。

**向下取整是安全性质。** 因为两边都做了 floor，结果保证放得下 —— 而函数并不信任自己，它会断言：

```python
if target_height * target_width > target_px:
    raise ValueError(f"Resizing [{height}x{width}] to [{target_height}x{target_width}] but this exceeds {max_patches} patches...")
```

**边界情况才是真正动脑的地方。** 一张 3000×200 的全景图在小预算下会把某一维 floor 到 0。函数捕获这一点并夹到一条 48 像素的带上，同时给另一边设上限以免撑爆预算：

```python
max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
if target_height == 0:
    target_height = side_mult
    target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
```

### 2. 它对真实图像做了什么

把求解器跑在一组形状上 —— 这张表能让整个设计瞬间明白：

| 输入 | 预算 70 | 预算 280 | 预算 1120 |
|---|---|---|---|
| 640×480 | 432×336 → 63 tokens | 912×672 → 266 | 1824×1344 → 1064 |
| 1024×1024 | 384×384 → 64 | 768×768 → 256 | 1584×1584 → 1089 |
| 1920×1080 | 528×288 → 66 | 1056×576 → 264 | 2112×1200 → 1100 |
| 3000×200 | 1536×96 → 64 | 3072×192 → 256 | 6192×384 → 1032 |
| **100×100** | **384×384 → 64** | **768×768 → 256** | **1584×1584 → 1089** |

三个观察，其中一个会花你的钱：

1. **长宽比确实被保留。** 640×480 和 480×640 产出转置的目标尺寸和完全相同的 token 数。15:1 的全景图仍是 16:1。没有压扁。
2. **你从来没真正用满预算。** 向下取整到 48 的倍数浪费掉 2–10% 的额度。而 `pixel_values` 反正会被补齐到完整的 `max_patches`（§4），所以那部分余量是你付了钱却没用上的真实算力。
3. **预算是目标而非上限 —— 小图会被放大。** 那张 100×100 的缩略图被**放大**到 768×768，并按 256 个 soft token 收费。双三次上采样不增加任何信息，只增加成本。如果你在批量处理缩略图或图标，请显式把 `max_soft_tokens` 调到 70。API 不会警告你这件事。

### 3. 不做 ImageNet 归一化，以及真正的缩放发生在哪

类属性说得异常直白：

```python
class Gemma4ImageProcessor(TorchvisionBackend):
    image_mean    = [0.0, 0.0, 0.0]
    image_std     = [1.0, 1.0, 1.0]
    do_normalize  = False
    do_rescale    = True        # rescale_factor = 1/255
```

归一化是关的，mean/std 是恒等，所以 `rescale_and_normalize` 就是一次除以 255。自己验证：

```python
out = ip([np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)], return_tensors="pt")
print(out["pixel_values"].min(), out["pixel_values"].max())   # 0.0  1.0
```

像素以 **[0, 1]** 抵达模型，未经标准化。模型想要的到 [-1, 1] 的平移被折叠进 patch embedding 层本身（04 章）。这在实践中有两个后果：如果你自己预处理图像，**绝不能**套上那串手指会自动打出来的 ImageNet 常数；如果你在调一个图像看起来发白的微调，这是第一个该查的地方。

还有一处值得注意的管道细节，它解释了一个奇怪的覆写：

```python
def _validate_preprocess_kwargs(self, **kwargs):
    # Gemma4 uses aspect_ratio_preserving_resize driven by patch_size, max_soft_tokens,
    # and pooling_kernel_size — not the standard `size` parameter.
    kwargs["do_resize"] = False
    super()._validate_preprocess_kwargs(**kwargs)
```

基类坚持 `do_resize=True` 就必须有 `size` 字典。Gemma 4 确实会 resize，但它的目标是算出来的、不是配出来的 —— 所以它对校验器撒了个谎。这个类上的 `size = None` 是刻意的。

### 4. Patch 化、位置、padding

Patch 化整段照搬自 SigLIP 2（源码里有 `# Copied from` 注释），纯粹是 reshape：

```python
patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
patched_image = patched_image.permute(1, 3, 2, 4, 0)
patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
```

一个 patch 变成 `16 × 16 × 3 = 768` 个数的向量。按行主序遍历网格：patch 下标从左到右、再从上到下。

位置用显式的 `"xy"` 索引约定构造：

```python
patch_grid = torch.meshgrid(torch.arange(patch_width), torch.arange(patch_height), indexing="xy")
real_positions = torch.stack(patch_grid, dim=-1).reshape(patches.shape[0], 2)
```

所以每一行是 **`(x, y)` —— 先列后行**。在 04 章手写 2D RoPE 时把它记反，一切都会照常运行、产出貌似合理的形状，并且悄悄地是错的。

然后一切被补齐到完整预算，padding **位置**标记为 `-1`：

```python
positions = torch.nn.functional.pad(positions, (0, 0, 0, padding_length), mode="constant", value=-1)
```

同时喂两张差别极大的图，看看结果：

```python
out = ip([img_480x640, img_100x100], return_tensors="pt")
# pixel_values             torch.Size([2, 2520, 768])
# image_position_ids       torch.Size([2, 2520, 2])
# num_soft_tokens_per_image  tensor([266, 256])
```

**`pixel_values` 的形状与输入图像无关，是固定的。** 2520 就是 `280 × 3²`，完整的 patch 预算。可变分辨率输入，恒定形状张量 —— 这正是批处理和 `torch.compile` 得以可行的原因，也正是 `(-1, -1)` 存在的理由：**位置**张量才是告诉塔这 2520 个槽位里哪些是真的那个东西。`num_soft_tokens_per_image` 是真实数量（这里是 266 和 256，不是 280），也正是 02 章 `replace_image_token` 用来决定占位段长度的值。

这个三元组 —— 固定形状的值、哨兵标记的位置、单独的真实计数 —— 你会在 05 章的音频那里原样再见一次。

### 5. `Gemma4ImageProcessorPil`

有两套实现：`Gemma4ImageProcessor` 继承 `TorchvisionBackend`（张量运算、可上 GPU、默认），`Gemma4ImageProcessorPil` 是 PIL 回退版。两者必须数值一致，这也是 resize 显式指定 `antialias=True` 的原因 —— PIL 的 `BICUBIC` 默认抗锯齿而 torchvision 的不。训练/服务之间的静默偏移就住在这种缝隙里。

## 设计空间

"任意网格 → 有界序列"的四种答案，以及各自优化的目标：

| 方案 | token 数 | 长宽比 | 大图的代价 |
|---|---|---|---|
| **固定正方形**（CLIP、LLaVA 1.0） | 恒定 | 被破坏 | 恒定 —— 细节直接丢失 |
| **Tiling**（InternVL、LLaVA-NeXT） | 按 tile 阶梯上升 | 每个 tile 内保留 | 分块增长；tile 接缝是真实问题 |
| **原生分辨率**（Qwen2-VL / Qwen3-VL） | 连续、无上界 | 保留 | 无限增长 —— 一张 4K 截图能花掉数千 token |
| **像素预算**（Gemma 4） | 从菜单里选 | 保留 | **由构造保证恒定** |

Gemma 4 这版最**可预测**：你在看到图像之前就知道 token 成本，而这正是规划上下文预算或给 API 定价所需要的。代价是它不自适应 —— 一份密集的 4K 文档和一张模糊快照都拿 280 个 token，除非你介入；而 Qwen 的原生分辨率方案会自动在文档上多花。菜单就是那个逃生舱，用好它是一个真实的工程决策，不是一个可以放着不管的默认值。

这个设计也顺带解释了 Gemma 4 的 OCR 行为。在 280 soft token 下，一张 1024×1024 的页面被渲染成 768×768 —— 整页大约 0.7 兆像素。小字撑不过去。调到 560 或 1120 是解法，04 章的 notebook 会测出悬崖在哪。

## 自测

1. 为什么两个维度必须能被 48 整除，而不只是被 16 整除？
2. 一张 4000×3000 的照片在默认预算下，求解器会挑什么目标尺寸？产出多少 soft token？
3. 你在把图像交给模型前自己套了常规的 ImageNet mean/std。什么坏了？为什么它不是一个异常？
4. 两张尺寸完全不同的图，`pixel_values` 都是 `[2, 2520, 768]`。关于它们实际尺寸的信息在哪？
5. 你在做扫描发票 OCR，模型读错小号数字。你改哪一个参数？代价是什么？
6. 位置是 `(x, y)`。如果你在自己写 2D RoPE 时假设成 `(row, col)`，会出什么问题？

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_pixels_to_patches.ipynb` | 手写复现 `get_aspect_ratio_preserving_size` 与 `convert_image_to_patches`，与库 `assert_close`；在真实照片上画出 patch 网格；用一张密集文字图扫描 soft token 菜单，看 OCR 在哪一档崩掉 | 🟢 CPU |
