# 03 · Image Processor — token 预算下的像素

本章可以独立阅读。它要回答的问题是：**一张普通图像，如何变成 Gemma 4 的 learned vision tower 能接收的固定形状张量？**

```text
PIL 图像 / NumPy array / tensor
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
      Gemma4VisionModel                （04 章）
              │
              ▼
      visual soft-token embeddings
```

本章跟的是中间这个方框：它分配图像算力并记录几何位置，但不含 learned weights，也不判断图像表达什么。[02 章](../02-text-io/index.md)用 `num_soft_tokens_per_image` 在文本 prompt 中预留正确数量的位置；[04 章](../04-vision-tower/index.md)接过 `pixel_values` 和 `image_position_ids`，产出学到的视觉特征。

每个视觉语言模型都要回答同一个问题：图像是任意尺寸的二维网格，Transformer 要的是长度有界的一维序列，怎么转？领域里试过的答案就是 VLM 图像处理的全部历史：把所有图都塞进固定正方形（CLIP，对带文字的东西损失惨重）、把大图切成固定 crop（InternVL），或者在设定的上下界内让保持长宽比的网格随图像增长（Qwen2-VL，能自适应但成本不固定）。

Gemma 4 的答案是**像素预算**：保持长宽比，把图缩放到总像素数落进预算内，并强制两边都能被 48 整除。你从五档菜单中选择一个上限。padding 后的 patch 张量形状因此是固定且预先已知的；由于图像两边都要向下取整，真实 soft token 数可能略低于这个上限。

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

9 个 patch（3×3）池化成 1 个 soft token，这就是两列之间 9 倍关系的由来。

这里的 `max_soft_tokens` **不是一个可以随便填写的整数上限**。「对这个集合做校验」的意思是：processor 会检查它是不是 `(70, 140, 280, 560, 1120)` 的成员。「raises」是 Python 的简写，完整意思是「抛出异常」：例如 `Gemma4ImageProcessor(max_soft_tokens=300)` 会立刻抛出 `ValueError`，而不是悄悄把 300 舍入到某一档。在 preprocess 时临时覆盖这个值，也会经过同样的检查。所以这五个数是配置档位，不是五个示例。

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

1. **长宽比先近似保留，再被量化。** 640×480 和 480×640 产出转置的目标尺寸和完全相同的 token 数。两边分别取整到 48 的倍数后，预算 70 下的 4:3 会变成约 1.29:1，15:1 的全景图会变成 16:1。这是小幅几何误差，不是某些固定画布 pipeline 那样把整张图彻底压成正方形。
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

### 4. 从二维图像到可组成 batch 的序列

下面四个词很容易混成一件事，先把各自的职责分开：

| 术语 | 它是什么 | 为什么需要它 |
|---|---|---|
| **patch** | 一个 16×16 的 RGB 小方块，摊平成 768 个数 | 把像素网格变成 Transformer 能接收的单位 |
| **position ID** | patch 在 patch 网格中的 `(x, y)` 坐标 | patch 变成一维序列后，仍保留二维布局 |
| **padding slot** | 值全为零、位置为 `(-1, -1)` 的假 patch | 让 batch 中每张图的张量形状相同 |
| **soft token** | vision tower 把 3×3 个 patch 池化后得到的 learned result | 它最终会占用语言模型的 context 位置 |

#### 第一步：切成 patch

沿用前面表格的第一行：640×480 输入在预算 70 下会 resize 为 **432×336**（宽 × 高）。使用 16×16 patch 后，它会成为 27 列、21 行的网格：

```text
432×336 像素
    │ 切成 16×16 的小方块
    ▼
27×21 patch 网格 = 567 个 patch
    │ 稍后，每个 3×3 patch 块做一次池化
    ▼
9×7 网格 = 63 个 soft token
```

processor 只执行第一次转换。3×3 池化发生在 04 章的 learned vision tower 内；`num_soft_tokens_per_image` 只是提前记下届时会得到的数量：`567 / 9 = 63`。

Patch 化整段照搬自 SigLIP 2（源码里有 `# Copied from` 注释），纯粹是 reshape：

```python
patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
patched_image = patched_image.permute(1, 3, 2, 4, 0)
patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
```

一个 patch 变成 `16 × 16 × 3 = 768` 个数的向量。上面的 567 个 patch 因此会成为形状为 `[567, 768]` 的张量。它按行摊平：第一行从左到右，然后第二行从左到右，依此类推。

#### 第二步：记住每个 patch 原来在哪里

位置用显式的 `"xy"` 索引约定构造：

```python
patch_grid = torch.meshgrid(torch.arange(patch_width), torch.arange(patch_height), indexing="xy")
real_positions = torch.stack(patch_grid, dim=-1).reshape(patches.shape[0], 2)
```

所以每一行是 **`(x, y)` —— 先列后行**。对上面的 27×21 网格，序列从 `(0,0), (1,0), …, (26,0), (0,1), …` 开始，到 `(26,20)` 结束。只有一维序号还不够：序号 27 之所以表示「第二行开头」，前提是模型知道这张图一行有 27 个 patch。显式的 `(x, y)` 坐标在摊平后仍保存了这个事实。在 04 章手写 2D RoPE 时把顺序记反，所有张量形状依然合法，但横向和纵向位置信息会被交换。

#### 第三步：padding 到统一形状

不同长宽比会得到不同数量的真实 patch。例如在预算 70 下，batch 张量无法把上面的 `[567, 768]` 和正方形图像的 `[576, 768]` 直接堆在一起，所以每张图都要补到该档完整的 630-patch 预算：patch 值补零，对应的位置标成 `(-1, -1)`：

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

把三个输出放在一起读：

- `pixel_values[i]` 在默认档位总有 2,520 行：先放真实 patch 向量，后面是全零 padding。
- `image_position_ids[i]` 给每个真实行写入 `(x, y)` 坐标，给每个 padding 行写入 `(-1, -1)`。vision tower 会从这个哨兵值构造 attention mask，因此不会把 padding 当作图像内容。
- `num_soft_tokens_per_image[i]` 是池化后真实 3×3 分组的数量。这里是 266 和 256，不是 280；02 章用它决定文本序列中 placeholder 段的长度。

到这里，可变分辨率输入变成了形状恒定的 patch 张量，同时没有丢掉「哪里是真数据、哪里是 padding」的边界。这使 batching 和 compilation 更容易。注意：padding 发生在 vision tower **之前**，soft-token pooling 发生在 tower **内部**。

这个三元组 —— 固定形状的值、哨兵标记的位置、单独的真实计数 —— 你会在 05 章的音频那里原样再见一次。

### 5. 为什么需要 PIL fallback？

这里的「fallback」是指**另一套预处理后端**，不是「图像损坏时改用 PIL 重试」，也不是「GPU 出错后的恢复机制」。安装了 torchvision 时，默认的 `Gemma4ImageProcessor` 使用 torchvision tensor 运算；它可以让 tensor 输入留在 accelerator 上，也适合 compiled tensor pipeline。`Gemma4ImageProcessorPil` 用 PIL 和 NumPy 实现同一套 resize → rescale → patchify → pad contract，让没有 torchvision 的轻量 CPU 环境也能运行 processor。Hugging Face 的 [`AutoImageProcessor`](https://huggingface.co/docs/transformers/main/en/image_processors) 可以在 torchvision 不可用时选择 PIL；调用者也可以为支持双后端的 processor 显式指定后端。

这是 common practice 吗？**用 PIL 做预处理很常见；提供两套可互换后端也是常见的兼容性工程，但它不是模型算法的必要组成，也不是所有现代 processor 都有。** PIL 多年来一直是 Python 图像处理的常规路径，今天仍适合可移植性和复现旧 pipeline；torchvision 则是当前的性能路径。Gemma 4 保留两套，是因为这里的预处理没有 learned computation，可以用任一后端表达。

代价是必须维护数值一致性。不同 resize library 在像素边界上可能略有差异，从而造成无声的 train/serve skew。Gemma 4 因此在 torchvision 的 bicubic resize 上显式传入 `antialias=True`，尽可能贴近 PIL 的 antialiased bicubic 行为。两套后端应实现相同语义，但浮点数和插值细节仍可能造成极小差异；需要精确复现时，训练与部署始终使用同一后端最稳妥。

## 设计空间

下面几种方案解决的是同一个接口问题——把任意 `H×W` 像素变成 Transformer token——但它们对「在哪里丢分辨率」和「成本是否随输入变化」给出了不同答案。

| 方案 | 机制 | token 成本 | 优势 | 典型失败模式 |
|---|---|---|---|---|
| **固定画布**（[CLIP](https://arxiv.org/abs/2103.00020)、早期 LLaVA） | 每张图都 resize 到一个训练分辨率；通过 crop、padding，有时也通过拉伸来得到正方形 | 完全固定 | 实现简单、batch 紧凑、延迟可预测 | 下采样丢细节；crop 可能丢内容；拉伸改变几何 |
| **任意分辨率 tiling**（[LLaVA-NeXT](https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py)、[InternVL 1.5](https://arxiv.org/abs/2404.16821)） | 选择一个画布/网格，切成固定大小的 crop，通常再附上一张全局 thumbnail | 以整块 tile 为单位阶梯上升，直到配置的 tile 上限 | 局部 tile 保住小字和小物体，thumbnail 提供全局上下文 | 重复编码和 padding 增加成本；模型要对齐 tile、边界与全局坐标 |
| **动态/原生网格**（[Qwen2-VL](https://arxiv.org/abs/2409.12191)） | 保留一张等比例网格，让 patch 数在设定的最小/最大像素界限内随面积变化 | 可变，大致与保留的面积成正比 | 小图可以便宜，大而精细的图可以获得更多 token | 可变序列长度增加 batching 和延迟管理难度；上限太宽会让大图昂贵 |
| **选定像素预算**（Gemma 4） | 把每张图缩放到所选面积档附近，两边取整到与 pooling 兼容的倍数，再把 patch pad 到档位上限 | padding 后的 patch shape 固定；真实 soft token 数接近但不超过所选档 | 内存形状可预测，有直接的质量/成本旋钮 | 不看内容密度；小图被放大，稠密大图被下采样 |

### 固定画布：无论输入是什么，都花同样的钱

经典 CLIP 风格 pipeline 给 vision encoder 的永远是同一种正方形分辨率。「固定正方形」并不代表只有一种 resize 规则：可以 resize 后 center crop，可以 letterbox padding，也可以拉伸。三者都会得到固定 patch 网格，让训练和 batching 非常简单。无法回避的代价是：超宽截图和竖版文档都必须塞进同一画布。crop 牺牲覆盖范围，padding 浪费网格面积，拉伸牺牲几何；严重下采样则会在所有变体里牺牲细节。

### Tiling：以整张 crop 为单位购买细节

Tiling 不把整张大图缩进一个小正方形。LLaVA-NeXT 的 AnyRes 会选择候选画布，把图等比例放入并 padding，再切成 vision-encoder 尺寸的 crop；InternVL 风格的 dynamic high resolution 同样会选择 tile 网格，并通常附上一张整图 thumbnail。这样模型既有高分辨率局部视图，也有全局上下文。成本不是逐 patch 而是逐 tile 增长。它很适合文档与大场景，但部分像素可能被重复编码，候选画布里的 padding 可能浪费，而且模型需要相应的结构或训练来理解各个 crop 的坐标如何拼回整图。

### 动态/原生网格：让图像决定账单

「原生分辨率」是方便的简称，不是说原始尺寸原封不动地进入模型。Qwen2-VL 会把图像 resize/取整到 patch-compatible 网格，并用 [`min_pixels` 和 `max_pixels`](https://huggingface.co/docs/transformers/model_doc/qwen2_vl#usage-tips) 夹住范围；在这个区间内，保留面积越大，visual token 越多。这一点比 Gemma 4 更自适应：小图标可以只用少量 token，大文档可以使用很多。代价是内存和延迟依赖输入，batch 也更不整齐。配置了 `max_pixels` 后它并非真的「无上界」——只是使用连续、由用户配置的边界，而不是 Gemma 4 的五个离散档位。

### Gemma 4 的选定预算：先决定账单

Gemma 4 要调用者先选档，然后把稠密的 4K 文档和模糊快照都缩放到同一个目标面积附近。它在运维上最可预测：看到图像之前，你就知道 padding 后的 patch shape 和最大的 context token 数。**但在长宽比取整完成前，你不知道确切的真实 token 数。** 代价是不理解内容：processor 不知道一张图里有发票小字，另一张没有。五档菜单就是手动的逃生舱；选好它是一项真实的工程决策，而不是一个可以放着不管的默认值。

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
