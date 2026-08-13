# 03 · Image Processor — token 预算下的像素

**在数据流中的位置**：`image ──► Gemma4ImageProcessor ──► pixel_values + image_position_ids`

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
| `image_processing_gemma4.py` | `_SUPPORTED_SOFT_TOKENS` | 上面那张菜单 |
| | `get_aspect_ratio_preserving_size` | 预算求解：由图像尺寸 + patch 预算算出目标尺寸 |
| | `convert_image_to_patches` | 网格 → patch 序列 |
| | `pad_along_first_dim` | 把 patch 数不等的一批补齐 |
| | `Gemma4ImageProcessor` | 快速路径；`Gemma4ImageProcessorPil` 是 PIL 回退实现 |

## Notebooks

| Notebook | 内容 | 硬件 |
|---|---|---|
| `01_pixels_to_patches.ipynb` | 手写复现 `get_aspect_ratio_preserving_size` 与 `convert_image_to_patches`，与库 `assert_close`；在真实照片上画出 patch 网格；用一张密集文字图扫描 soft token 菜单，看 OCR 在哪一档崩掉 | 🟢 CPU |

> 📝 本章正文（源码走读与「设计空间」小节）以英文版为准，中文翻译进行中。完整内容见 [English edition](https://xinli95.github.io/Multimodal-101/03-image-processor/index.html)。
