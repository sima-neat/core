---
title: "預處理 ROI 列表"
description: "在一個或多個影像上，針對多個執行階段的感興趣區域 (ROI) 視窗執行預處理。"
sidebar_position: 7
---

# 預處理 ROI 列表

當您的應用程式已經有一個或多個影像視窗，並且希望對每個視窗進行預處理，使其成為模型輸入時，請使用預處理 ROI 清單：調整大小或使用信箱式調整，色彩轉換，正規化，量化，並可選擇進行鑲嵌。

這對於第二階段的分類器、偵測器串聯、追蹤器，或任何從一批原始影像產生動態裁剪清單的工作流程來說，都是一個理想的工具。

## 最簡示例

C++：

```cpp
#include <neat.h>
#include <opencv2/imgcodecs.hpp>

using namespace simaai::neat;

Model model("/path/to/model.tar.gz");

std::vector<cv::Mat> images = {
    cv::imread("/data/camera0.jpg", cv::IMREAD_COLOR),
    cv::imread("/data/camera1.jpg", cv::IMREAD_COLOR),
};

std::vector<PreprocessRoi> rois = {
    {0, 0, 0, 320, 240},
    {0, 120, 80, 160, 160},
    {1, -20, 30, 224, 224},
};

TensorList out = stages::Preproc(images, model, rois);

for (std::size_t i = 0; i < out.size(); ++i) {
  const auto& meta = out[i].semantic.preprocess;
  // out[i] is the preprocessed tensor for rois[i].
  // meta carries the ROI and inverse-coordinate breadcrumbs.
}
```

Python：

```python
import cv2
import pyneat

model = pyneat.Model("/path/to/model.tar.gz")

images = [
    cv2.imread("/data/camera0.jpg", cv2.IMREAD_COLOR),
    cv2.imread("/data/camera1.jpg", cv2.IMREAD_COLOR),
]

rois = [
    pyneat.PreprocessRoi(0, 0, 0, 320, 240),
    pyneat.PreprocessRoi(0, 120, 80, 160, 160),
    pyneat.PreprocessRoi(1, -20, 30, 224, 224),
]

out = pyneat.stages.preproc(
    images,
    model,
    rois=rois,
    image_format=pyneat.PixelFormat.BGR,
)

for i, tensor in enumerate(out):
    meta = tensor.semantic.preprocess
    # tensor is the preprocessed output for rois[i].
    # meta carries the ROI and inverse-coordinate breadcrumbs.
```

## 設定調整大小、寬螢幕顯示和標準化。

ROI 列表預處理使用與全畫面預處理相同的模型預處理選項。

C++：

```cpp
Model::Options opt;
opt.preprocess.resize.enable = AutoFlag::On;
opt.preprocess.resize.width = 640;
opt.preprocess.resize.height = 640;
opt.preprocess.resize.mode = ResizeMode::Letterbox;
opt.preprocess.resize.pad_value = 114;
opt.preprocess.resize.scaling_type = "BILINEAR";
opt.preprocess.normalize.enable = AutoFlag::On;
opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
opt.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
opt.preprocess.tessellate.enable = AutoFlag::Auto;

Model model("/path/to/model.tar.gz", opt);
TensorList out = stages::Preproc(images, model, rois);
```

Python：

```python
opt = pyneat.ModelOptions()
opt.preprocess.resize.enable = pyneat.AutoFlag.On
opt.preprocess.resize.width = 640
opt.preprocess.resize.height = 640
opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
opt.preprocess.resize.pad_value = 114
opt.preprocess.resize.scaling_type = "BILINEAR"
opt.preprocess.normalize.enable = pyneat.AutoFlag.On
opt.preprocess.normalize.mean = [0.0, 0.0, 0.0]
opt.preprocess.normalize.stddev = [1.0, 1.0, 1.0]
opt.preprocess.tessellate.enable = pyneat.AutoFlag.Auto

model = pyneat.Model("/path/to/model.tar.gz", opt)
out = pyneat.stages.preproc(
    images,
    model,
    rois=rois,
    image_format=pyneat.PixelFormat.BGR,
)
```

支援的 `scaling_type` 參數包括 `BILINEAR`、`NEAREST_NEIGHBOUR`、`BICUBIC`、`INTERAREA` 和 `NO_SCALING`。`NEAREST_NEIGHBOR` 和 `INTER_AREA` 也是可接受的別名。

## 批次語義

| 輸入 | 意義 |
| --- | --- |
| `images` | 原始影像批次。當提供 ROI（感興趣區域）時，向量/列表必須不為空。Python 接受 uint8 格式的 HW/HWC NumPy/Torch/`pyneat.Tensor` 影像輸入。|
| `rois` | 執行階段的 ROI 列表。輸出張量的順序與此向量的順序相符。|
| `PreprocessRoi::batch_index` | 選擇從 `images` 中的哪個來源影像讀取 ROI。|
| `PreprocessRoi::x`、`y` |，表示原始影像框架中，預處理區域左上角的座標。允許使用負值。|
| `PreprocessRoi::width`, `height` | ROI 尺寸。兩者都必須為正數。 |

在 ROI 列表呼叫中，所有來源影像都必須具有相同的寬度、高度、OpenCV 類型和通道數。階段 API 支援用於 ROI 列表的 8 位元 RGB/BGR (`CV_8UC3`) 和 GRAY/GRAY8 (`CV_8UC1`) 格式的來源影像。

對於 Python，對於 `cv2.imread` 影像，傳遞 `image_format=pyneat.PixelFormat.BGR`，對於 RGB 陣列，傳遞 `RGB`，對於 HW 灰階陣列，傳遞 `GRAY8`。CHW 張量會被 `pyneat.stages.preproc` 拒絕；在呼叫階段之前，將它們轉換為 HWC 格式。

## 輸出語義

`stages::Preproc(images, model, rois)` 會傳回：

- 對於有效的非空請求，請提供 `out.size() == rois.size()`。
- 從 ROI `rois[i]` 產生的輸出張量 `out[i]`。
- 模型路徑所選擇的資料類型和佈局，包括 BF16 或 INT8，以及密集或鑲嵌式輸出；
- 每個輸出的 `tensor.semantic.preprocess` 中繼資料。

每個輸出的預處理元資料包含選取的感興趣區域 (ROI)、信箱幾何形狀、正規化/量化/細分旗標，以及一個仿射變換，用於將模型/預處理後的座標映射回原始來源框架座標系統。

## 超出畫面的感興趣區域 (ROI)

感興趣區域 (ROI) 可能會延伸到原始影像的範圍之外：

```cpp
std::vector<PreprocessRoi> rois = {
    {0, -16, -16, 128, 128},
    {0, image.cols - 64, image.rows - 64, 128, 128},
};
```

將影像中的區域複製，並使用「Preproc pad」值對影像外的區域進行填充。這樣可以保持輸出影像的形狀穩定，並避免在影像邊緣附近出現特殊情況。

## 寬螢幕比例和畫面比例

對於 `ResizeMode::Letterbox` 模式，預處理程序會針對每個感興趣區域 (ROI) 計算信箱比例和填充值。因此，一個較高的感興趣區域和一個較寬的感興趣區域可能會產生不同的 `scaled_width`、`scaled_height`、`pad_left` 和 `pad_top` 中繼資料，即使它們共享相同的目標大小。

後續程式碼應讀取中繼資料，而不是根據假設重新計算填充值。

## 驗證檢查清單

在責備推論結果之前，請先確認：

- 影像向量不為空；
- 每個影像都具有相同的尺寸、OpenCV 類型和通道數；
- Python 的輸入是 uint8 格式的 HW/HWC 圖像，而不是 CHW 格式的張量。
- 每個感興趣區域 (ROI) 都具有有效的 `batch_index`。
- 每個感興趣區域（ROI）都具有正的 `width` 和 `height`。
- 對於 ROI 列表模式，原始影像的格式為 RGB、BGR、GRAY 或 GRAY8。
- 調整後的尺寸模式和正規化方式與模型的訓練前處理方式相符。
- 下游協調消費者會使用 `tensor.semantic.preprocess` 中繼資料。

## 涵蓋此行為的測試。

這個儲存庫包含針對使用者介面路徑和功能性投資報酬行為的快速覆蓋測試：

- `preproc_roi_batch_functional_test`
- `preproc_roi_user_smoke_test`

這些功能涵蓋多個影像、多個感興趣區域 (ROI)、超出畫面的填補、調整大小/黑邊處理行為、標準化輸出，以及密集/鑲嵌式的 BF16/INT8 運算路徑。

## 請參閱

- [預處理節點](/reference/nodes/preproc)
- [在進行推論之前，請先預處理影像。](/tutorials/preprocess-images)
- [資料格式](/develop-apps/advanced-concepts/data_formats)
- [BoxDecode 解碼類型](/reference/boxdecode_decode_types)
