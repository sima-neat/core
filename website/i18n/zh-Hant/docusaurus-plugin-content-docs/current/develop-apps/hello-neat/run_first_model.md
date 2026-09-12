---
title: "執行模型"
description: "在範例圖片上執行模型，使用 Neat。"
sidebar_position: 2
---

# 執行模型

使用與 [您好，Neat！](/develop-apps/hello-neat/minimal) 相同的運作目錄，來執行一個用於物件偵測的實際模型。
這個應用程式會載入一個 YOLOv8 模型，讀取一張範例圖片，執行推論，解碼邊界框，並印出偵測到的物件數量。

本頁介紹兩個 Neat 概念：

* [`Model`](/develop-apps/development-workflow/model) 載入編譯後的模型套件，並提供 `run(...)` 進入點。
* [`ModelOptions`](/tutorials/configure-model-options) 告訴 Neat 如何準備圖片並解碼偵測器的輸出。

您目前不需要完全掌握整個 API；現在，請專注於 `Model` 和 `ModelOptions` 如何協同工作，以執行一個編譯後的模型。

![您好，這是 Neat YOLOv8 的流程。](@site/../docs/images/hello-neat-yolov8-flow.svg)

## 取得模型和範例圖片

1. **建立一個「assets」目錄**，我們將在其中儲存模型和輸入圖片：
    ```bash
    mkdir -p assets
    cd assets
    ```
2. **下載模型：**
    ```bash
    sima-cli modelzoo -v 2.0.0 get yolo_v8s
    ```
    :::note sima-cli 模型下載
    如果 `sima-cli` 將模型寫入的目錄與 `assets` 目錄不同，請將該檔案複製到 `assets/yolo_v8s_mpk.tar.gz`。
    :::
3. **從檔案下載範例圖片**，並將其儲存為 `tutorial_sample_image.png`，儲存在 `assets` 目錄中。

    [開啟或下載範例圖片](../../images/tutorial_sample_image.png)。
4. **傳回您的專案目錄：**
    ```bash
    cd ..
    ```

## 逐步說明

我們將以 [您好，Neat！](/develop-apps/hello-neat/minimal) 中的程式碼為基礎：保留相同的 `CMakeLists.txt`（它已經連結了 Neat 和 OpenCV），並將程式碼主體替換為以下四個步驟。每個步驟都是最終程式碼的一個小部分——請依序閱讀它們，然後取得 [完整程式](#full-program) 並複製貼上後執行。在任何程式碼區塊中選擇一個語言標籤；您的選擇將遵循網站範圍內的選擇器。

### 1. 載入並調整圖像大小 {#step-load-image}

YOLOv8s 需要一個固定大小的 `640×640` BGR 圖像，因此我們使用 OpenCV 讀取範例圖像並調整其大小。這僅僅是普通的圖像輸入/輸出操作——目前還沒有使用 Neat 的 API。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
cv::Mat load_sample_image() {
  cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png", cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to load sample image");
  cv::resize(bgr, bgr, cv::Size(640, 640));   // YOLOv8s input size
  return bgr;
}
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
def load_image(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.resize(bgr, (640, 640))   # YOLOv8s input size
```

</CodeTab>
</CodeTabs>

### 2. 描述輸入和解碼，並使用 `ModelOptions` {#step-model-options}

`ModelOptions` 是您的影像與模型之間的執行階段合約。它在此宣告兩件事：Neat 應如何預處理解碼後的像素，然後再進行推論，以及如何將檢測器的原始輸出解碼為框。`decode_type` 選擇 YOLOv8 解碼器，閾值用於刪除較弱或重疊的框，而 `top_k` 則限制了框的數量。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::Model::Options opt;
opt.preprocess.kind = simaai::neat::InputKind::Image;
opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
opt.score_threshold = 0.55f;
opt.nms_iou_threshold = 0.5f;
opt.top_k = 100;
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
opt = pyneat.ModelOptions()
opt.preprocess.kind = pyneat.InputKind.Image
opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
opt.decode_type = pyneat.BoxDecodeType.YoloV8
opt.score_threshold = 0.55
opt.nms_iou_threshold = 0.5
opt.top_k = 100
```

</CodeTab>
</CodeTabs>

### 3. 載入模型並執行推論 {#step-run}

從已編譯的 `.tar.gz` 封包和選項中建構一個 `Model`，然後呼叫 `run(...)` 並設定逾時時間。它會同步執行，並傳回輸出張量。`timeout_ms` 可讓執行卡住時，不會只是靜止不動，而是會立即失敗。

**C++** 每次會傳遞一個 `cv::Mat` 作為模型輸入。**Python** 首先將 NumPy 影像包裝成帶有 `BGR` 標籤的 `Tensor`，以便 Neat 知道位元組設定，然後傳遞 `[tensor]`，因為 Python 模型輸入是一個序列。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::Model yolo("assets/yolo_v8s_mpk.tar.gz", opt);
simaai::neat::TensorList outputs = yolo.run(std::vector<cv::Mat>{bgr}, /*timeout_ms=*/2000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
model = pyneat.Model(str(mpk), opt)
tensor = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)
outputs = model.run([tensor], timeout_ms=2000)
```

</CodeTab>
</CodeTabs>

### 4. 讀取檢測數量 {#step-read}

由於已設定 `decode_type`，因此第一個輸出張量包含已解碼的框。BBOX 張量以一個 `uint32` 檢測數量開始，因此我們讀取其前四個位元組。完整的序列格式（每個框的座標、分數和類別）已在 [從模型輸出的結果中讀取檢測框](/tutorials/read-detection-boxes) 中說明。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
std::uint32_t detections = 0;
if (!outputs.empty()) {
  simaai::neat::Mapping view = outputs.front().map_read();
  if (view.size_bytes >= sizeof(detections))
    std::memcpy(&detections, view.data, sizeof(detections));
}
std::cout << "detections=" << detections << "\n";
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
detections = 0
if len(outputs) > 0:
    payload = outputs[0].to_numpy(copy=False).tobytes()
    detections = struct.unpack_from("<I", payload, 0)[0] if len(payload) >= 4 else 0
print(f"detections={detections}")
```

</CodeTab>
</CodeTabs>

## 完整程式碼 {#full-program}

保留來自 [您好，Neat！](/develop-apps/hello-neat/minimal) 的 `CMakeLists.txt` 檔案（它已經將應用程式與 Neat 和 OpenCV 連結），並將程式碼主體替換為以下完整檔案。標記的行是核心的三個步驟：建立 `Model`、建構輸入，以及呼叫 `run()`。

<details>
<summary>顯示完整的程式。</summary>

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp title="main.cpp" {43-45}
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

cv::Mat load_sample_image() {
  cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png", cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to load sample image");

  // YOLOv8s expects a 640 x 640 input in this tutorial.
  cv::resize(bgr, bgr, cv::Size(640, 640));
  return bgr;
}

int main() {
  // 1. Load the sample image and resize it for the model.
  cv::Mat bgr = load_sample_image();

  // 2. Tell Neat how to preprocess pixels and decode YOLO boxes.
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  opt.score_threshold = 0.55f;
  opt.nms_iou_threshold = 0.5f;
  opt.top_k = 100;

  // 3. Load the compiled model package and run inference.
  simaai::neat::Model yolo("assets/yolo_v8s_mpk.tar.gz", opt);
  simaai::neat::TensorList outputs = yolo.run(std::vector<cv::Mat>{bgr}, /*timeout_ms=*/2000);

  // 4. The BBOX output starts with a uint32 detection count.
  std::uint32_t detections = 0;
  if (!outputs.empty()) {
    simaai::neat::Mapping view = outputs.front().map_read();
    if (view.size_bytes >= sizeof(detections))
      std::memcpy(&detections, view.data, sizeof(detections));
  }
  std::cout << "detections=" << detections << "\n";
  std::cout << "[OK] YOLOv8 completed\n";
  return 0;
}
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python title="hello_neat.py" {50-53}
#!/usr/bin/env python3
from __future__ import annotations

import struct
import sys
from pathlib import Path

try:
    import pyneat
except ImportError:
    sys.exit(
        "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
        "Run: source ~/pyneat/bin/activate"
    )

import cv2


def load_image(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    # YOLOv8s expects a 640 x 640 input in this tutorial.
    return cv2.resize(bgr, (640, 640))


def main() -> int:
    mpk = Path("assets/yolo_v8s_mpk.tar.gz")
    image = Path("assets/tutorial_sample_image.png")

    # 1. Load the sample image and resize it for the model.
    bgr = load_image(image)

    # 2. Tell Neat how to preprocess pixels and decode YOLO boxes.
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = 0.55
    opt.nms_iou_threshold = 0.5
    opt.top_k = 100

    # 3. Load the compiled model package and run inference.
    model = pyneat.Model(str(mpk), opt)
    tensor = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)
    outputs = model.run([tensor], timeout_ms=2000)

    # 4. The BBOX output starts with a uint32 detection count.
    detections = 0
    if len(outputs) > 0:
        payload = outputs[0].to_numpy(copy=False).tobytes()
        detections = struct.unpack_from("<I", payload, 0)[0] if len(payload) >= 4 else 0
    print(f"detections={detections}")

    print("[OK] YOLOv8 completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

</CodeTab>
</CodeTabs>

</details>

## 建置與執行

從您的專案目錄（包含 `assets/` 的目錄）執行這些指令。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

使用與「Hello Neat!」相同的指令重新建置：

<ShellCommand prompt="sdk|devkit">
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
</ShellCommand>

然後執行二進位檔：

<ShellCommand prompt="devkit">
./build/sima_neat_hello
</ShellCommand>

<ShellCommand prompt="sdk">
dk build/sima_neat_hello
</ShellCommand>

</CodeTab>
<CodeTab label="Python" lang="python">

執行腳本：

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 hello_neat.py
</ShellCommand>

<ShellCommand prompt="sdk">
dk hello_neat.py
</ShellCommand>

</CodeTab>
</CodeTabs>

您應該會看到類似以下的偵測摘要：

```text
detections=3
[OK] YOLOv8 completed
```

確切的數量可能會因模型套件和執行階段版本而異。重要的是，應用程式可以成功建置、執行，並達到 `[OK] YOLOv8 completed`。

## 您建置了什麼

這個範例遵循與較大的 Neat 應用程式相同的基本流程：

- 載入已編譯的模型套件（`.tar.gz`），作為一個 `Model`。
- 將輸入影像轉換為模型預期的格式。
- 透過 Neat 執行階段階段執行推論。
- 將原始檢測器輸出解碼為邊界框。

若要更深入地了解邊界框解碼、閾值、NMS 和檢測器輸出結構，請繼續閱讀 [從模型輸出的結果中讀取檢測框](/tutorials/read-detection-boxes)。

## 後續步驟

一旦 YOLOv8 執行完成，請繼續使用更廣泛的 SiMa.ai Neat 學習資源：

- 繼續使用 **[執行應用程式](/develop-apps/hello-neat/run_an_app)**，將相同的模型組合成一個 `Graph` 應用程式——一個命名的輸入 → 模型 → 輸出管線，您可以建立一次，然後使用推/拉方式驅動它——而不是直接呼叫 `Model.run(...)`。
- 學習 [核心程式設計模型](/develop-apps/development-workflow/overview)，它解釋了主要的 Neat 概念，例如模型、圖和執行。
- 按照 [教學指南](/tutorials/)，逐步了解特定的概念和工作流程。
- 探索 [應用程式入口網站](https://apps.neat.sima.ai/portal) 上的精選應用程式，其中包含 [GitHub 上的應用程式儲存庫](https://github.com/sima-neat/apps) 中的原始程式碼。
