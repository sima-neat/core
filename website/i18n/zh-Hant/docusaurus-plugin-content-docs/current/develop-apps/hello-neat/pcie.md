---
title: "PCIe 協同處理"
description: "執行您的第一個 PCIe 協同處理推論。"
sidebar_position: 4
---

# PCIe 協同處理

![PCIe 協同處理流程](@site/../docs/images/hello-neat-pcie-coprocessing-flow.svg)

PCIe 協同處理將應用程式控制和輸入準備保留在主機機器上，而連接的 Modalix PCIe 卡則執行模型。在本範例中，主機會讀取、調整大小和正規化一張影像，然後將準備好的張量傳送到卡上進行 ResNet-50 推論，並列印傳回的分類輸出。

## 在開始之前

您需要：

- 在主機機器上安裝 [Neat PCIe 主機封裝](/getting-started/neat-library/pcie-host/)；
- 在卡上安裝相容的 Neat Library 版本；以及
- 從主機可存取的卡管理介面。卡 0 預設使用 `10.0.0.2`。

## 取得模型和影像

在主機機器上建立一個工作目錄，並從 Model Zoo 下載 ResNet-50：

<ShellCommand prompt="pcie-host">
mkdir -p pcie-host-quickstart/assets
cd pcie-host-quickstart
sima-cli modelzoo get resnet_50
</ShellCommand>

將下載的壓縮檔放在這個目錄中，檔案名稱為 `resnet_50.tar.gz`。

下載 [拉布拉多犬範例圖片](../../images/hello-neat-pcie-labrador.jpg)，並將其儲存為 `assets/sample.png`。

圖片的原始素材為 [黃色拉布拉多犬，看起來很新。](https://commons.wikimedia.org/wiki/File:YellowLabradorLooking_new.jpg)，由 Elf 拍攝，Djmirko 和 FT2 進行修改，並以 [知識共享署名-相同方式分享 3.0](https://creativecommons.org/licenses/by-sa/3.0/) 授權。

您的工作目錄現在應該包含：

```text
pcie-host-quickstart/
├── assets/
│   └── sample.png
└── resnet_50.tar.gz
```

## 建立應用程式

選擇 Python 或 C++，並在工作目錄中建立應用程式。

<CodeTabs>
<CodeTab label="Python" lang="python">

建立 `pcie_host.py`：

```python title="pcie_host.py"
import cv2
import numpy as np

import pyneatpcie as pcie


def load_input(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    bgr = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    stddev = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (rgb - mean) / stddev

connection = pcie.ConnectionOptions(
    card_host="10.0.0.2",
    card_id=0,
    queue=0,
)

with pcie.Model("resnet_50.tar.gz", connection=connection) as model:
    input_spec = model.info().inputs[0]
    input_tensor = pcie.Tensor.from_numpy(
        load_input("assets/sample.png"),
        copy=True,
        route_name=input_spec.name,
    )
    model.build(readiness_timeout_ms=180000)
    outputs = model.run([input_tensor], timeout_ms=30000)

output = outputs[0].to_numpy().reshape(-1)
print("output:", outputs[0].route.name, outputs[0].shape)
print("top1:", int(np.argmax(output)))
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

建立 `pcie_host.cpp`：

```cpp title="pcie_host.cpp"
#include <simaai/neat/pcie/Model.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pcie = simaai::neat::pcie;

pcie::Tensor load_input(const std::string& path,
                        const std::string& route_name) {
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to read image: " + path);

  cv::resize(bgr, bgr, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  if (!rgb.isContinuous())
    rgb = rgb.clone();

  constexpr float mean[] = {0.485F, 0.456F, 0.406F};
  constexpr float stddev[] = {0.229F, 0.224F, 0.225F};
  std::vector<float> input(rgb.total() * rgb.channels());
  for (int row = 0; row < rgb.rows; ++row) {
    const auto* pixels = rgb.ptr<std::uint8_t>(row);
    for (int col = 0; col < rgb.cols; ++col) {
      for (int channel = 0; channel < 3; ++channel) {
        const std::size_t index =
            (static_cast<std::size_t>(row) * rgb.cols + col) * 3 + channel;
        const float value = pixels[col * 3 + channel] / 255.0F;
        input[index] = (value - mean[channel]) / stddev[channel];
      }
    }
  }
  return pcie::Tensor::from_vector(
      std::move(input), {rgb.rows, rgb.cols, rgb.channels()}, route_name);
}

int main() {
  pcie::ConnectionOptions connection;
  connection.card_host = "10.0.0.2";
  connection.card_id = 0;
  connection.queue = 0;

  pcie::Model model("resnet_50.tar.gz", {}, connection);
  const pcie::ModelInfo info = model.info();
  if (info.inputs.empty())
    throw std::runtime_error("model reports no inputs");

  model.build(/*readiness_timeout_ms=*/180000);
  pcie::TensorList outputs = model.run(
      load_input("assets/sample.png", info.inputs[0].name),
      /*timeout_ms=*/30000);

  if (outputs.empty() || outputs[0].dtype != pcie::TensorDType::Float32 ||
      outputs[0].data == nullptr || outputs[0].size_bytes == 0)
    throw std::runtime_error("expected one FP32 classification output");

  const auto* scores = static_cast<const float*>(outputs[0].data);
  const std::size_t count = outputs[0].size_bytes / sizeof(float);
  const auto best = std::max_element(scores, scores + count);

  std::cout << "output: " << outputs[0].route.name << " [";
  for (std::size_t index = 0; index < outputs[0].shape.size(); ++index) {
    if (index != 0)
      std::cout << ", ";
    std::cout << outputs[0].shape[index];
  }
  std::cout << "]\n";
  std::cout << "top1: " << std::distance(scores, best) << '\n';
  model.close();
}
```

建立 `CMakeLists.txt`：

```cmake title="CMakeLists.txt"
cmake_minimum_required(VERSION 3.16)
project(pcie_host LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaPCIeHost REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(pcie_host pcie_host.cpp)
target_include_directories(pcie_host PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
  pcie_host
  PRIVATE SimaPCIeHost::sima_neat_pcie_host ${OpenCV_LIBS}
)
```

</CodeTab>
</CodeTabs>

## 執行應用程式

<CodeTabs>
<CodeTab label="Python" lang="python">

在主機上啟用 Python 環境，然後執行腳本：

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 pcie_host.py
</ShellCommand>

</CodeTab>
<CodeTab label="C++" lang="cpp">

建立並執行 C++ 應用程式：

<ShellCommand prompt="pcie-host">
cmake -S . -B build
cmake --build build
./build/pcie_host
</ShellCommand>

</CodeTab>
</CodeTabs>

執行成功時，會列印傳回的輸出張量及其分數最高的 ImageNet 類別索引。使用提供的 ResNet-50 成品和拉布拉多犬圖片，預期的分數最高的類別索引為 208，即拉布拉多犬：

```text
output: resnetv17_dense0_fwd [1, 1000]
top1: 208
```

Python 的上下文管理器會自動呼叫 `close()`。C++ 範例則會明確地呼叫它。這兩種版本在結束前都會釋放佇列 0。

## 下一步

繼續進行 [PCIe 協同處理](/develop-apps/development-workflow/pcie-model/)，以了解完整的模型 API、使用 `push()` 和 `pull()` 進行的管線請求，以及影像預處理選項。
