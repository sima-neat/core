---
title: "PCIe 공동 처리"
description: "첫 번째 PCIe 공동 처리 추론을 실행하세요."
sidebar_position: 4
---

# PCIe 공동 처리

![PCIe 공동 처리 흐름](@site/../docs/images/hello-neat-pcie-coprocessing-flow.svg)

PCIe 공동 처리 방식을 사용하면 애플리케이션 제어 및 입력 준비 작업을 호스트 시스템에서 수행하고, 연결된 Modalix PCIe 카드는 모델을 실행합니다. 이 예제에서는 호스트가 이미지를 읽고, 크기를 조정하고, 정규화한 다음, 준비된 텐서를 카드에 전송하여 ResNet-50 추론을 수행하고, 반환된 분류 결과를 출력합니다.

## 시작하기 전에

다음이 필요합니다:

- 호스트 머신에 설치된 [Neat PCIe 호스트 패키지](/getting-started/neat-library/pcie-host/);
- 카드에 설치된 호환 가능한 Neat Library 버전; 그리고
- 호스트에서 접근할 수 있는 카드 관리 인터페이스. 카드 0은 기본적으로 `10.0.0.2`를 사용합니다.

## 모델과 이미지를 가져오세요

호스트 시스템에 작업 디렉터리를 생성하고 Model Zoo에서 ResNet-50을 다운로드합니다.

<ShellCommand prompt="pcie-host">
mkdir -p pcie-host-quickstart/assets
cd pcie-host-quickstart
sima-cli modelzoo get resnet_50
</ShellCommand>

다운로드한 압축 파일을 이 디렉터리에 `resnet_50.tar.gz`로 저장하십시오.

[래브라도어 견종 이미지 예시](../../images/hello-neat-pcie-labrador.jpg)를 다운로드하고 `assets/sample.png`로 저장하세요.

이 이미지는 엘프가 촬영하고, 디미르코와 FT2가 수정한 [노란 래브라도어, 새것처럼 보이는 모습.jpg](https://commons.wikimedia.org/wiki/File:YellowLabradorLooking_new.jpg)를 기반으로 하며, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) 라이선스에 따라 배포됩니다.

현재 작업 디렉터리에는 다음 파일이 포함되어 있어야 합니다.

```text
pcie-host-quickstart/
├── assets/
│   └── sample.png
└── resnet_50.tar.gz
```

## 애플리케이션을 생성합니다.

Python 또는 C++를 선택하고 해당 애플리케이션을 작업 디렉터리에 생성합니다.

<CodeTabs>
<CodeTab label="Python" lang="python">

`pcie_host.py`를 생성합니다.

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

`pcie_host.cpp`를 생성합니다.

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

`CMakeLists.txt` 파일을 생성합니다.

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

## 애플리케이션을 실행합니다.

<CodeTabs>
<CodeTab label="Python" lang="python">

호스트 시스템에서 파이썬 환경을 활성화하고 스크립트를 실행합니다.

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 pcie_host.py
</ShellCommand>

</CodeTab>
<CodeTab label="C++" lang="cpp">

C++ 애플리케이션을 빌드하고 실행합니다.

<ShellCommand prompt="pcie-host">
cmake -S . -B build
cmake --build build
./build/pcie_host
</ShellCommand>

</CodeTab>
</CodeTabs>

성공적으로 실행되면 반환된 출력 텐서와 가장 높은 점수를 받은 ImageNet 클래스 인덱스가 출력됩니다. 문서화된 ResNet-50 아티팩트와 래브라도 이미지를 사용하면 예상되는 최상위 클래스는 인덱스 208, 즉 래브라도 리트리버입니다.

```text
output: resnetv17_dense0_fwd [1, 1000]
top1: 208
```

Python 컨텍스트 관리자는 `close()`를 자동으로 호출합니다. C++ 예제는 이를 명시적으로 호출합니다. 두 버전 모두 종료하기 전에 큐 0을 해제합니다.

## 다음 단계

[PCIe 공동 처리](/develop-apps/development-workflow/pcie-model/)를 계속 진행하여 전체 모델 API, `push()` 및 `pull()`을 사용한 파이프라인 요청, 그리고 이미지 전처리 옵션에 대해 알아보세요.
