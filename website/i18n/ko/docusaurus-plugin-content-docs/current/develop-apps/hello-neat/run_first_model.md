---
title: "모델 실행"
description: "샘플 이미지에서 Neat 모델을 실행합니다."
sidebar_position: 2
---

# 모델 실행

[안녕하세요 Neat!](/develop-apps/hello-neat/minimal)에서 사용한 것과 동일한 작업 디렉터리를 사용하여 객체 감지를 위한 실제 모델을 실행합니다.
이 애플리케이션은 YOLOv8 모델을 로드하고, 샘플 이미지를 읽고, 추론을 실행하고, 바운딩 박스를 디코딩하고, 감지된 객체의 수를 출력합니다.

이 페이지에서는 두 가지 Neat 개념을 소개합니다.

* [`Model`](/develop-apps/development-workflow/model)은 컴파일된 모델 패키지를 로드하고 `run(...)` 진입점을 제공합니다.
* [`ModelOptions`](/tutorials/configure-model-options)는 Neat에게 이미지를 준비하고 감지기 출력을 디코딩하는 방법을 알려줍니다.

지금은 전체 API를 완전히 숙지할 필요가 없습니다. `Model`과 `ModelOptions`가 함께 작동하여 컴파일된 모델을 실행하는 방법에 집중하십시오.

![안녕하세요, Neat YOLOv8 플로우입니다.](@site/../docs/images/hello-neat-yolov8-flow.svg)

## 모델 및 샘플 이미지 가져오기

1. **모델과 입력 이미지를 저장할 assets 디렉터리를 만듭니다.**
    ```bash
    mkdir -p assets
    cd assets
    ```
2. **모델을 다운로드하세요:**
    ```bash
    sima-cli modelzoo -v 2.0.0 get yolo_v8s
    ```
    :::note sima-cli 모델 다운로드
    `sima-cli`가 모델을 `assets` 디렉터리 이외의 다른 곳에 저장하는 경우, 해당 파일을 `assets/yolo_v8s_mpk.tar.gz`로 복사합니다.
    :::
3. **샘플 이미지를 다운로드**하여 `assets` 디렉터리에 `tutorial_sample_image.png`로 저장합니다.

    [샘플 이미지를 열거나 다운로드하세요.](../../images/tutorial_sample_image.png).
4. **프로젝트 디렉터리로 돌아갑니다:**
    ```bash
    cd ..
    ```

## 단계별 설명

[안녕하세요 Neat!](/develop-apps/hello-neat/minimal)의 프로그램을 기반으로 구축합니다. 동일한 `CMakeLists.txt` 파일을 유지합니다(이미 Neat 및 OpenCV와 연결되어 있음). 그런 다음 프로그램 본문을 아래의 네 단계로 대체합니다. 각 단계는 최종 프로그램의 작은 부분입니다. 순서대로 읽은 다음 [전체 프로그램](#full-program)을 가져와 붙여넣고 실행합니다. 모든 블록에서 언어 탭을 선택합니다. 선택은 사이트 전체 선택기를 따릅니다.

### 1. 이미지 {#step-load-image} 로드 및 크기 조정

YOLOv8s는 고정된 `640×640` BGR 이미지를 필요로 하므로 OpenCV를 사용하여 샘플 이미지를 읽고 크기를 조정합니다. 이는 일반적인 이미지 입출력이며, 아직 Neat API는 사용하지 않습니다.

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

### 2. 입력 및 디코딩 과정을 설명합니다. `ModelOptions` {#step-model-options}

`ModelOptions` 이미지와 모델 간의 런타임 계약입니다. 여기서는 다음 두 가지를 명시합니다. Neat 추론 전에 디코딩된 픽셀을 전처리해야 하며, 검출기의 원시 출력을 박스로 디코딩하는 방법을 설명합니다. `decode_type` YOLOv8 디코더를 선택하고, 임계값을 사용하여 약하거나 겹치는 박스를 제거합니다. `top_k` 개수를 제한합니다.

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

### 3. 모델을 로드하고 추론을 실행합니다. {#step-run}

컴파일된 `.tar.gz` 패키지와 옵션을 사용하여 `Model`을 생성한 다음, `run(...)`을 호출하고 타임아웃을 설정합니다. 이 함수는 동기적으로 실행되며 출력 텐서를 반환합니다. `timeout_ms`는 실행이 중단될 경우 오류를 발생시켜 프로그램이 멈추는 대신 명확하게 실패하도록 합니다.

**C++**는 모델 입력당 하나의 `cv::Mat`을 전달합니다. **Python**은 먼저 NumPy 이미지를 `BGR` 태그가 지정된 `Tensor`로 래핑하여 Neat이 바이트 레이아웃을 알 수 있도록 한 다음, Python 모델 입력이 시퀀스이므로 `[tensor]`를 전달합니다.

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

### 4. 감지 개수 읽기 {#step-read}

`decode_type`이 설정되었으므로, 첫 번째 출력 텐서에는 디코딩된 박스 정보가 포함됩니다. BBOX 텐서는 `uint32` 형식의 감지 개수로 시작하므로, 처음 4바이트를 읽습니다. 전체 와이어 형식(박스별 좌표, 점수 및 클래스)은 [모델 출력에서 객체 감지 영역을 읽습니다.](/tutorials/read-detection-boxes)에서 다룹니다.

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

## 전체 프로그램 {#full-program}

[안녕하세요 Neat!](/develop-apps/hello-neat/minimal)의 `CMakeLists.txt` 파일을 그대로 사용합니다(이미 앱을 Neat 및 OpenCV와 연결합니다). 그리고 프로그램 본문을 아래의 전체 파일로 대체합니다. 강조 표시된 세 줄이 핵심입니다. `Model`을 생성하고, 입력을 구성하고, `run()`을 호출합니다.

<details>
<summary>전체 프로그램을 보여주세요.</summary>

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

## 빌드 및 실행

다음 명령을 프로젝트 디렉터리(즉, `assets/`가 포함된 디렉터리)에서 실행합니다.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

Hello Neat!과 동일한 명령어를 사용하여 다시 빌드합니다.

<ShellCommand prompt="sdk|devkit">
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
</ShellCommand>

그런 다음 실행 파일을 실행합니다.

<ShellCommand prompt="devkit">
./build/sima_neat_hello
</ShellCommand>

<ShellCommand prompt="sdk">
dk build/sima_neat_hello
</ShellCommand>

</CodeTab>
<CodeTab label="Python" lang="python">

스크립트를 실행합니다.

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 hello_neat.py
</ShellCommand>

<ShellCommand prompt="sdk">
dk hello_neat.py
</ShellCommand>

</CodeTab>
</CodeTabs>

다음과 유사한 탐지 결과 요약을 확인할 수 있습니다.

```text
detections=3
[OK] YOLOv8 completed
```

정확한 숫자는 모델 패키지 및 런타임 버전에 따라 달라질 수 있습니다. 중요한 점은 앱이 빌드되고 실행되어 `[OK] YOLOv8 completed` 상태에 도달한다는 것입니다.

## 구축한 내용

이 예제는 더 큰 Neat 애플리케이션에서 사용하는 것과 동일한 일반적인 단계를 따릅니다.

- 컴파일된 모델 패키지(`.tar.gz`)를 `Model`로 로드합니다.
- 입력 이미지를 모델이 예상하는 형식으로 변환합니다.
- Neat 런타임 단계를 통해 추론을 실행합니다.
- 원시 감지기 출력을 바운딩 박스로 디코딩합니다.

바운딩 박스 디코딩, 임계값, NMS 및 감지기 출력 구조에 대한 자세한 설명은 [모델 출력에서 객체 감지 영역을 읽습니다.](/tutorials/read-detection-boxes)를 참조하십시오.

## 다음 단계

YOLOv8이 실행되면 더 광범위한 SiMa.ai Neat 학습 자료를 계속 진행하십시오.

- `Model.run(...)`을 직접 호출하는 대신, 동일한 모델을 `Graph` 애플리케이션(한 번 빌드하고 푸시/풀 방식으로 구동하는 명명된 입력 → 모델 → 출력 파이프라인)으로 구성하기 위해 **[앱 실행](/develop-apps/hello-neat/run_an_app)**을 계속 진행합니다.
- 모델, 그래프 및 런타임 실행과 같은 주요 Neat 개념을 설명하는 [핵심 프로그래밍 모델](/develop-apps/development-workflow/overview)을 학습합니다.
- 특정 개념과 워크플로를 단계별로 안내하는 [튜토리얼](/tutorials/)를 따릅니다.
- [깃허브의 앱 저장소](https://github.com/sima-neat/apps)에 있는 소스 코드와 함께 [앱 포털](https://apps.neat.sima.ai/portal)에서 선별된 애플리케이션을 탐색합니다.
