---
title: "앱 실행"
description: "Python 또는 C++를 사용하여 그래프 애플리케이션 내에서 YOLOv8을 실행하고 디코딩된 경계 상자 정보를 읽어옵니다."
sidebar_position: 3
mdx:
  format: mdx
---

# 앱 실행

![그래프를 만드는 방법](@site/../docs/images/hello-neat-graph-add-animation.svg)

![원본 이미지에서 Neat을 사용하여 YOLOv8으로 감지한 결과](@site/../docs/images/first_inference_hook.png)

*아래 프로그램에서 생성된 감지 결과가 원본 이미지에 표시됩니다.*

이것은 작은 **애플리케이션**으로서 동일한 YOLOv8 추론입니다. `Model.run(...)`을 직접 호출하는 대신([모델 실행](/develop-apps/hello-neat/run_first_model)와 같이), 모델을 입력, 모델, 출력을 포함하는 명명된 그래프 흐름인 [`Graph`](/develop-apps/development-workflow/graph)로 구성한 다음, 이를 구축하고 푸시/풀합니다. Python과 C++에서 동일한 프로그램을 사용할 수 있습니다. 각 코드 블록에서 언어 탭을 선택하세요.

이 첫 번째 앱의 구조는 의도적으로 단순합니다.

- 명명된 _입력_(`nodes.input("image")`)은 데이터가 앱에 들어오는 위치를 나타냅니다.
- _모델_(`graph.add(model)`)은 모델을 그래프의 한 단계로 실행합니다.
- 명명된 _출력_(`nodes.output("detections")`)은 애플리케이션이 결과를 읽는 위치를 나타냅니다.

동일한 API는 나중에 훨씬 더 복잡한 애플리케이션으로 확장될 수 있습니다. 여기서 목표는 핵심 구성 패턴입니다.

:::tip 원하는 언어를 선택하세요.
모든 코드 블록에서 **Python / C++** 탭을 사용하세요. 선택은 사이트 전체의 언어 선택 도구를 따르므로, 모든 코드 조각과 전체 프로그램이 함께 변경됩니다.
:::

## 프로젝트 설정

:::tip 이미 [모델 실행](/develop-apps/hello-neat/run_first_model)을 실행했나요?
이 섹션은 건너뛰셔도 됩니다. 여기서는 동일한 `assets/` 디렉터리, 모델 패키지 및 샘플 이미지를 사용합니다. 바로 [코드를 한 줄씩 살펴보세요.](#walk-through-the-code)로 이동하세요.
:::

1. **모델과 입력 이미지에 대한 자산 디렉터리를 만듭니다.**
    ```bash
    mkdir -p assets
    ```
2. **모델을 다운로드하세요:**
    ```bash
    sima-cli modelzoo -v 2.0.0 get yolo_v8s
    ```
    :::note sima-cli 모델 다운로드
    만약 `sima-cli` 모델을 지정된 위치가 아닌 다른 곳에 기록합니다. `assets` 디렉터리, 해당 파일을 복사하여 `assets/yolo_v8s_mpk.tar.gz`.
    :::
3. **샘플 이미지를 다운로드**하여 문서에 있는 대로 저장합니다. `assets/tutorial_sample_image.png`.

    [샘플 이미지를 열거나 다운로드하세요.](../../images/tutorial_sample_image.png).

## 코드 살펴보기

이 프로그램은 짧은 코드 조각 8개로 구성되어 있습니다. 각 코드 블록에서 언어 탭을 전환하세요.

### 1. 이미지 읽기

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
import cv2

bgr = cv2.imread("assets/tutorial_sample_image.png")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
#include <opencv2/opencv.hpp>

cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png");
cv::Mat rgb;
cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
```

</CodeTab>
</CodeTabs>

OpenCV는 BGR 형식으로 이미지를 읽고, YOLOv8은 RGB 형식을 예상합니다. 이 단계는 Neat하지 않습니다. 애플리케이션은 파일, 카메라 또는 디코더에서 픽셀 데이터를 가져옵니다. Neat은 다음 단계에서 시작됩니다.

### 2. 모델 옵션 설명

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
import pyneat as neat

opt = neat.ModelOptions()
opt.preprocess.kind   = neat.InputKind.Image
opt.preprocess.preset = neat.NormalizePreset.COCO_YOLO
opt.decode_type       = neat.BoxDecodeType.YoloV8
opt.score_threshold   = 0.25
opt.nms_iou_threshold = 0.45
opt.top_k             = 100
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>
namespace neat = simaai::neat;

neat::Model::Options opt;
opt.preprocess.kind   = neat::InputKind::Image;
opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
opt.decode_type       = neat::BoxDecodeType::YoloV8;
opt.score_threshold   = 0.25f;
opt.nms_iou_threshold = 0.45f;
opt.top_k             = 100;
```

</CodeTab>
</CodeTabs>

`ModelOptions`는 하나의 객체 내에서 모델 경로를 선언합니다. 즉, Neat이 입력 픽셀을 어떻게 전처리하고 감지기의 출력을 어떻게 디코딩하는지를 정의합니다.

| 필드 | 설정 내용 |
|---|---|
| `preprocess.kind = Image` | 입력은 미리 정의된 형태의 텐서가 아닌 원시 픽셀입니다. |
| `preprocess.preset = COCO_YOLO` | 크기 조정 + 레터박스 적용하여 모델 입력으로 변환, RGB, `1/255`로 스케일 조정, 평균값 빼기 없음. |
| `decode_type = YoloV8` | 감지 헤드 디코더 패밀리. |
| `score_threshold` / `nms_iou_threshold` / `top_k` | 신뢰도 하한값, NMS 오버랩, 유지할 최대 박스 수. |

### 3. 모델 로드

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
model = neat.Model("assets/yolo_v8s_mpk.tar.gz", opt)
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Model model("assets/yolo_v8s_mpk.tar.gz", opt);
```

</CodeTab>
</CodeTabs>

`Model`은 `.tar.gz` 파일을 읽고, 사용자가 전달한 `ModelOptions`를 기준으로 **MPK 계약**의 유효성을 검사한 다음, 모델 조각을 인스턴스화합니다. 아직 아무것도 실행되지 않았습니다.

### 4. 이미지를 `Tensor`로 래핑합니다.

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
tensor = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB)
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Tensor input = neat::Tensor::from_cv_mat(
    rgb,
    neat::ImageSpec::PixelFormat::RGB);
```

</CodeTab>
</CodeTabs>

`Tensor`는 Neat의 형식화된 데이터 컨테이너입니다. 여기에는 셰이프, 데이터 유형, 레이아웃, 그리고 프레임워크가 바이트를 해석하는 데 필요한 픽셀 형식이 포함됩니다. Neat이 레이아웃을 파악할 수 있도록 `PixelFormat`을 전달하는 것이 필수입니다. 단순히 바이트 정보만으로는 충분하지 않습니다.

### 5. 그래프 구성

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
graph = neat.Graph("hello_neat_app")
graph.add(neat.nodes.input("image"))
graph.add(model)
graph.add(neat.nodes.output("detections"))
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Graph graph("hello_neat_app");
graph.add(neat::nodes::Input("image"));
graph.add(model);
graph.add(neat::nodes::Output("detections"));
```

</CodeTab>
</CodeTabs>

`Graph`는 애플리케이션 흐름입니다. 각 `add(...)`는 다음 단계를 추가하므로, 이를 통해 선형 흐름 `image → model → detections`가 구축됩니다. 3단계의 모델 조각은 그 안에 포함된 하나의 단계가 됩니다.

### 6. 그래프를 구축하고 실행합니다.

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
run = graph.build()
try:
    run.push("image", [tensor])
    run.close_input()
    outputs = run.pull_tensors("detections", timeout_ms=2000)
finally:
    run.close()
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Run run = graph.build();
run.push("image", neat::TensorList{input});
run.close_input();
neat::TensorList outputs = run.pull_tensors("detections", /*timeout_ms=*/2000);
run.close();
```

</CodeTab>
</CodeTabs>

`build()`는 공개 그래프를 단일 실행 가능한 런타임 그래프로 축소하여 노드 이름을 보존합니다. 그런 다음 명명된 입력에 입력을 `push`하고, 더 이상 입력이 들어오지 않을 때 `close_input()`를 호출하고, 타임아웃과 함께 명명된 출력에서 결과를 `pull`합니다. `pull_tensors`는 `TensorList`를 반환합니다. 이는 `Model.run`이 생성했을 것과 동일한 형태이며, 여기에는 패킹된 YOLOv8 `BBOX` 출력이 포함됩니다.

### 7. 박스 디코딩

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
decoded = neat.decode_bbox(outputs)
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::TensorList decoded = neat::decode_bbox(outputs);
```

</CodeTab>
</CodeTabs>

`decode_bbox`는 1:1 비율의 위치 변환을 수행하는 `TensorList → TensorList` 변환입니다. 디코딩된 각 출력은 `[num_detections, 6]` 형태의 `float32` 텐서이며, 열은 `(x1, y1, x2, y2, score, class_id)`로 구성됩니다.

### 8. 바운딩 박스 읽기

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
labels = {0: "person", 27: "tie"}
for x1, y1, x2, y2, score, cls in decoded[0].to_numpy():
    name = labels.get(int(cls), f"id{int(cls)}")
    print(f"{name:<8} {score:.2f}  [{x1:4.0f} {y1:4.0f} {x2:4.0f} {y2:4.0f}]")
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
const neat::Tensor& boxes = decoded.front();      // [num_detections, 6] float32
auto m = boxes.map_read();
const float* d = static_cast<const float*>(m.data);
for (int64_t i = 0; i < boxes.shape[0]; ++i) {
  const float* r = d + i * 6;                     // x1 y1 x2 y2 score class_id
  const int cls = static_cast<int>(r[5]);
  const char* name = (cls == 0) ? "person" : (cls == 27) ? "tie" : "?";
  std::printf("%-8s %.2f  [%4.0f %4.0f %4.0f %4.0f]\n", name, r[4], r[0], r[1], r[2], r[3]);
}
```

</CodeTab>
</CodeTabs>

Python에서는 디코딩된 텐서가 다음과 같이 표시됩니다. `[N, 6]` NumPy 배열을 통해 `to_numpy()`C++에서는 텐서를 매핑하고 부동 소수점 값을 읽습니다. 모델은 COCO 클래스 ID를 출력하며, 이를 표시 이름에 매핑하는 작업은 애플리케이션에서 수행합니다.

## 전체 프로그램

프로젝트 디렉터리에 파일을 생성한 다음 빌드하고 실행합니다.

<CodeTabs>
<CodeTab label="Python" lang="python">

`app.py`:

```python {18-24,34-35,38-41,44-46,48}
#!/usr/bin/env python3
import sys

try:
    import pyneat as neat
except ImportError:
    sys.exit(
        "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
        "Run: source ~/pyneat/bin/activate"
    )

import cv2

LABELS = {0: "person", 27: "tie"}


def yolo_model_options():
    opt = neat.ModelOptions()
    opt.preprocess.kind   = neat.InputKind.Image
    opt.preprocess.preset = neat.NormalizePreset.COCO_YOLO
    opt.decode_type       = neat.BoxDecodeType.YoloV8
    opt.score_threshold   = 0.25
    opt.nms_iou_threshold = 0.45
    opt.top_k             = 100
    return opt


def main() -> int:
    bgr = cv2.imread("assets/tutorial_sample_image.png")
    if bgr is None:
        raise RuntimeError("failed to read assets/tutorial_sample_image.png")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    model = neat.Model("assets/yolo_v8s_mpk.tar.gz", yolo_model_options())
    tensor = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB)

    # Compose the model into a Graph application: image -> model -> detections.
    graph = neat.Graph("hello_neat_app")
    graph.add(neat.nodes.input("image"))
    graph.add(model)
    graph.add(neat.nodes.output("detections"))

    # Build the app, push the image into the named input, pull the named output.
    run = graph.build()
    try:
        run.push("image", [tensor])
        run.close_input()
        outputs = run.pull_tensors("detections", timeout_ms=2000)
    finally:
        run.close()

    decoded = neat.decode_bbox(outputs)
    for x1, y1, x2, y2, score, cls in decoded[0].to_numpy():
        name = LABELS.get(int(cls), f"id{int(cls)}")
        print(f"{name:<8} {score:.2f}  [{x1:4.0f} {y1:4.0f} {x2:4.0f} {y2:4.0f}]")
    print("[OK] Graph app completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**실행:**

* **DevKit에서**
  ```bash
  source ~/pyneat/bin/activate
  python3 app.py
  ```
* **Neat SDK 호스트에서**
  ```bash
  dk app.py
  ```

</CodeTab>
<CodeTab label="C++" lang="cpp">

`CMakeLists.txt`와 `main.cpp`를 생성합니다.

```cmake title="CMakeLists.txt" {18,23-27}
cmake_minimum_required(VERSION 3.16)
project(sima_neat_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Supports both DevKit/native installs (system paths) and
# cross builds with SYSROOT exported (SDK sysroot paths).
if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH
    "$ENV{SYSROOT}/usr"
    "$ENV{SYSROOT}/usr/lib"
    "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
endif()

find_package(SimaNeat REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)

add_executable(sima_neat_app main.cpp)
target_link_libraries(sima_neat_app
  PRIVATE
    SimaNeat::sima_neat
    PkgConfig::OPENCV
)
```

강조 표시된 두 줄은 앱을 Neat에 연결하는 역할을 합니다. `find_package(SimaNeat REQUIRED CONFIG)`는 설치된 Neat 패키지(`SimaNeatConfig.cmake`를 통해)를 찾고, `target_link_libraries(sima_neat_app PRIVATE SimaNeat::sima_neat ...)`는 이를 연결합니다. 가져온 `SimaNeat::sima_neat` 대상은 Neat의 포함 디렉터리와 전이적 종속성을 자동으로 전파하므로 수동으로 포함/라이브러리 경로를 지정할 필요가 없습니다. (`PkgConfig::OPENCV`는 이 앱이 이미지를 로드하기 위해 OpenCV를 사용하기 때문에 필요합니다.)

```cpp title="main.cpp" {13-19,30-31,34-37,40-42,44-46}
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace neat = simaai::neat;

neat::Model::Options yolo_model_options() {
  neat::Model::Options opt;
  opt.preprocess.kind   = neat::InputKind::Image;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  opt.decode_type       = neat::BoxDecodeType::YoloV8;
  opt.score_threshold   = 0.25f;
  opt.nms_iou_threshold = 0.45f;
  opt.top_k             = 100;
  return opt;
}

int main() {
  cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png");
  if (bgr.empty())
    throw std::runtime_error("failed to read assets/tutorial_sample_image.png");
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  neat::Model model("assets/yolo_v8s_mpk.tar.gz", yolo_model_options());
  neat::Tensor input = neat::Tensor::from_cv_mat(
      rgb,
      neat::ImageSpec::PixelFormat::RGB);

  // Compose the model into a Graph application: image -> model -> detections.
  neat::Graph graph("hello_neat_app");
  graph.add(neat::nodes::Input("image"));
  graph.add(model);
  graph.add(neat::nodes::Output("detections"));

  // Build the app, push the image into the named input, pull the named output.
  neat::Run run = graph.build();
  run.push("image", neat::TensorList{input});
  run.close_input();
  neat::TensorList outputs = run.pull_tensors("detections", /*timeout_ms=*/2000);
  run.close();

  neat::TensorList decoded = neat::decode_bbox(outputs);
  const neat::Tensor& boxes = decoded.front();      // [num_detections, 6] float32
  auto m = boxes.map_read();
  const float* d = static_cast<const float*>(m.data);
  for (int64_t i = 0; i < boxes.shape[0]; ++i) {
    const float* r = d + i * 6;                     // x1 y1 x2 y2 score class_id
    const int cls = static_cast<int>(r[5]);
    const char* name = (cls == 0) ? "person" : (cls == 27) ? "tie" : "?";
    std::printf("%-8s %.2f  [%4.0f %4.0f %4.0f %4.0f]\n", name, r[4], r[0], r[1], r[2], r[3]);
  }
  std::printf("[OK] Graph app completed\n");
  return 0;
}
```

**구축:**

<ShellCommand prompt="sdk|devkit">
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
</ShellCommand>

**실행:**

* **DevKit에서**
  ```bash
  ./build/sima_neat_app
  ```
* **Neat SDK 호스트에서**
  ```bash
  dk build/sima_neat_app
  ```

</CodeTab>
</CodeTabs>

각 감지 결과에 대해 한 줄씩 표시됩니다. 그런 다음:

```text
[OK] Graph app completed
```

## Neat이 구축한 것

![안녕하세요. Neat 그래프 앱의 사용 흐름입니다.](@site/../docs/images/hello-neat-graph-app-flow.svg)

API는 해당 구조에 직접적으로 매핑됩니다.

- `Graph`는 애플리케이션 흐름을 포함하고, `graph.add(...)`는 각 단계를 순서대로 추가합니다.
- 명명된 입력 및 출력은 런타임 엔드포인트가 됩니다: `run.push("image", ...)` 및 `run.pull_tensors("detections")`.
- `Model`은 `Model.run`을 사용하여 직접 호출할 수 있는 동일한 조각이며, 여기서는 애플리케이션 내의 하나의 노드로 실행됩니다.

## 다음 단계

더욱 심층적인 그래프 구성을 위해 [그래프 프로그래밍 모델](/develop-apps/development-workflow/graph)을 계속 진행합니다.

그 후, 더 광범위한 SiMa.ai Neat 학습 자료를 통해 학습을 이어갑니다.

- 모델, 그래프 및 실행과 같은 주요 Neat 개념을 설명하는 [핵심 프로그래밍 모델](/develop-apps/development-workflow/overview)을 학습합니다.
- 특정 개념과 워크플로우를 단계별로 안내하는 [튜토리얼](/tutorials/)를 따릅니다.
- [깃허브의 앱 저장소](https://github.com/sima-neat/apps)에 있는 소스 코드와 함께 [앱 포털](https://apps.neat.sima.ai/portal)에서 선별된 애플리케이션을 살펴봅니다.
