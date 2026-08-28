---
title: "사전 처리된 ROI 목록"
description: "하나 이상의 이미지에서 여러 개의 런타임 ROI(관심 영역) 창에 대해 전처리 작업을 실행합니다."
sidebar_position: 7
---

# 사전 처리된 ROI 목록

애플리케이션에 이미 하나 이상의 이미지 창이 있고 각 창을 모델 입력에 맞게 정확히 전처리하려는 경우(크기 조정 또는 레터박스 처리, 색상 변환, 정규화, 양자화, 그리고 선택적으로 테셀레이션) 전처리된 ROI 목록을 사용하세요.

이 방법은 2차 분류기, 검출기 캐스케이드, 추적기 또는 소스 이미지의 일괄 처리에서 동적 작물 목록이 생성되는 모든 흐름에 적합합니다.

## 최소 예제

C++:

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

파이썬:

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

## 크기 조정, 레터박스, 정규화 설정을 구성합니다.

ROI 목록 전처리 기능은 전체 프레임 전처리 기능과 동일한 모델 전처리 옵션을 사용합니다.

C++:

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

파이썬:

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

지원되는 `scaling_type` 토큰은 `BILINEAR`, `NEAREST_NEIGHBOUR`, `BICUBIC`, `INTERAREA` 및 `NO_SCALING`입니다. `NEAREST_NEIGHBOR` 및 `INTER_AREA`는 허용되는 별칭입니다.

## 배치 의미론

| 입력 | 의미 |
| --- | --- |
| `images` | 원본 이미지 배치입니다. ROI가 제공될 때 벡터/리스트는 비어 있지 않아야 합니다. Python에서는 uint8 형식의 HW/HWC NumPy/Torch/`pyneat.Tensor` 이미지 입력을 허용합니다. |
| `rois` | 런타임 ROI 목록입니다. 출력 텐서의 순서는 이 벡터의 순서와 일치합니다. |
| `PreprocessRoi::batch_index` |은 `images`에서 ROI가 읽어올 소스 이미지를 선택합니다. |
| `PreprocessRoi::x`, `y` | 부호 있는 소스 프레임의 왼쪽 상단 좌표입니다. 음수 값을 사용할 수 있습니다. |
| `PreprocessRoi::width`, `height` | ROI 크기입니다. 둘 다 양수여야 합니다. |

ROI 목록 호출에 사용되는 모든 소스 이미지에는 일치하는 너비, 높이, OpenCV 유형 및 채널 수가 있어야 합니다. 스테이지 API는 ROI 목록에 대해 8비트 RGB/BGR(`CV_8UC3`) 및 GRAY/GRAY8(`CV_8UC1`) 형식의 소스 이미지를 지원합니다.

Python의 경우, `cv2.imread` 이미지에는 `image_format=pyneat.PixelFormat.BGR`를, RGB 배열에는 `RGB`를, HW 흑백 배열에는 `GRAY8`를 전달합니다. CHW 텐서는 `pyneat.stages.preproc`에서 거부되므로, 스테이지를 호출하기 전에 HWC로 변환해야 합니다.

## 출력 의미

`stages::Preproc(images, model, rois)`는 다음 값을 반환합니다.

- 유효한 비어 있지 않은 요청에 대한 `out.size() == rois.size()` 값입니다.
- ROI `rois[i]`에서 생성된 출력 텐서 `out[i]`;
- 모델 경로에서 선택한 데이터 유형과 레이아웃(예: BF16 또는 INT8, 덴스 또는 테셀레이션 출력)을 의미합니다.
- 각 출력에 대한 `tensor.semantic.preprocess` 메타데이터.

각 출력 결과의 전처리 메타데이터에는 선택된 ROI, 레터박스 기하학적 정보, 정규화/양자화/테셀레이션 플래그, 그리고 모델/전처리된 좌표를 원래 소스 프레임 좌표계로 매핑하는 어파인 변환이 포함됩니다.

## 프레임 바깥 영역

관심 영역(ROI)은 원본 이미지의 범위를 벗어날 수 있습니다.

```cpp
std::vector<PreprocessRoi> rois = {
    {0, -16, -16, 128, 128},
    {0, image.cols - 64, image.rows - 64, 128, 128},
};
```

프레임 내 영역은 복사되고, 프레임 외 영역은 Preproc 패드 값을 사용하여 채워집니다. 이렇게 하면 출력 형상이 안정적으로 유지되고 이미지 가장자리 근처에서 특별한 처리가 필요하지 않습니다.

## 레터박스와 화면 비율

`ResizeMode::Letterbox`의 경우, 전처리 과정에서 ROI별로 레터박스 크기 조정 및 패딩을 계산합니다. 따라서 높이가 큰 ROI와 너비가 큰 ROI는 동일한 대상 크기를 공유하더라도 서로 다른 `scaled_width`, `scaled_height`, `pad_left` 및 `pad_top` 메타데이터를 생성할 수 있습니다.

후속 코드는 가정을 기반으로 패딩을 다시 계산하는 대신 메타데이터를 읽어야 합니다.

## 검증 체크리스트

추론 결과에 대해 비판하기 전에 다음 사항을 확인하십시오.

- 이미지 벡터는 비어 있지 않습니다.
- 모든 이미지의 크기, OpenCV 유형, 채널 수가 동일합니다.
- Python 입력은 CHW 텐서가 아닌 uint8 형식의 HW/HWC 이미지입니다.
- 각 ROI에는 유효한 `batch_index`가 있습니다.
- 각 ROI는 양의 `width` 및 `height` 값을 가집니다.
- ROI 목록 모드의 경우 소스 이미지 형식은 RGB, BGR, GRAY 또는 GRAY8입니다.
- 크기 조정 모드와 정규화 방식은 모델 학습 시 사용된 전처리 방식과 일치합니다.
- 하위 수준 좌표 소비자는 `tensor.semantic.preprocess` 메타데이터를 사용합니다.

## 이 동작을 검증하는 테스트

이 저장소에는 사용자 인터페이스 경로와 기능적 ROI 동작에 대한 빠른 테스트 커버리지가 포함되어 있습니다.

- `preproc_roi_batch_functional_test`
- `preproc_roi_user_smoke_test`

이 기능은 여러 이미지, 여러 관심 영역(ROI), 프레임 외부 패딩, 크기 조정/레터박스 동작, 정규화된 출력, 그리고 고밀도/테셀레이션된 BF16/INT8 처리 방식을 지원합니다.

## 참조:

- [전처리 노드](/reference/nodes/preproc)
- [추론 전에 이미지 전처리](/tutorials/preprocess-images)
- [데이터 형식](/develop-apps/advanced-concepts/data_formats)
- [BoxDecode 디코딩 유형](/reference/boxdecode_decode_types)
