---
title: "BoxDecode 디코딩 유형"
description: "객체 감지 후처리 작업에 적합한 BoxDecodeType을 선택하세요."
sidebar_position: 6
---

# BoxDecode 디코딩 유형

`nodes::SimaBoxDecode`는 원시 감지 헤드 텐서를 감지 결과로 변환합니다. 이 노드는 모델 추론 후에 실행되며, 선택한 모델 제품군에 대한 디코딩 연산을 적용하고, 낮은 신뢰도 박스를 필터링하고, NMS(Non-Maximum Suppression)를 실행한 다음, 디코딩된 박스로 시작하는 텐서 페이로드를 출력합니다. 감지 모델은 해당 페이로드를 박스로 파싱할 수 있으며, 자세 및 분할 모델은 박스 다음에 나오는 키포인트 또는 마스크도 파싱할 수 있습니다.

일반적인 모델 패키지 사용의 경우, `Model`을 인식하는 생성자를 사용하는 것이 좋습니다. 모델 아카이브는 디코더에 필요한 텐서 순서, 레이아웃, 양자화, 클래스 수, 크기 조정 메타데이터 및 점수 범위 힌트를 제공합니다. 일반적으로 애플리케이션은 디코딩 제품군과 필터링 임계값만 선택합니다.

## 빠른 시작

```cpp
using namespace simaai::neat;

Model model("/path/to/yolov8_model.tar.gz");

auto boxdecode = nodes::SimaBoxDecode(
    model,
    BoxDecodeType::YoloV8,
    /* detection_threshold */ 0.25,
    /* nms_iou_threshold */ 0.45,
    /* top_k */ 100);
```

독립적인 스테이지 사용 시:

```cpp
simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
opt.detection_threshold = 0.25;
opt.nms_iou_threshold = 0.45;
opt.top_k = 100;
```

## 인수

| 인자 | 의미 |
| --- | --- |
| `decode_type` | 모델 패밀리/헤드 형식(예: `BoxDecodeType::YoloV8` 또는 `BoxDecodeType::YoloX`). 필수 항목입니다. |
| `detection_threshold` | 감지 결과를 유지하기 위해 필요한 최소 점수입니다. `0.25`와 같이 모델에 적합한 값을 사용하십시오. |
| `nms_iou_threshold` | 비최대 억제(non-maximum suppression)에 사용되는 IoU 임계값입니다. |
| `top_k` | 유지할 최대 감지 개수입니다. `0`은 백엔드/모델의 기본값을 사용합니다. |
| `original_width`, `original_height` | 원본 이미지 크기(raw-geometry 생성자를 사용할 때 좌표 매핑에 사용). |
| `model_width`, `model_height` | 모델 입력 크기 재정의. 다음을 사용하여 `Model` 생성자는 패키징된 텐서 계약이 아닌 공간 디코딩 관련 설정을 변경합니다. |
| `resize_mode_override` | 업스트림 `Preproc` 단계에서 크기 조정 메타데이터를 기록하지 않고, 크기 조정/레터박스/자르기 동작을 명시적으로 지정해야 할 때만 사용하십시오. |
| `decode_type_option` | 고급 하위 레이아웃 선택기입니다. 내보낸 헤드 레이아웃을 알고 있지 않다면 모델 팩을 사용할 때 `Auto`로 설정하십시오. |

## 입력 및 출력

**입력:** 모델에서 생성된 원시 감지 텐서입니다. 예상되는 텐서의 형태는 모델 종류에 따라 다릅니다. MPK/모델 아카이브를 사용하면 Neat이 패키지된 계약에서 해당 세부 정보를 읽어옵니다.

**출력:** 디코딩된 감지를 포함하는 하나의 BoxDecode 텐서입니다. 감지 모델은 표준 `BBOX` 페이로드를 사용합니다. 자세 및 분할 모델은 동일한 초기 박스를 유지하고 작업별 페이로드를 추가합니다.

| 모델 작업 | C++ 헬퍼 | Python 헬퍼 | 디코딩된 텐서 |
| --- | --- | --- | --- |
| 감지 | `decode_bbox(...)` | `pyneat.decode_bbox(...)` | `[N, 6]` float32 박스: `x1, y1, x2, y2, score, class_id` |
| 자세 | `decode_pose(...)` | `pyneat.decode_pose(...)` | 박스 `[N, 6]` 및 키포인트 `[N, 17, 3]` float32: `x, y, visibility` |
| 분할 | `decode_segmentation(...)` | `pyneat.decode_segmentation(...)` | 박스 `[N, 6]` float32 및 마스크 `[N, 160, 160]` uint8 |
| SuperPoint | `decode_superpoint(...)` | `pyneat.decode_superpoint(...)` | 키포인트 `[N,2]`, 점수 `[N]`, 디스크립터 `[N,D]` |

감지-표시 그래프는 결과를 `SimaRender`에 전달할 수 있습니다. 단순히 박스만 필요한 애플리케이션 코드는 BoxDecode 출력에 대해 `decode_bbox(...)`를 계속 사용할 수 있습니다.

## 슈퍼포인트

SuperPoint는 BoxDecode 제품군의 일부로 남아 있지만, 박스인 척하는 대신 특징점을 출력합니다. 최소 A65 기본 구성은 다음과 같습니다.

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.descriptor_output_dtype = TensorDType::Float32;

auto decoder = nodes::SimaBoxDecode(model, options);
```

Python에서도 동일한 기본값을 사용합니다.

```python
options = pyneat.BoxDecodeOptions(pyneat.BoxDecodeType.SuperPoint)
options.superpoint.descriptor_output_dtype = pyneat.TensorDType.Float32

decoder = pyneat.nodes.sima_box_decode(model, options=options)
```

`A65V1`은 기본 프로필입니다. 모델이 다른 숫자 연산 방식을 필요로 할 때 다른 프로필을 명시적으로 선택하십시오. Neat은 텐서의 형태나 값으로부터 연산 방식을 추론하지 않습니다.

| 프로필 | 선택 시점 | 생산 상태 |
|---|---|---|
| `LightGlueV1` | LightGlue 호환 감지기, NMS, 좌표 및 디스크립터 동작 | 지원 |
| `MagicLeapDemoV1` | 고정된 Magic Leap 데모 동작 | 지원 |
| `A65V1` | 이전 A65 슈퍼포인트 디코더와의 호환성 | 지원; 기본값 |
| `PaperBicubicV1` | 향후 완전히 정의된 양방향 보간 정책을 위한 예약된 숫자 ID | 프로덕션에서 정의될 때까지 거부됨 |

수치 처리 방식과 출력 인코딩은 서로 독립적입니다. 예를 들어, 기본 V1 출력에서 A65 수치 처리 방식을 선택할 수 있습니다.

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::FeaturePointsV1;
```

기존 바이트 레이아웃은 선택적으로 사용할 수 있으며, 다음과 같은 추가적인 제약 조건이 있습니다.

```cpp
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::LegacyA65InterleavedV0;
options.superpoint.descriptor_output_dtype = TensorDType::Int8;
```

`SuperPointProfile::Auto`는 먼저 권한 있는 MPK `superpoint.profile` 메타데이터를 사용합니다. API(`Model::Options.superpoint.profile`) 또는 MPK에서 프로필을 제공하지 않으면 `A65V1`로 해결됩니다. Neat는 텐서 모양, 값, 파일 이름 또는 다운스트림 노드를 기반으로 프로필을 추측하지 않습니다.

공개 센티널 값이 변경되지 않은 상태로 유지되면 `detection_threshold=0.0`, `top_k=0`, `nms_radius=-1` 및 `border_margin=-1`은 선택한 프로필에서 값을 가져옵니다. `A65V1`은 임계값 `0.1`, Top-K `600`, NMS 반경 `4` 및 경계 마진 `0`으로 해결됩니다. LightGlueV1 및 MagicLeapDemoV1은 각각 임계값 `0.0005` 및 `0.015`를 사용하며, 둘 다 Top-K `600`, NMS 반경 `4` 및 경계 마진 `4`를 사용합니다.

`nms_iou_threshold`는 SuperPoint에 적용되지 않습니다. 대신 픽셀 반경 `superpoint.nms_radius`를 사용하십시오. 기본 출력은 버전이 지정된 `FEATURE_POINTS_V1` 구조화된 배열 페이로드입니다. `LegacyA65InterleavedV0`는 명시적인 마이그레이션 형식이며 256차원 INT8 디스크립터가 필요합니다. `decode_bbox` 또는 `BoxDecodeResults` 대신 `decode_superpoint`를 사용하십시오.

버전이 지정된 MPK `superpoint` 스키마 v1 레코드는 실패 시 안전하게 작동하도록 설계되었습니다. 프로필, 고유한 디텍터 및 디스크립터 텐서 ID, 64개의 16진수로 구성된 `sha256:` 핑거프린트 및 지원되는 입력 표현 `raw-logits-65` 및 `coarse-pre-l2`를 명시해야 합니다. 스키마 0은 마이그레이션/수동 레코드로서만 허용되며, 생략된 스키마 0 표현 필드는 해당 두 가지 원시 입력 표현으로 표준화되고 진단 정보에 기본값으로 기록됩니다. 알 수 없는 스키마 버전 또는 표현 토큰은 컴파일 계약을 위반합니다. API 프로필 재정의가 다른 MPK 프로필에 대해 스탬프된 핑거프린트와 충돌하는 경우, 선택한 프로필에 대해 MPK를 다시 스탬프하십시오. Neat는 해당 출처를 삭제하거나 재해석하지 않습니다.

## BBOX 와이어 페이로드

객체 감지 디코더는 입력 프레임당 하나의 텐서를 출력하며, 이 텐서에는 `BBOX` 태그가 지정됩니다. 해당 텐서는 1차원 `UInt8` 바이트 버퍼입니다.

| 필드 | 값 |
| --- | --- |
| `semantic.detection.format` | `"BBOX"` |
| `dtype` | `UInt8` |
| `shape` | `[N_bytes]`이며, 여기서 `N_bytes`는 모델 아카이브에서 가져온 압축된 버퍼의 용량입니다. |

텐서의 형태는 감지 횟수가 아닌 바이트 수를 나타냅니다. 페이로드는 리틀 엔디안 레이아웃을 사용합니다.

```text
offset  size  content
------  ----  -------
  0      4    uint32  N = valid detections in this frame
  4     24    RawBox[0]
 28     24    RawBox[1]
  .      .      ...
  .      .    RawBox[N-1]
                   trailing bytes are padding and must be ignored
```

각 `RawBox` 레코드는 24바이트입니다.

| 오프셋 | 크기 | 유형 | 필드 | 의미 |
| --- | --- | --- | --- | --- |
| 0 | 4 | `int32` | `x` | 원본 이미지의 왼쪽 상단 x 좌표입니다. |
| 4 | 4 | `int32` | `y` | 원본 이미지의 왼쪽 상단 y 좌표. |
| 8 | 4 | `int32` | `w` | 원본 이미지의 픽셀 단위 너비입니다. |
| 12 | 4 | `int32` | `h` | 원본 이미지의 픽셀 단위 높이입니다. |
| 16 | 4 | `float32` | `score` | `[0.0, 1.0]`에서의 NMS 후 신뢰도. |
| 20 | 4 | `int32` | `class_id` | 모델에서 정의한 클래스 ID입니다. |

하나의 레코드에 대한 일치하는 Python `struct` 형식은 `"<iiiifi"`입니다.

업스트림 전처리 메타데이터가 있는 경우 좌표는 원본 이미지 픽셀 단위로 표시됩니다. 좌표는 `[0, 1]`로 정규화되지 않으며 모델의 내부 레터박스 입력 공간으로 표현되지 않습니다.

## `model.run`이 원본 헤더를 반환할 때

일부 모델 경로는 디코딩된 `BBOX` 텐서 대신 `model.run(...)`에서 원시 특징 맵 헤드를 반환합니다. 이는 실패한 실행이 아닙니다. 모델이 실행되었지만 해당 경로에 출력을 읽는 시점에 BoxDecode가 포함되지 않았다는 의미입니다.

다음 규칙을 사용하십시오.

- `detections=...` 또는 `BBOX` 텐서: 압축된 BBOX 페이로드를 파싱하거나 사용합니다.
  디코딩 도우미 함수.
- `raw_output_heads=...`: BoxDecode 단계를 추가하거나, 모델 경로를 검사하거나.
  모델별 후처리를 통해 원시 텐서를 처리합니다.

원시 헤드를 박스로 파싱하지 마십시오. 원시 텐서 레이아웃은 내보낸 모델 패밀리와 모델 아카이브 계약에 따라 달라집니다.

## 계약 변경

모델 아카이브는 디코딩 유형, 임계값, `top_k` 및 소스 지오메트리에 대한 기본값을 제공할 수 있습니다. 런타임 인수는 비어 있지 않거나 양수 값을 전달할 때만 해당 기본값을 재정의합니다.

| 런타임 인수 | 전달된 값 | 동작 |
| --- | --- | --- |
| `decode_type` | 비어 있음 / `Unspecified` | 지원되는 경우 모델 아카이브를 유지하거나 경로 계획 추론을 사용합니다. |
| `decode_type` | 구체적인 유형 | 이번 실행에 대해 디코딩 방식을 재정의합니다. |
| `original_width` / `original_height` | `0` | 패키징된 지오메트리 또는 상위 프로세스 전처리 메타데이터를 보존합니다. |
| `original_width` / `original_height` | 양의 정수 | 좌표 매핑을 위해 원본 크기를 재정의합니다. |
| `detection_threshold` / `score_threshold` | `0.0` | 패키징된 임계값을 유지합니다. |
| `detection_threshold` / `score_threshold` | `> 0.0` | 점수 기준을 재정의합니다. |
| `nms_iou_threshold` | `0.0` | 패키지된 NMS IoU 값을 유지합니다. |
| `nms_iou_threshold` | `> 0.0` | NMS IoU 값을 재정의합니다. |
| `top_k` | `0` | 패키징된 상위 K개 항목을 유지합니다. |
| `top_k` | `> 0` | 유지할 최대 감지 개수를 재정의합니다. |
| `num_classes` | `0` | MPK에서 추론된 클래스 헤드 깊이를 사용합니다. |
| `num_classes` | MPK와 일치하는 양의 정수 | 명시적인 클래스 수를 사용합니다. MPK가 단일 클래스 헤드를 안정적으로 분할할 수 없는 경우 이는 필수입니다. |
| `num_classes` |는 양의 정수이며, 이는 YOLO26 MPK와 모순됩니다. | 파이프라인 구축 전에 오류가 발생하고 두 값을 모두 보고합니다. YOLO26은 클래스 깊이에서 그룹화된 원시 헤드 레이아웃을 파생시키므로, 이 불일치는 모델 계약 오류입니다. |
| `num_classes` | SSD 또는 YOLO26 이전 버전의 비자세 추정 YOLO 계열 모델에 사용할 양의 정수입니다. | 기존의 명시적 재정의 동작을 유지합니다. 자세 추정 디코더와 SuperPoint는 각자 고유한 규칙을 유지합니다. |

`detection_threshold`는 BoxDecode 노드/단계 생성자에서 사용하는 이름입니다. `ModelOptions.score_threshold`는 동일한 제어에 사용되는 모델 경로 옵션입니다.

## 디코딩 유형 매핑

| API 열거형 | 백엔드 토큰 | 일반적인 모델 제품군 |
| --- | --- | --- |
| `BoxDecodeType::Yolo` | `yolo` | 일반적인 YOLO 스타일 헤드 |
| `BoxDecodeType::YoloV5` | `yolov5` | YOLOv5 감지 |
| `BoxDecodeType::YoloV5Seg` | `yolov5-seg` | YOLOv5 분할 |
| `BoxDecodeType::YoloV7` | `yolov7` | YOLOv7 감지 |
| `BoxDecodeType::YoloV7Seg` | `yolov7-seg` | YOLOv7 분할 |
| `BoxDecodeType::YoloV8` | `yolov8` | YOLOv8 감지 |
| `BoxDecodeType::YoloV8Seg` | `yolov8-seg` | YOLOv8 분할 |
| `BoxDecodeType::YoloV8Pose` | `yolov8-pose` | YOLOv8 자세 추정 |
| `BoxDecodeType::YoloV9` | `yolov9` | YOLOv9 감지 |
| `BoxDecodeType::YoloV9Seg` | `yolov9-seg` | YOLOv9 분할 |
| `BoxDecodeType::YoloV10` | `yolov10` | YOLOv10 감지 |
| `BoxDecodeType::YoloV10Seg` | `yolov10-seg` | YOLOv10 분할 |
| `BoxDecodeType::YoloV26` | `yolo26` | YOLO26 감지 |
| `BoxDecodeType::YoloV26Pose` | `yolo26-pose` | YOLO26 자세 |
| `BoxDecodeType::YoloV26Seg` | `yolo26-seg` | YOLO26 분할 |
| `BoxDecodeType::YoloV6` | `yolov6` | YOLOv6 감지 |
| `BoxDecodeType::YoloX` | `yolox` | YOLOX 감지 |
| `BoxDecodeType::Ssd` | `ssd` | 주문된 헤드 지오메트리에서 선택한 정확한 준비된 SSD300, SSD-Mobile-300, SSD-Mobile-320 또는 SSDlite-Mobile-320 계약 |
| `BoxDecodeType::SuperPoint` | `superpoint` | SuperPoint 검출기 및 특징점 후처리 |
| `BoxDecodeType::Detr` | `detr` | DETR 스타일의 트랜스포머 기반 객체 검출 |
| `BoxDecodeType::EffDet` | `effdet` | 효율적인 객체 감지 |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` R-CNN 제안 단계 | |
| `BoxDecodeType::Centernet` | `centernet` | CenterNet 감지 |

`BoxDecodeType::Unspecified`는 설정되지 않은 값이며, 런타임 전에 오류가 발생합니다. SSD 레시피 식별자는 내부 Core 계약이며(`ssd300-v1`, `ssd-mobile-300-v1`, `ssd-mobile-320-v1` 또는 `ssdlite-mobile-320-v1`), 다른 공개 디코딩 유형이나 백엔드 토큰이 아닙니다. Core는 로우어링 전에 이를 해결하고, 설치된 객체 디코더는 계속해서 지원되는 `ssd` 계열 토큰을 수신하고 이미 검증된 헤드 지오메트리에서 해당 고정 구현을 선택합니다.

## 올바른 유형을 선택하세요.

- SiMa에서 제공하거나 SiMa에서 컴파일한 모델 팩을 사용하는 경우, 모델 제품군과 일치하는 `BoxDecodeType`을 선택하고 `decode_type_option`을 `Auto`로 설정하십시오.
- 탐지 결과가 누락되거나 모든 점수가 예상보다 낮게 나오는 경우, 먼저 디코딩 모델이 내보낸 모델 헤드와 일치하는지 확인하십시오. YOLOX, YOLOv6 및 YOLO26은 원시/로짓 스타일 헤드를 사용하므로 확률 기반 YOLO 헤드와 동일하게 처리해서는 안 됩니다.
- 경계 상자가 잘못 이동하거나 크기가 조정된 경우 이미지 크기 조정 정책을 확인하십시오. `resize_mode_override`는 그래프에 크기 조정 메타데이터를 기록하는 상위 `Preproc` 단계가 없을 때만 사용하십시오.
- 사용자 지정 모델 패키지를 제작하는 경우, 아카이브에 감지 헤드에 대한 정확한 정보(텐서 순서, 논리적 형태, 물리적 저장 방식, dtype/양자화, 점수 범위, 클래스 수, 슬라이싱된 출력 등)가 명확하게 설명되어 있는지 확인하십시오. 애플리케이션 코드는 이러한 세부 사항을 보정할 필요가 없습니다.

## 모양 및 레이아웃 지침

다양한 감지 모델은 서로 다른 헤드 레이아웃을 사용합니다. 일부 모델은 각 특징 맵 레벨에 대해 하나의 텐서를 사용하는 반면, 다른 모델은 박스, 객체성, 클래스, 키포인트 또는 마스크를 별도의 텐서로 분할합니다. 일부 모델의 출력은 밀집된 HWC 텐서이고, 다른 모델은 컴파일러/런타임에 의해 패킹되거나 분할됩니다.

모델 패키지 흐름의 경우, 이는 패키지된 계약에 의해 처리됩니다. 수동으로 연결된 텐서의 경우, 핵심 규칙은 다음과 같습니다. 내보낸 헤드 형식을 정확히 일치시켜야 합니다. 순위 또는 채널 수만 기준으로 디코딩 유형을 선택하지 마십시오.

고급 텐서 계약 규칙:

- YOLO 계열 디코딩 유형: `Yolo`, `YoloV5`, `YoloV7`, `YoloV8`, `YoloV9`.
  `YoloV10` 및 분할/자세 추정 변형 모델은 분리된 헤드 또는 해당 모델 제품군과 일치하는 패킹된 헤드를 사용합니다.
- 패키징된 YOLO 헤드는 클래스 수와 헤드 깊이를 일관되게 유지해야 합니다.
  기능 수준.
- `YoloV26`은 그룹화된 원시 l/t/r/b 바운딩 박스 헤드와 클래스-점수 헤드를 사용합니다.
- `Ssd`는 일반적인 SSD 디코더가 아닙니다. 미리 준비된 **네 가지 프로필**만 처리합니다.
  컴파일 시점에 완전하고 순서가 지정된 로컬/구성 H/W/C 서명을 기준으로 합니다. 다른 헤드 세트 또는 순서를 사용하면 오류가 발생하며, 오류 메시지에는 관찰된 서명과 지원되는 서명이 함께 출력됩니다.
  - **SSD300**`dboxes300_coco`): 300x300 입력, 특징 맵
    `{38,19,10,5,3,1}`, 셀당 사전 확률 `{4,6,6,6,4,4}`, 신뢰도 채널 순서
    `class*A + anchor`, 클래스 차원을 기준으로 **소프트맥스** 함수를 적용한 클래스 점수 (배경은 인덱스 0에 포함).
  - **SSD-Mobile-300-v1** (`ssd_anchor_generator`): 300x300 입력, 특징
    맵 `{19,10,5,3,2,1}`, 셀당 사전 확률 `{3,6,6,6,6,6}`, 신뢰 채널
    순서 `anchor*C + class`, 클래스별 점수는 클래스별 **시그모이드** 함수를 통해 계산되며(배경은 무시됨).
  - **SSD-Mobile-320-v1** (`ssd_anchor_generator`): 320x320 입력, 특징
    맵 `{20,10,5,3,2,1}`, 셀당 사전 확률 `{3,6,6,6,6,6}`, 신뢰 채널
    순서 `anchor*C + class`, 클래스별 점수는 클래스별 **시그모이드** 함수를 통해 계산되며(배경은 무시됨).
  - **SSDlite-Mobile-320-v1** (TorchVision `DefaultBoxGenerator`): 320x320
    입력, 특징 맵 `{20,10,5,3,2,1}`, 모든 레벨에서 셀당 6개의 사전 값,
    위치 순서 `anchor*4 + {dx,dy,dw,dh}`, 신뢰도 순서
    `anchor*C + class` 및 배경을 포함한 모든 91개 클래스에 대한 **소프트맥스**를 통한 클래스 점수.

  모든 레시피는 그룹화된 레벨별 위치 헤드(깊이 = `4 * priors-per-cell`)와 클래스-신뢰도 헤드(깊이 = `num_classes * priors-per-cell`)를 쌍으로 사용하며, FasterRcnnBoxCoder 분산 스케일링(`scale_xy 0.1`, `scale_wh 0.2`) 및 **스트레치**(비등방성) 전처리 리사이징을 사용합니다. 점수 활성화는 레시피에 의해 고정됩니다(온디바이스 디코더와 일치). 역할별 그룹화된 레이아웃은 자동으로 선택되므로 `decode_type_option`을 `Auto`로 둡니다. 그룹화되지 않은 레이아웃 토큰은 거부됩니다.

  **모델 프레임은 헤드 지오메트리뿐만 아니라 프로필의 일부입니다.** SSD300-v1 및 SSD-Mobile-300-v1은 300x300이 필요하고, 두 개의 320-v1 프로필은 320x320이 필요합니다. 사전 테이블과 스트레치 역투영은 해당 프레임에서만 유효하므로, 해결된 전처리 리사이즈 대상 또는 다른 크기의 모델 차원 재정의는 빌드 시간에 거부됩니다.

  원시/독립적인 `SimaBoxDecode` 구성은 리사이즈 모드를 생성하지 않습니다. 상위 `Preproc` 메타데이터 요구 사항을 유지하거나, 명시적인 원시 오버로드를 사용하여 외부에서 수행된 `ResizeMode::Stretch`를 주장합니다. Letterbox 및 Crop은 거부됩니다.

  **`num_classes` 계약.** 인코딩된 클래스 수는 항상 신뢰도 헤드 깊이(`conf_depth / priors-per-cell`, 인덱스 0에 있는 배경 포함)에서 파생됩니다. SSD300-v1은 준비된 81-to-8 경로와 같은 연속적인 접두사 선택을 허용합니다. 다른 세 가지 프로필은 정확한 인코딩된 수를 요구합니다. 유효하지 않은 선택은 빌드 시간에 거부됩니다. 프로필 기본값을 사용하려면 설정하지 둡니다.
- `Detr`은 최대 헤드 깊이를 기반으로 클래스 채널을 추론하며, 유효한 값을 필요로 합니다.
  차원 클래스.
- `EffDet`, `RcnnStage1` 및 `Centernet`은 해당 모델 계열의 계약을 사용합니다.
  YOLO 디코딩 방식을 거치지 않고 경로를 설정합니다.
- `*-seg` 디코딩 유형은 박스 경계선을 표시하는 출력과 함께 특정 작업에 필요한 마스크 데이터를 생성합니다.

사용자 지정 모델 패키지가 전체 주문 서명 중 어느 것과도 일치하지 않으면 매처를 약화시키는 대신 새로운 명시적으로 지원되는 프로필을 준비하십시오.

## 파이썬 참고 사항

Python에서 모델 옵션을 구성할 때, 문자열 대신 사용할 수 있다면 형식화된 열거형을 사용하는 것이 좋습니다.

```python
opt = pyneat.ModelOptions()
opt.decode_type = pyneat.BoxDecodeType.YoloV8
```

모델 작업에 맞는 헬퍼를 사용하여 출력을 분석합니다.

```python
outputs = model.run([image])

boxes = pyneat.decode_bbox(outputs)[0].to_numpy()

pose = pyneat.decode_pose(outputs)[0]
pose_boxes = pose.boxes.to_numpy()
keypoints = pose.keypoints.to_numpy()

seg = pyneat.decode_segmentation(outputs)[0]
seg_boxes = seg.boxes.to_numpy()
masks = seg.masks.to_numpy()
```
