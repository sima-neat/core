---
title: "dtype 계약"
description: "텐서 데이터 유형, 양자화, 테셀레이션, 그리고 공개 페이로드 계약이 어떻게 서로 연관되어 작동하는가?"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/dtype_contract
---

# dtype 계약

Neat 모델 경로는 두 가지 계약을 갖습니다.

- 앱이 `Tensor`, `Sample`, `InputOptions`, 모델 사양 및 그래프 엔드포인트를 통해 확인하는 **공개 계약**
- 컴파일된 모델 아카이브와 선택된 전처리/후처리 경로에서 Neat이 결정하는 **모델 경로 계약**

모든 공개 경계가 FP32라고 가정하지 마십시오. 일부 경계는 이미지, 인코딩된 미디어, 패킹된 감지 페이로드, INT8 텐서, BF16 텐서 또는 애플리케이션 정의 텐서 의미를 포함합니다. 먼저 사양을 확인하십시오. 사양이 계약입니다.

경로 내부에서 Neat은 컴파일된 모델 계약에서 요구하는 경우 양자화, 테셀레이션, 캐스트, 디테셀레이션, 디양자화 및 후처리 단계를 삽입합니다.

## 4가지 MLA 입력 사례

모델 아카이브는 첫 번째 MLA 단계에 대해 Neat에게 두 가지 중요한 정보를 알려줍니다.

- MLA 입력 dtype(일반적으로 **BF16** 또는 **INT8**)
- MLA 측면의 테셀레이션이 이미 컴파일된 커널의 일부인지 여부

이를 통해 4가지 전처리 그래프 패밀리가 생성됩니다.

| MLA dtype | MLA 테셀레이션 | 전처리 그래프 패밀리 | Neat이 MLA 전에 삽입하는 항목 |
|---|---|---|---|
| BF16 | 예 | `Preproc` | 크기 조정, 색상 변환, 정규화. MLA 단계는 내부적으로 테셀레이션합니다. |
| BF16 | 아니요 | `Tess` | 크기 조정, 색상 변환, 정규화, 테셀레이션. |
| INT8 | 예 | `Quant` | 크기 조정, 색상 변환, 정규화, 양자화. MLA 단계는 내부적으로 테셀레이션합니다. |
| INT8 | 아니요 | `QuantTess` | 크기 조정, 색상 변환, 정규화, 양자화, 테셀레이션. |

[`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan)을 검사하여 플래너가 무엇을 선택했는지 확인하십시오.

## 테셀레이션의 의미

테셀레이션은 텐서 바이트를 MLA 입력 스크래치 패드가 예상하는 타일 지오메트리로 배열합니다. 이는 레이아웃 변환입니다. 동일한 논리적 텐서이지만 다른 메모리 순서입니다.

일치하는 디테셀레이션은 MLA 출력 후 경로가 다음 단계 또는 앱에 자연스러운 텐서 레이아웃을 반환해야 할 때 발생합니다.

## 경계 업그레이드

Neat은 4가지 사례 dtype 결정 위에 더 높은 수준의 경로 단계를 추가할 수 있습니다.

- **일반 전처리**: `PreprocessOptions`를 사용하여 추론 전에 크기 조정, 색상, 레이아웃, 정규화, 양자화, 테셀레이션 또는 명시적 변환 의도를 적용합니다.
- **BoxDecode**: 감지 후처리 단계가 필요한 모델에 대한 감지 헤드를 디코딩합니다. 앱은 `BoxDecodeType`(예: `YoloV8`) 및 `score_threshold`, `nms_iou_threshold` 및 `top_k`와 같은 필터링 필드를 사용하여 패밀리를 선택합니다.

이러한 업그레이드는 실행되는 커널과 앱이 수신하는 출력 계약을 변경합니다. 예를 들어, 원시 모델 출력 텐서와 디코딩된 감지 텐서는 동일한 공개 계약이 아닙니다.

## 이것이 앱 코드에 미치는 영향

- 입력을 할당하거나 출력을 디코딩하기 전에 `model.input_specs()` 및 `model.output_specs()`를 검사합니다.
- `ModelOptions.preprocess`를 사용하여 어떤 종류의 입력을 제공하는지 지정합니다. 예를 들어 이미지 입력, 텐서 입력, 크기 조정, 색상, 레이아웃, 정규화, 양자화 또는 테셀레이션 의도를 지정할 수 있습니다.
- `model.resolved_preprocess_plan()` / `model.preprocess_plan()`를 사용하여 옵션과 모델 아카이브를 기반으로 Neat이 무엇을 계획했는지 확인합니다.
- 출력 dtype, shape 또는 레이아웃을 가정하지 마십시오. 출력 사양을 읽고 필요한 경우 반환된 텐서 메타데이터를 읽습니다.
- 출력 계약이 일치하는 패킹된 형식인 경우에만 박스, 포즈 또는 세그멘테이션을 디코딩합니다.
- INT8/BF16/테셀레이션 세부 정보를 명시적인 공개 사양 또는 텐서가 이를 노출하지 않는 한 런타임 동작으로 처리합니다.

어떤 느낌도 주지 마십시오. 계약을 읽고 바이트를 이동하십시오.

## 출력 디코딩을 신중하게 수행

출력 계약과 일치하는 디코딩 도우미를 사용합니다.

| 출력 계약 | C++ | Python |
|---|---|---|
| 원시 텐서 | 반환된 `Tensor` / `TensorList`를 직접 사용 | 반환된 텐서를 직접 사용하거나 `to_numpy(...)` / `to_torch(...)`를 사용 |
| 패킹된 박스 | `simaai::neat::decode_bbox(...)` | `pyneat.decode_bbox(...)` |
| 패킹된 포즈 | `simaai::neat::decode_pose(...)` | `pyneat.decode_pose(...)` |
| 패킹된 세그멘테이션 | `simaai::neat::decode_segmentation(...)` | `pyneat.decode_segmentation(...)` |

디코딩된 박스는 `x1`, `y1`, `x2`, `y2`, `score` 및 `class_id` 열이 있는 float32 `[N, 6]` 텐서를 사용합니다. 포즈 및 세그멘테이션 디코더는 박스와 더불어 키포인트 또는 마스크에 대한 작업별 텐서를 반환합니다.

## 좌표 메타데이터 보존

감지 좌표는 종종 모델 공간에서 소스 프레임 공간으로 매핑하기 위해 전처리 메타데이터가 필요합니다. 레터박스, 크기 조정, ROI 목록, 렌더링 또는 감지 디코딩을 사용할 때 그래프를 통해 메타데이터를 보존합니다.

관련 메타데이터에는 대상 크기, 크기 조정된 크기, 패딩, 색상 변환, 축 순열, 정규화, 양자화, 테셀레이션, ROI 창 및 ROI별 어파인 변환이 포함될 수 있습니다.

디코딩된 박스가 잘못된 위치에 나타나는 경우 NMS를 비난하기 전에 메타데이터 전파를 확인하십시오. [데이터 형식](/develop-apps/advanced-concepts/data_formats#preprocess-metadata-and-roi-breadcrumbs) 및 [사전 처리된 관심 영역 목록](/reference/preproc_roi)를 참조하십시오.

## 관련 유형

- [`PreprocessOptions`](/reference/cppapi/structs/simaai-neat-preprocessoptions) — 애플리케이션 전처리 의도.
- [`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan) — 플래너가 컴파일한 내용.
- [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h) — 선택된 전처리 패밀리.
- [`Tensor`](/reference/{lsa}/structs/simaai-neat-tensor) — 공개 텐서 페이로드 및 메타데이터.
- [`Sample`](/reference/{lsa}/structs/simaai-neat-sample) — 페이로드와 런타임 메타데이터.

## 추가 정보

- [텐서 및 샘플](/develop-apps/development-workflow/core_types)
- [데이터 형식](/develop-apps/advanced-concepts/data_formats)
