---
title: "데이터 형식"
description: "형식 태그, 페이로드 계열, 레이아웃, 텐서 의미론"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/data_formats
---

# 데이터 형식 및 텐서 의미

이 페이지에서는 `InputOptions::format`, `OutputTensorOptions::format`, 텐서 이미지 메타데이터 및 샘플 페이로드 태그에서 사용되는 공개 형식 어휘를 설명합니다.

태스크 수준에서 사용하려면 [텐서 및 샘플](/develop-apps/development-workflow/core_types)부터 시작합니다. 그래프 경계에 명시적인 형식 계약이 필요한 경우 여기에 접속합니다.

## 형식 태그

`FormatTag` / `FormatSpec`은 페이로드 형식을 지정합니다. Python에서는 형식 필드에 대해 `pyneat.Format` 또는 `pyneat.FormatTag` 값을 사용합니다. Python 형식 필드에 원시 문자열을 할당하지 마십시오.

Python은 일반 사용자가 사용하는 형식 태그를 노출합니다. `BBOX`, `MLA`, `ARGMAX` 및 `DETESSDEQUANT`와 같은 일부 하위 수준 C++ 태그는 일반적으로 텐서 의미 메타데이터, 페이로드 태그 또는 진단 정보를 통해 나타나며, 할당 가능한 `pyneat.Format` 값으로 직접 나타나지는 않습니다.

일반 태그:

| 태그 | 일반 페이로드 | 의미 |
|---|---|---|
| `RGB` | 이미지 | 8비트/채널로 압축된 RGB. |
| `BGR` | 이미지 | 8비트/채널로 압축된 BGR. OpenCV는 기본적으로 이 형식을 사용합니다. |
| `GRAY8` | 이미지 | 8비트 그레이스케일. |
| `NV12` | 이미지/비디오 | Y 평면과 인터리브된 UV 평면. 너비와 높이는 짝수여야 합니다. |
| `I420` | 이미지/비디오 | Y, U 및 V 평면. 너비와 높이는 짝수여야 합니다. |
| `H264` | 인코딩됨 | H.264 액세스 단위 / NAL 스트림. `AVC`는 별칭입니다. |
| `H265` | 인코딩됨 | H.265 / HEVC 액세스 단위 / NAL 스트림. `HEVC`는 별칭입니다. |
| `ENCODED` | 인코딩됨 | 일반 인코딩된 페이로드. 캡스 문자열은 전용 형식 태그 없이 코덱을 식별합니다. |
| `FP32` | 텐서 | Float32 텐서 페이로드. |
| `INT8` | 텐서 | 부호 있는 INT8 텐서 페이로드. |
| `UINT8` | 텐서 | 부호 없는 UINT8 텐서 페이로드. |
| `BF16` | 텐서 | BF16 텐서 페이로드. |
| `BBOX` | 감지 | 압축된 경계 상자 페이로드. |
| `ByteStream` | 텐서 의미 | 다운스트림 계약에 의해 해석되는 불투명 바이트 스트림. |

## 페이로드 패밀리

`PayloadType`은 그래프 경계를 넘는 광범위한 패밀리를 선택합니다.

| 페이로드 패밀리 | 내부/미디어 의미 | 일반 메타데이터 |
|---|---|---|
| `Image` | 디코딩된 픽셀 | 픽셀 형식, 너비, 높이, 레이아웃, 이미지 의미 메타데이터 |
| `Tensor` | 모델 또는 앱 텐서 | dtype, shape, 레이아웃, 텐서 의미 메타데이터 |
| `Encoded` | H.264, H.265 또는 JPEG와 같은 인코딩된 미디어 | 캡스 문자열, 코덱 형식, 타임스탬프 |
| `Auto` | 가능한 경우 추론 | 텐서/샘플 메타데이터만으로 충분할 때 사용 |

텍스트, 오디오, 바이트 스트림 및 불투명 바이트 페이로드는 텐서 의미 또는 특수 사양을 사용합니다. 이 릴리스에서 검토된 공개 API의 별도 `PayloadType` 열거형 값은 아닙니다.

## 원시 이미지 매핑

| 형식 | 페이로드 유형 | 텐서 레이아웃/모양 | 참고 사항 |
|---|---|---|---|
| `RGB` | `Image` | `HWC`, `[H, W, 3]` | 촘촘하게 압축된 픽셀. |
| `BGR` | `Image` | `HWC`, `[H, W, 3]` | `cv2.imread` 또는 OpenCV BGR 프레임에 사용. |
| `GRAY8` | `Image` | `HW`, `[H, W]` | 단일 채널 그레이스케일. |
| `NV12` | `Image` | `HW`, `[H, W]` + 평면 메타데이터 | 복합 Y + UV 평면. |
| `I420` | `Image` | `HW`, `[H, W]` + 평면 메타데이터 | 복합 Y + U + V 평면. |

압축된 형식의 경우, 깊이는 채널 수입니다. 텐서 페이로드의 경우, 깊이는 선택한 레이아웃과 모양에서 파생됩니다.

## 형식, 레이아웃 및 축 의미를 함께 읽기

단일 필드만 분리하여 읽지 마십시오.

| 필드 | 알려주는 내용 |
|---|---|
| `PixelFormat` / 이미지 형식 메타데이터 | RGB, BGR, GRAY8, NV12 또는 I420와 같이 픽셀 채널을 해석하는 방법. |
| `TensorLayout` | HWC, CHW 또는 HW와 같이 텐서 차원이 정렬되는 방식. |
| `TensorAxisSemantic` | 텐서가 더 풍부한 의미 메타데이터를 포함할 때 축이 의미하는 바. |
| `TensorDType` | UInt8, INT8, FP32 또는 BF16와 같이 각 요소가 저장되는 방식. |
| `ByteFormat` / 바이트 스트림 메타데이터 | 다음 단계에서 불투명 바이트를 어떻게 해석해야 하는지. |

바이트 자체는 의미가 없습니다. 버퍼를 재해석하기 전에 메타데이터 필드를 함께 사용하십시오.

## 입력 옵션 형식 예제

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::InputOptions input;
input.payload_type = simaai::neat::PayloadType::Image;
input.format = simaai::neat::FormatTag::BGR;
input.width = 640;
input.height = 480;
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
input_options = pyneat.InputOptions()
input_options.payload_type = pyneat.PayloadType.Image
input_options.format = pyneat.Format.BGR
input_options.width = 640
input_options.height = 480
```

</CodeTab>
</CodeTabs>

경계에 필요한 필드만 설정합니다. 텐서 또는 샘플에 이미 충분한 메타데이터가 포함되어 있는 경우, 중복된 추정을 피하십시오.

H.264 및 H.265는 각각 전용 `H264` 및 `H265` 태그를 가지고 있습니다. 입력 경계에 일치하는 태그를 설정합니다. 미디어 유형은 이 태그에서 파악됩니다. 전용 태그가 없는 코덱에 대해서만 명시적인 캡스 문자열과 함께 `ENCODED`를 사용하십시오.

## 고급 이미지/비디오 출력 어댑터

일반적인 모델 출력의 경우, `nodes.output(...)`를 사용하고 `pull_tensors(...)`를 사용하여 텐서를 가져옵니다. 이미지 또는 비디오 출력을 CPU 친화적인 `UInt8` 텐서로 변환, 크기 조정 또는 비율 조정해야 하는 경우에만 `OutputTensorOptions`를 사용합니다. 그런 다음 앱에서 텐서를 가져옵니다.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::OutputTensorOptions output;
output.format = simaai::neat::FormatTag::BGR;
output.target_width = 640;
output.target_height = 480;

graph.add_output_tensor(output);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
output = pyneat.OutputTensorOptions()
output.format = pyneat.Format.BGR
output.target_width = 640
output.target_height = 480

graph.add_output_tensor(output)
```

</CodeTab>
</CodeTabs>

`add_output_tensor(...)`는 기본 출력 dtype인 `TensorDType::UInt8`을 허용합니다. 모델 텐서와 전체 `Sample` 엔벨로프를 원하는 출력의 경우 일반적인 `nodes.output(...)` 경로를 유지합니다. 다른 dtype이 필요한 경우 명시적인 그래프 또는 앱 측 변환을 추가합니다.

## 샘플 페이로드 태그

`Sample::payload_tag`는 다운스트림 소비자를 위한 권장 레이블입니다. 이는 더 이상 사용되지 않는 `Sample::format` 필드를 대체합니다.

인코딩된 미디어 또는 그래프 경계 협상 시 디버깅할 때 `payload_tag`, `payload_type`, `media_type` 및 `caps_string`을 함께 사용합니다.

## 전처리 메타데이터 및 ROI 경로

감지 디코딩, 렌더링 및 ROI 워크플로는 모델 공간 좌표를 소스 프레임 좌표로 다시 매핑하기 위해 전처리 메타데이터가 필요합니다.

해당 메타데이터에는 다음이 포함될 수 있습니다.

- 대상 너비 및 높이
- 크기 조정된 콘텐츠 너비 및 높이
- 크기 조정 또는 레터박스 모드
- 패딩 값 및 기하학
- 입력 및 출력 색상 형식
- 축 순열
- 정규화, 양자화 및 테셀레이션 플래그
- ROI 창, 소스 이미지 크기, ROI 배치 크기 및 ROI별 어파인 변환

박스 또는 마스크가 잘못된 위치에 나타나는 경우 임계값을 변경하기 전에 전처리 메타데이터가 디코딩 또는 렌더링 단계에 도달했는지 확인합니다. ROI 목록 전처리에 대한 자세한 내용은 [사전 처리된 관심 영역 목록](/reference/preproc_roi)를 참조하십시오.

## 추가 정보

- [텐서 및 샘플](/develop-apps/development-workflow/core_types)
- [dtype 계약](/develop-apps/advanced-concepts/dtype_contract)
- [노드](/develop-apps/development-workflow/node)
