---
title: "프레임별 속성"
description: "선택한 다중 부분 HTTP 헤더를 캡처하고, 해당 헤더가 포함된 디코딩된 프레임에서 다시 읽어옵니다."
sidebar_position: 4
slug: /develop-apps/advanced-concepts/frame_attributes
---

# 프레임별 속성

일부 카메라는 전송을 통해 각 프레임을 설명합니다. 여기에는 시퀀스 번호, 캡처 타임스탬프, 채널 이름 등이 포함됩니다. Neat는 이러한 값 중 선택된 세트를 디코딩하여 해당 프레임에 속하는 `Sample`에 전달할 수 있습니다.

`Sample::attributes`는 일반적인 문자열-문자열 매핑입니다. Neat는 이를 복사하여 해당 프레임과 연결하고, 대상 버퍼가 재사용될 때 이를 지웁니다. 값은 파싱, 병합 또는 재해석하지 않습니다. 키와 값은 유효한 UTF-8이어야 하며, GStreamer 문자열 표현 방식에서는 보존할 수 없으므로 임베디드된 NUL 바이트를 포함할 수 없습니다.

## 보장되는 사항

> 선택된 멀티파트 부분 헤더는 기본 `HttpMjpegDecodedInput` 경로, 큐/분기, 코어 샘플-GStreamer 구체화 경계를 통해 디코딩된 프레임과 연결된 상태로 유지됩니다.

이것이 이번 릴리스에 대한 전체적인 약속입니다. 이 범위를 벗어나는 사항은 [지원되지 않는 경로](#unsupported-paths)에 나열되며, Neat는 속성을 조용히 삭제하는 대신 그래프 구성을 실패시킵니다.

## 캡처 활성화

캡처는 기본적으로 비활성화되어 있습니다. 활성화하려는 헤더 이름을 지정하면 캡처가 활성화됩니다.

```cpp
#include "nodes/groups/HttpMjpegDecodedInput.h"

simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions opt;
opt.url = "http://camera.local/stream";
opt.header_capture.headers = {"Image-Index", "Image-Time"};

auto source = simaai::neat::nodes::groups::HttpMjpegDecodedInput(opt);
```

다시 읽어보기:

```cpp
simaai::neat::Sample sample;
if (run.pull(1000, sample) == simaai::neat::PullStatus::Ok) {
  const auto it = sample.attributes.find("image-index");
  if (it != sample.attributes.end()) {
    // it->second is the value this frame was sent with.
  }
}
```

Python에서 동일한 표면을 사용할 때, `attributes`는 실시간 매핑을 제공하며, 항목 할당은 기본 `Sample`에 도달하고, 딕셔너리를 할당하면 내용이 대체됩니다.

```python
import pyneat

opt = pyneat.HttpMjpegDecodedInputOptions()
opt.url = "http://camera.local/stream"
opt.header_capture.headers = ["Image-Index", "Image-Time"]
source = pyneat.groups.http_mjpeg_decoded_input(opt)

# ... later, on a pulled sample:
index = sample.attributes.get("image-index")

sample.attributes["image-index"] = "42"     # reaches the Sample
sample.attributes = {"image-time": "..."}   # replaces the whole map
```

## 헤더 규칙

구성된 목록은 **허용 목록**입니다. 빈 목록은 캡처를 완전히 비활성화하고 기존 토폴로지와 동작을 변경하지 않습니다.

| 규칙 | 동작 |
|---|---|
| 대/소문자 | 구성된 이름과 출력되는 키는 ASCII 소문자로 정규화됩니다. 대/소문자를 구분하지 않고 일치시킵니다. 소문자 키를 사용하여 속성을 다시 읽습니다. |
| 허용 목록의 중복 항목 | 정규화 후 병합됩니다. |
| 단일 파트 내에서 반복되는 헤더 | 마지막 값이 우선합니다. |
| 파트에 없는 헤더 | 키가 생략됩니다. 이전 프레임에서 상속되지 않습니다. |
| 헤더가 있지만 비어 있음 | 빈 문자열로 유지됩니다. |
| 공백 | 주변의 SP/HTAB만 제거됩니다. 값은 다른 방식으로 재해석되지 않습니다. |
| MIME 유형 | 존재하는 `Content-Type`은 `image/jpeg`여야 합니다(매개변수 허용). 없는 경우 JPEG 페이로드 검사를 통해 파트 유형을 결정합니다. |
| JPEG 페이로드 | 파트에는 SOI에서 EOI까지 정확히 하나의 완전한 JPEG가 포함되어야 합니다. 잘리거나 비어 있거나 연결된 이미지는 스트림에서 오류가 발생합니다. |
| 잘못된 입력 | 잘못된 헤더 이름, 접힌 헤더 줄 및 CR/LF/NUL 삽입은 거부됩니다. 안전하게 보이는 형태로 정규화하는 대신 스트림에 오류가 발생합니다. |

"없음"과 "비어 있음"을 빈 문자열을 테스트하는 대신 `count()` / `get()`을 사용하여 구별합니다.

### 제한

다음 중 하나라도 초과하면 잘리는 대신 구문 분석이 실패합니다.

| 제한 | 값 |
|---|---|
| `kMultipartHeaderCaptureMaxHeaders` | 64개의 선택된 헤더 이름 |
| `kMultipartHeaderCaptureMaxNameBytes` | 이름당 128바이트 |
| `kMultipartHeaderCaptureMaxLineBytes` | 헤더 줄당 8KiB |
| `kMultipartHeaderCaptureMaxBlockBytes` | 파트 헤더 블록당 64KiB |
| 멀티파트 JPEG 본문 | MIME 파트당 64MiB |

잘못 구성된 허용 목록은 `std::invalid_argument`를 사용하여 생성 시 거부됩니다.

## 지원되지 않는 경로

캡처가 활성화된 동안 `HttpMjpegDecodedInput`은 `use_videoconvert`, `use_videoscale`, `use_videorate` 또는 `extra_fragment`을 포함하는 그래프를 생성하지 않습니다. 이러한 요소들을 통해 보존되는 것이 입증되지 않았으며, 명확한 구성 오류가 조용히 사라지는 메타데이터보다 낫습니다.

또한 여러 입력에서 새 논리적 샘플을 생성하는 노드에 대해서는 속성이 정의되지 않습니다. 이러한 노드는 속성을 병합하지 않습니다.

## 연결 상태를 유지하는 방법

캡처가 활성화된 그래프는 파트 경계와 파트 헤더를 단일 상태 머신에서 구문 분석하는 전용 인프로세스 요소를 사용하므로 파트의 헤더가 해당 바이트를 전달하는 버퍼에 직접 연결됩니다. 따라서 데이터가 손실될 가능성이 없습니다. 해당 요소는 완전하고 구문 분석된 JPEG 프레임을 출력하므로 `jpegparse`는 캡처가 활성화된 경로에 삽입되지 않습니다. 선택한 속성을 연결하는 데 실패하면 프레임이 전달되지 않고 스트림에서 오류가 보고됩니다.

디코딩을 통해 플러그인은 수신된 각 인코딩된 이미지의 속성을 캡처하고, 디코더가 해당 이미지와 연결하여 디코딩된 출력에 해당 속성을 복원합니다. 이는 다음에 도착하는 임의의 출력에 적용되는 것이 아닙니다. 수신된 모든 이미지는 정확히 하나의 최종 결과에 도달하므로, 순서 변경, 삭제 또는 출력 풀 재사용을 통해 값을 다른 프레임으로 이동시킬 수 없습니다. 이 메커니즘은 코덱 및 전송 방식에 독립적이므로, 추후 다른 인코딩된 소스에도 적용할 수 있으며, 이를 위해 재설계할 필요가 없습니다.

## 호환성

`Sample` 및 소스 옵션 구조체에 필드가 추가되었습니다. 해당 필드 이름 또는 집계 초기화를 사용하여 이 구조체를 사용하는 소스 코드는 계속 컴파일됩니다.

해당 공개 구조체의 바이너리 레이아웃이 변경되었으므로, **이미 빌드된 구성 요소는 다시 빌드해야 합니다**. Neat ABI/SOVERSION은 **4**로 유지됩니다. 0.4.0은 아직 출시되지 않았으므로, 모든 ABI-4 구성 요소를 함께 다시 빌드하고 출시하여 ABI를 업데이트하는 대신 유지합니다.

## 추후 다른 소스 추가

디코더 경로는 일반적입니다. 새로운 인코딩된 소스는 디코더에 전달하는 버퍼에 중첩된 속성 구조를 연결하기만 하면 됩니다. 디코더 또는 샘플 경계 내에 전송 방식에 특정한 요소는 없습니다. 각 새로운 소스가 여전히 소유하는 것은 자체 추출 규칙과 자체적으로 보장하는 그래프 형태입니다.
