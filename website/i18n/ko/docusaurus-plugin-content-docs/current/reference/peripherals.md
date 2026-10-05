---
title: "주변 장치 카탈로그"
description: "보드 로컬 카탈로그 서비스가 보고하는 DevKit 주변 장치를 나열합니다"
sidebar_position: 8
---

# 주변 장치 카탈로그

로컬 DevKit에 현재 연결된 장치를 확인하려면 주변 장치 카탈로그를
사용하십시오. 이 API는 C++와 Python에서 사용할 수 있으며 두 언어에서 동일한 형식화된
스냅샷을 반환합니다.

카탈로그는 SiMa Sentinel(`simaai-sentinel.service`)에 속합니다.
Sentinel은 이 카탈로그를 `GET /v1/peripherals`로 제공하는 보드 로컬 데몬입니다. Sentinel은
하드웨어 사실만 보고합니다. 각 `list()` 호출은 Sentinel에 제한된 요청을
한 번 수행하고, 그다음 Core가 `CameraInput`이 지원하는 카메라 모드를 판단합니다. Core는
하드웨어를 스캔하거나, 두 번째 카탈로그를 캐시하거나, 다른 검색 경로로
폴백하지 않습니다.

## 주변 장치 나열

Python:

```python
import pyneat

catalog = pyneat.peripherals.list()
for peripheral in catalog:
    print(peripheral.id, peripheral.type)
```

C++:

```cpp
#include <neat.h>

auto catalog = simaai::neat::peripherals::list();
for (const auto& peripheral : catalog) {
  // Use peripheral.id and peripheral.type.
}
```

반환된 카탈로그는 반복할 수 있습니다. 또한 다음 항목을 포함합니다.

| 필드 | 의미 |
| --- | --- |
| `revision` | `devices` 또는 `errors`가 바뀔 때마다 변경됩니다. 동등 여부 비교에만 사용하십시오. |
| `observed_at` | 이 스냅샷의 바탕이 된 스캔이 시작된 시각입니다. Sentinel의 첫 스캔이 완료될 때까지는 설정되지 않습니다. 그때까지는 장치가 연결되어 있어도 카탈로그가 비어 있습니다. |
| `errors` | 최근 스캔에서 실패한 공급자이며, 각각 `provider`, `code`, `reason`을 포함합니다. 실패한 공급자가 마지막으로 성공한 스캔에서 찾은 장치는 `devices`에 남아 있습니다. |
| `devices` | 주변 장치 목록입니다. |

카탈로그에 장치가 하나도 없을 수 있습니다. 이 경우도 성공한 결과입니다.

## 카메라 세부 정보

`peripheral.type == "camera"`이면 `peripheral.camera`에 다음 항목이 포함됩니다.

| 필드 | 의미 |
| --- | --- |
| `camera_name` | `CameraInputOptions`가 허용하는 정확한 libcamera 이름이며 선택 사항입니다. 현재 입력 API가 선택할 수 없는 카메라에는 없습니다. |
| `model` | 공급자가 보고하는 경우의 장치 모델입니다. |
| `backend` | `mipi` 또는 `v4l2`와 같은 검색 백엔드입니다. |
| `modes` | 개별 크기 또는 명시적 크기 범위, 프레임 속도, 지원 플래그 및 거부 이유입니다. |

각 모드의 `framerate_num`/`framerate_den`은 Sentinel이 해당 모드에 대해 나열하는
프레임 간격 중 가장 빠른 속도이며, 나열된 간격이 없으면 `0/1`입니다.
DevKit의 ISP는 MIPI 모드에 대해 간격을 나열하지 않으며, `CameraInput`은
caps를 통해 속도를 설정합니다.

Core는 `type`을 알 수 없는 간격 항목이나, 최댓값이 최솟값보다 작은 단계형 또는
연속 범위를 건너뜁니다. 건너뛴 항목은
속도를 설정하지도 않고 기본 속도를 포괄하지도 않지만, 해당 모드는 여전히
간격을 나열하는 모드로 간주됩니다.

Core는 `CameraInput`의 기본 libcamera 프로필
(`profile=Default`, 카메라를 `camera_name`으로 선택)에 대해 각 모드를 분류합니다. 다음 조건이
모두 충족되면 모드가 지원되며, 조건은 이 순서대로 확인됩니다. `reason`은
처음으로 실패한 조건을 나타냅니다.

1. 카메라의 `backend`가 `mipi`입니다.
2. 형식이 `CameraInputOptions`의 기본 형식(`NV12`)입니다.
3. 모드가 프레임 간격을 나열하는 경우, 그중 하나가 기본 프레임 속도
   (`30/1`)를 포괄합니다. 즉 1/30초의 개별 간격이거나 이를 포함하는 단계형 또는 연속
   범위입니다. 간격을 나열하지 않는 모드는 속도를 이유로 거부되지 않으며
   속도는 `0/1`입니다. 나열된 간격을 모두 건너뛴 모드는 속도를 이유로
   거부되며, 속도 역시 `0/1`입니다.
4. 모드가 ISP 출력 크기입니다(`isp_output`이 true).

Core는 카메라를 검사하거나, 획득하거나, 구성하거나, 카메라에서 스트리밍하지 않습니다. 따라서
카탈로그의 지원 표시는 이후의 독점 획득이 성공한다는 것을
보장하지 않습니다.

이 규칙은 Metoak SIMOR raw V4L2 프로필
(`CameraProfile::MetoakSimor`, RAW8 1920×360, `profile` 및
선택적으로 `device`로 선택)을 분류하지 않습니다. 규칙에는 센서별 조건이 없으므로,
Sentinel이 `mipi` 카메라로 나열하는 SIMOR 센서(이름이 `simor_metoak`으로 시작)는
다른 MIPI 센서와 동일한 ISP 모드 분류를 받습니다. 이 카메라에서
`supported: true`는 모드가 기본 프로필의 백엔드, 형식, 프레임 속도,
ISP 출력 크기와 일치한다는 뜻일 뿐이며, libcamera가 해당 센서에 대해
검증되었다는 뜻은 아닙니다. SIMOR 카메라는 카탈로그의 `camera_name`이 아니라
[`CameraInput`](/reference/nodes/camera-input)에
설명된 대로
`profile=MetoakSimor`로 선택하십시오.

## 모든 주변 장치 형식의 세부 정보

모든 주변 장치는 `type`과 관계없이 Sentinel이 게시한 전체 장치 레코드를
`details_json`에 압축된 JSON으로 담고 있습니다. Python에서는
`details_json`을 새 `dict`로 디코딩하는 `details`도 제공합니다.

`details_json`은 `microphone`처럼 Core에 형식화된 접근자가 없는 주변 장치 형식을
읽는 방법입니다. 새 장치 형식은 Core를 업데이트하지 않아도
Sentinel이 보고하는 즉시 사용할 수 있습니다. 이 Core 릴리스가 알지 못하는 필드를 포함하여
모든 필드와 값이 보존됩니다. JSON은 다시 직렬화되므로 키 순서와 공백은
데몬 응답과 다를 수 있습니다. 각 형식의 필드는
Sentinel 문서에 설명되어 있습니다.

Python:

```python
for peripheral in pyneat.peripherals.list():
    if peripheral.type == "microphone":
        print(peripheral.id, peripheral.details["capture_target"])
```

C++:

```cpp
#include <nlohmann/json.hpp>

for (const auto& peripheral : simaai::neat::peripherals::list()) {
  if (peripheral.type == "microphone") {
    const auto details = nlohmann::json::parse(peripheral.details_json);
    // Read details["capture_target"] and other fields.
  }
}
```

`details_json`은 어떤 JSON 라이브러리로든 구문 분석할 수 있습니다. 이 예제에서는 nlohmann/json을 사용합니다.

Core는 자신이 읽는 필드만 검증합니다. 카탈로그 필드(`revision`,
`observed_at`, `errors`, `devices`)가 잘못되었거나 유효한 `id`와 `type`이 없는 장치가 있으면
`list()`가 구문 분석 오류로 실패합니다. 반면 Core가 카메라 필드를
읽을 수 없는 카메라 레코드(예: 더 새로운 Sentinel에서 온 레코드)는 실패를 일으키지 않습니다. 해당 장치는
`id`, `type`, `details_json`을 유지하고 `camera`는 설정되지 않은 채로 남으며, 나머지
모든 장치는 평소대로 반환됩니다.

## 실패 및 범위

서비스가 없거나, 카탈로그를 제공하기에는 너무 오래되었거나,
주변 장치 검색이 비활성화되거나 중지된 경우,
권한이 거부된 경우, 요청 시간이 초과된 경우, 또는 응답이 잘못되었거나 너무 큰 경우
`list()`는 안정적인 코드가 포함된 `NeatError`를 발생시킵니다. 메시지에는 다음 운영 조치와,
Sentinel이 제공하는 경우 Sentinel 자체의 오류 텍스트가 포함됩니다.
[오류 코드 카탈로그](./error-codes.md)를 참조하십시오.

Sentinel이 설치되어 있지 않거나 주변 장치 카탈로그를 제공하기에는 너무 오래되었으면
`sima-cli neat install sentinel`로 설치하거나 업데이트하십시오. 설치되어 있지만
실행 중이 아니면 `simaai-sentinel.service`를 시작하십시오.

이 API는 로컬 DevKit의 `/run/simaai-sentinel/api.sock`에만
연결합니다. SSH를 사용하거나 원격 보드를 선택하지 않습니다. Insight와 향후 CLI
클라이언트는 Core를 거치지 않고 동등한 클라이언트로서 데몬에 연결합니다.

이 릴리스는 일회성 카탈로그 읽기를 제공합니다. 새로 고침 요청과 데몬
수명 주기 제어는 공개 Core API에 포함되지 않습니다.
