---
title: 주변 장치 카탈로그
description: 보드 로컬 카탈로그 서비스가 보고한 DevKit 주변 장치를 나열합니다
sidebar_position: 8
---

# 주변 장치 카탈로그

로컬 DevKit에 현재 연결된 장치를 확인하려면 주변 장치 카탈로그를 사용하십시오. 이 API는 C++와
Python에서 사용할 수 있으며 두 언어에서 동일한 형식화된 스냅샷을 반환합니다.

신뢰할 수 있는 카탈로그는 SiMa Sentinel(`simaai-sentinel.service`)이 소유합니다. Sentinel은 이 카탈로그를
`GET /v1/peripherals`로 제공하는 보드 로컬 데몬입니다. 각 `list()` 호출은 Sentinel에 제한 시간이 있는
요청을 한 번 수행합니다. Core는 하드웨어를 스캔하거나,
두 번째 카탈로그를 캐시하거나, 다른 검색 경로로 폴백하지 않습니다.

## 주변 장치 나열

Python:

```python
import pyneat

catalog = pyneat.peripherals.list()
for peripheral in catalog:
    print(peripheral.id, peripheral.type, peripheral.provider)
```

C++:

```cpp
#include <neat.h>

auto catalog = simaai::neat::peripherals::list();
for (const auto& peripheral : catalog) {
  // Use peripheral.id, peripheral.type, and peripheral.provider.
}
```

반환된 카탈로그는 반복할 수 있습니다. 또한 데몬 `instance_id`, 카탈로그 `revision`, 이벤트 `sequence`,
완료된 `scan_sequence`, 최신 상태와 타임스탬프, 현재 구조화된 오류, 공급자 문제 및 `devices`
컬렉션을 포함합니다.

준비된 카탈로그에 장치가 없어도 성공한 결과입니다. 성능 저하 상태의 카탈로그에는 마지막으로 성공한
장치 목록이 포함될 수 있습니다. 이 경우 `stale`은 `true`이고 `error` 또는 `issues`가 새로 고침 실패를
설명합니다.

## 카메라 세부 정보

`peripheral.type == "camera"`이면 `peripheral.camera`에 다음 항목이 포함됩니다.

| 필드 | 의미 |
| --- | --- |
| `camera_name` | `CameraInputOptions`에서 허용하는 정확한 libcamera 이름(선택 사항)입니다. 현재 입력 API가 선택할 수 없는 카메라에는 없습니다. |
| `model` | 공급자가 보고한 경우의 장치 모델입니다. |
| `backend` | `mipi` 또는 `v4l2`와 같은 검색 백엔드입니다. |
| `modes` | 개별 크기 또는 명시적 크기 범위, 프레임 속도, 지원 플래그 및 거부 이유입니다. |

지원 여부는 이 Core 패키지가 `/usr/share/simaai-sentinel/support/neat-core.json`에 설치하는 규칙을 Sentinel이 적용하여 분류하므로, 결과는 설치된 `CameraInput`의 기본 libcamera 프로필(`camera_name`으로 카메라를 선택하는 `profile=Default`)과 일치합니다. 클라이언트는 그 결과를 보존하며 카메라를 검사, 재분류, 획득, 구성 또는
스트리밍하지 않습니다. 따라서 카탈로그의 지원 표시는 이후의 독점 획득 성공을 보장하지 않습니다.

이 규칙은 Metoak SIMOR raw V4L2 프로필(`CameraProfile::MetoakSimor`, RAW8 1920×360, `profile`과 선택적 `device`로 선택)을 분류하지 않습니다. 규칙에는 센서별 조건이 없으므로 Sentinel이 `mipi` 카메라로 나열한 SIMOR 센서(이름이 `simor_metoak`으로 시작)는 다른 MIPI 센서와 같은 ISP 모드 분류를 받습니다. 이 카메라에서 `supported: true`는 모드가 기본 프로필의 백엔드, 형식, 프레임 속도, ISP 출력 크기와 일치한다는 뜻일 뿐이며, libcamera가 해당 센서를 검증했다는 뜻이 아닙니다. SIMOR 카메라는 카탈로그의 `camera_name`이 아니라 [`CameraInput`](/reference/nodes/camera-input)에 설명된 `profile=MetoakSimor`로 선택하세요.

## 모든 주변 장치 형식의 세부 정보

모든 주변 장치는 `type`과 관계없이 형식별 세부 정보를 `details_json`에 담습니다. 이는 Sentinel이
`type`과 같은 이름의 레코드 키(예: `camera`, `microphone`, `lidar`) 아래에 게시하는 압축된 JSON
객체입니다. 해당 키가 없거나 `null`이면 `details_json`은 `"{}"`입니다. Python에서는
`details_json`을 새 `dict`로 디코딩하는 `details`도 제공합니다.

`details_json`은 Core에 형식화된 접근자가 없는 주변 장치 형식을 읽는 공식적인 방법입니다. 새 장치
형식은 Core를 업데이트하지 않아도 Sentinel이 보고하는 즉시 사용할 수 있습니다. 이 Core 릴리스가
알지 못하는 필드를 포함하여 모든 필드와 값이 보존됩니다. JSON은 다시 직렬화되므로 키 순서와 공백은
데몬 응답과 다를 수 있습니다. 카메라에는 `details_json`과 형식화된 `camera` 필드가 모두 있습니다.

Python:

```python
for peripheral in pyneat.peripherals.list():
    if peripheral.type == "microphone":
        print(peripheral.id, peripheral.details.get("channels"))
```

C++:

```cpp
#include <nlohmann/json.hpp>

for (const auto& peripheral : simaai::neat::peripherals::list()) {
  if (peripheral.type == "microphone") {
    const auto details = nlohmann::json::parse(peripheral.details_json);
    // Read details.value("channels", 0) and other provider fields.
  }
}
```

`details_json`은 모든 JSON 라이브러리로 구문 분석할 수 있습니다. 이 예제에서는 nlohmann/json을 사용합니다.

`camera` 이외의 형식에서 세부 정보 값이 JSON 객체가 아니어도 `list()`는 실패하지 않습니다. 해당
주변 장치는 카탈로그에 남고 `details_json`은 `"{}"`가 됩니다. Core는 형식화된 접근자가 없는 형식의
세부 정보를 그 밖에는 검증하지 않습니다. 잘못된 `camera` 세부 정보는 프로토콜 결함이므로 `list()`가
구문 분석 오류로 실패합니다. `camera`와 같은 형식화된 필드는 알지 못하는 프로토콜 v1 선택 필드를
무시하지만, 해당 필드는 `details_json`에서 계속 사용할 수 있습니다. Core는 Sentinel이 각 스냅샷과
함께 게시하는 최상위 `changes` 로그와 `support` 상태도 수락하지만 노출하지 않습니다.

## 실패 및 범위

서비스 누락 또는 카탈로그를 제공하지 못하는 오래된 서비스, 권한 거부, 요청 시간 초과, 준비되지 않은 데몬, 잘못되거나 너무 크거나 호환되지 않는 응답이
발생하면 `list()`는 안정적인 코드가 포함된 `NeatError`를 발생시킵니다. 메시지에는 다음 운영 조치가
포함됩니다. [오류 코드 카탈로그](./error-codes.md)를 참조하십시오.

Sentinel이 설치되어 있지 않거나 주변 장치 카탈로그를 제공하지 못할 만큼 오래되었으면
`sima-cli neat install sentinel`로 설치하거나 업데이트하십시오. 설치되어 있지만
실행 중이 아니면 `simaai-sentinel.service`를 시작하십시오.

이 API는 로컬 DevKit의 `/run/simaai-sentinel/api.sock`에만 연결합니다. SSH를 사용하거나 원격 보드를
선택하지 않습니다. Insight와 향후 CLI 클라이언트는 Core를 통하지 않고 동급 클라이언트로 데몬에
연결합니다.

이 릴리스는 일회성 카탈로그 읽기를 제공합니다. 이벤트 구독, 새로 고침 요청 및 데몬 수명 주기 제어는
공개 Core API에 포함되지 않습니다.
