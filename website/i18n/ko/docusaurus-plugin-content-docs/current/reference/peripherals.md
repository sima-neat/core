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

지원 여부는 이 Core 패키지가 `/usr/share/simaai-sentinel/support/neat-core.json`에 설치하는 규칙을 Sentinel이 적용하여 분류하므로, 결과는 설치된 `CameraInput`과 일치합니다. 카탈로그의 지원 표시는 이후의 독점 획득 성공을 보장하지 않습니다.

## 모든 주변 장치 형식의 세부 정보

모든 주변 장치는 `type`과 관계없이 형식별 세부 정보를 `details_json`에 담습니다. 이는 Sentinel이
`type`과 같은 이름의 레코드 키(예: `camera`, `microphone`, `lidar`) 아래에 게시하는 압축된 JSON
객체입니다. 해당 키가 없거나 `null`이면 `details_json`은 `"{}"`입니다. Python에서는
`details_json`을 새 `dict`로 디코딩하는 `details`도 제공합니다.

```python
for peripheral in pyneat.peripherals.list():
    if peripheral.type == "microphone":
        print(peripheral.id, peripheral.details.get("channels"))
```

C++에서는 모든 JSON 라이브러리로 `details_json`을 구문 분석합니다. 새 장치 형식은 Core를 업데이트하지
않아도 Sentinel이 보고하는 즉시 사용할 수 있습니다. `camera` 이외의 형식에서 세부 정보 값이 JSON
객체가 아니어도 `list()`는 실패하지 않습니다. 해당 주변 장치는 카탈로그에 남고 `details_json`은
`"{}"`가 됩니다. 잘못된 `camera` 세부 정보는 `list()`를 구문 분석 오류로 실패시킵니다.

## 실패

서비스 누락 또는 카탈로그를 제공하지 못하는 오래된 서비스, 권한 거부, 요청 시간 초과, 준비되지 않은 데몬, 잘못되거나 너무 크거나 호환되지 않는 응답이
발생하면 `list()`는 안정적인 코드가 포함된 `NeatError`를 발생시킵니다. 메시지에는 다음 운영 조치가
포함됩니다. [오류 코드 카탈로그](./error-codes.md)를 참조하십시오.

Sentinel이 설치되어 있지 않거나 주변 장치 카탈로그를 제공하지 못할 만큼 오래되었으면
`sima-cli neat install sentinel`로 설치하거나 업데이트하십시오. 설치되어 있지만
실행 중이 아니면 `simaai-sentinel.service`를 시작하십시오.
