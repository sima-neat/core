---
title: 주변기기 카탈로그 데몬
---

# 주변기기 카탈로그 데몬

`simaai-peripherals`는 DevKit의 모든 로컬 클라이언트를 위해 하나의 현재 주변기기
카탈로그를 유지합니다. 전체 Core 배포에는 별도의 `sima-neat-peripherals` 구성 요소
패키지가 포함되며 `simaai-peripherals.service`가 관리합니다.

## 아키텍처

```text
Linux/udev notification or explicit refresh
                  |
                  v
       simaai-peripherals daemon
       - debounce refresh triggers
                  |
                  v
       private provider registry
       - isolate provider failures
       - retain last-good provider results
       - normalize and classify capabilities
                  |
                  v
       daemon catalog and event replay
                  |
                  v
       Unix socket clients
```

데몬은 모니터링, 공급자, 새로 고침 일정, 현재 스냅샷, 스캔 및 카탈로그 시퀀스와
재생 가능한 이벤트를 소유합니다. udev 알림은 새로 고침 신호일 뿐이며 공급자 코드가
카메라를 획득하거나 스트리밍하지 않고 기능 조회와 지원 분류를 수행합니다.

첫 비공개 공급자는 MIPI/libcamera 카메라를 검색합니다. 카메라가 없는 상태는 유효한
빈 데몬 카탈로그입니다. 공급자는 서로 독립적으로 실행됩니다. 실패한 공급자는 구조화된
문제를 추가하고 자체 마지막 정상 레코드만 유지합니다. 정상 공급자의 현재 결과는 계속
표시되므로 향후 마이크, LiDAR 또는 다른 공급자를 추가해도 하나의 백엔드 장애가 관련 없는
장치를 숨기지 않습니다.

## 로컬 API

HTTP/JSON은 다음 Unix 소켓으로만 노출되며 TCP를 사용하지 않습니다.

```text
/run/simaai-peripherals/api.sock
```

systemd 유닛은 권한이 없는 `sima` 사용자로 실행되고 `sima` 그룹의 런타임 디렉터리를
만들며 소켓 모드를 `0660`으로 설정합니다.

| 메서드와 경로 | 동작 |
| --- | --- |
| `GET /v1/health` | 준비 상태, 최신성, 인스턴스 ID, 리비전, 시퀀스 및 최근 오류를 반환합니다. |
| `GET /v1/catalog` | 내부적으로 일관된 현재 스냅샷을 반환합니다. |
| `GET /v1/events?after_sequence=N&wait_ms=M&instance_id=ID` | 커서 이후 이벤트를 반환합니다. `wait_ms` 최대값은 30000입니다. |
| `POST /v1/refresh` | 데몬 소유 검색 새로 고침을 예약하고 완료 토큰 `target_scan_sequence`와 함께 `202`를 반환합니다. |

요청은 8 KiB, 응답 본문은 4 MiB, 동시 연결은 64개로 제한됩니다. 너무 큰 카탈로그는
제한 없이 전송되는 대신 `response_too_large`로 실패합니다.

### V1 응답 스키마

상태 및 카탈로그 응답은 다음 필드를 공유합니다.

| 필드 | 의미 |
| --- | --- |
| `schema_version` | 정수 카탈로그 스키마 버전이며 현재 값은 `1`입니다. |
| `instance_id` | 이 데몬 프로세스를 위해 생성된 UUID입니다. |
| `state` | `starting`, `ready` 또는 `degraded`입니다. |
| `ready` | 유효한 빈 결과를 포함해 하나 이상의 공급자가 정상적으로 완료되었는지 나타냅니다. |
| `stale` | 더 최근의 실패 후 마지막 성공 스캔의 장치를 반환하는지 나타냅니다. |
| `revision` | 장치 스냅샷 리비전입니다. 오류나 복구만으로는 증가하지 않습니다. |
| `sequence` | 재생 가능한 최신 이벤트 시퀀스입니다. |
| `scan_sequence` | 완료된 검색 시도 횟수입니다. 성공과 실패 모두 증가시킵니다. |
| `last_success_at` | 최근 성공 스캔의 UTC 타임스탬프 또는 `null`입니다. |
| `last_attempt_at` | 최근 시도한 스캔의 UTC 타임스탬프 또는 `null`입니다. |
| `error` | `null` 또는 `code`와 실행 가능한 `reason`을 포함하는 객체입니다. |
| `issues` | 공급자별 장애와 해당 공급자가 마지막 정상 레코드를 유지했는지 여부입니다. |

`GET /v1/health`는 `api_version: "v1"`과 `device_count`도 반환합니다.
`GET /v1/catalog`는 대신 `devices`를 반환하며, 유형별 기능은 장치 유형 아래에
중첩됩니다.

```json
{
  "schema_version": 1,
  "instance_id": "7ff4c8a1-...",
  "state": "ready",
  "ready": true,
  "stale": false,
  "revision": 1,
  "sequence": 0,
  "scan_sequence": 1,
  "last_success_at": "2026-10-01T01:23:45.678Z",
  "last_attempt_at": "2026-10-01T01:23:45.678Z",
  "error": null,
  "issues": [],
  "devices": [
    {
      "id": "camera:imx477 5-001a",
      "type": "camera",
      "provider": "daemon.camera.libcamera",
      "camera": {
        "camera_name": "imx477 5-001a",
        "model": "imx477",
        "backend": "libcamera",
        "modes": [
          {
            "format": "NV12",
            "width": 1920,
            "height": 1080,
            "framerate_num": 30,
            "framerate_den": 1,
            "supported": true,
            "reason": ""
          }
        ]
      }
    }
  ]
}
```

공급자가 보고하지 않으면 `model`은 생략됩니다. 범위 모드에는 개별 `width`와
`height` 대신 `min_width`, `min_height`, `max_width`, `max_height`, `step_width`,
`step_height`를 포함하는 `size_range`가 있습니다.

이벤트 응답에는 `schema_version`, `instance_id`, `revision`, 최신 `sequence`, `scan_sequence`,
`resync_required`, `shutting_down`, `events` 배열이 포함됩니다. 각 이벤트에는
`sequence`, `revision`, `kind`가 포함됩니다. 장치 이벤트에는 `device_id`,
`device_type`, `previous` 및/또는 `current`가 추가됩니다. `error` 이벤트에는 구조화된
`error`가 포함됩니다. 성공한 스캔이 장치 변경 없이 저하 상태를 해제하면 장치가 없는
`recovered` 이벤트가 생성됩니다.

```bash
curl --unix-socket /run/simaai-peripherals/api.sock \
  http://localhost/v1/catalog
```

명시적 새로 고침이 수락되면 다음과 같은 토큰을 반환합니다.

```json
{"accepted":true,"target_scan_sequence":2}
```

health 또는 catalog의 `scan_sequence`가 이 값 이상이 될 때까지 폴링하십시오.
같은 대기 스캔이 시작되기 전에 수락된 요청은 토큰을 공유할 수 있습니다. 검색 실행 중에
수락된 요청은 다음 스캔을 대상으로 합니다.

### 카탈로그 식별자

데몬이 시작될 때마다 새 `instance_id`가 생성됩니다. 첫 성공 검색은 합성 `added`
이벤트 없이 리비전을 `1`로 설정합니다. 이후 한 번의 검색에서 여러 장치가 변경되어도
리비전은 한 번만 증가하며 해당 이벤트는 같은 리비전을 공유합니다.

완료된 검색 시도마다 `scan_sequence`가 증가합니다. 각 이벤트에는 단조 증가하는
`sequence`가 있으며 종류는 `added`, `removed`, `changed`, `error`, `recovered`입니다.
카메라 식별자는 불안정한 `/dev/videoN` 대신 공급자의 정확한 libcamera 이름을 사용합니다. 카메라 세부 정보는 모델, 모드 또는 크기 범위,
프레임레이트, 지원 분류 및 거부 이유를 보존합니다.

### 클라이언트 재동기화

클라이언트는 `instance_id`와 `sequence`를 모두 보관합니다.

1. `/v1/catalog`를 읽고 `instance_id`와 `sequence`를 저장합니다.
2. 해당 값으로 `/v1/events`를 롱 폴링합니다.
3. 반환된 이벤트를 시퀀스 순서로 적용합니다.
4. `resync_required`가 true이면 로컬 상태를 버리고 `/v1/catalog`를 다시 읽습니다.

데몬 재시작, 미래 커서 또는 재생 버퍼보다 오래된 커서는 재동기화가 필요합니다.
느리거나 연결이 끊긴 클라이언트는 모니터링 또는 다른 클라이언트를 막지 않습니다.

## 모니터링 및 새로 고침

데몬은 등록된 공급자가 선언한 하위 시스템으로 udev 필터를 구성합니다. 현재 카메라
공급자는 `media`와 `video4linux`를 감시합니다. 250 ms 디바운스 후 각 비공개 공급자를
한 번 실행합니다. 장치 이벤트를 만들지 않는 변경에는 `POST /v1/refresh`
또는 `systemctl reload`를 사용합니다.

주기적 스캔이나 유휴 상태의 상태 확인 타이머는 없습니다. 시작 후 모니터, API 리스너,
프로세스 감독자는 udev 알림, 명시적 새로 고침, 클라이언트 연결, 종료 신호 또는 작업자
오류가 도착할 때까지 커널의 블로킹 대기 상태로 잠듭니다. 디바운스 타이머는 실제 udev
알림 후에만 존재하며 클라이언트 I/O 기한은 연결을 처리하는 동안에만 존재합니다.

보드 전원이 켜진 상태에서 MIPI 리본 카메라를 물리적으로 연결하거나 분리하지
마십시오. 추가/제거 검증에는 안전하게 핫플러그할 수 있는 주변기기를 사용하십시오.

## 운영

```bash
sudo systemctl status simaai-peripherals
sudo systemctl reload simaai-peripherals
sudo systemctl restart simaai-peripherals
journalctl -u simaai-peripherals
```

서비스는 `sima-neat-peripherals` 패키지 설치 시 자동 시작되며 예기치 않은 실패 후 다시 시작됩니다. 정상
종료 시 진행 중인 검색을 기다리기 전에 롱 폴링 클라이언트를 깨우고 소켓을 제거합니다.
클라이언트 I/O에는 절대 기한이 있으며 외부 공급자가 반환하지 않으면 systemd가 최종
10초 중지 시간 제한을 적용합니다.

## 문제 해결

| 증상 | 조치 |
| --- | --- |
| 소켓이 없음 | `systemctl status simaai-peripherals`와 서비스 저널을 확인합니다. |
| 소켓 접근 거부 | 클라이언트가 `sima` 그룹에 속하는지 확인하고 세션을 다시 연결합니다. |
| 상태가 `degraded` | `/v1/health`의 `error`와 공급자별 `issues`를 확인합니다. 유지된 레코드는 `retained_last_good`로 표시됩니다. |
| `resync_required`가 true | 새 카탈로그로 로컬 상태를 교체하고 해당 시퀀스부터 계속합니다. |
| 설정 변경이 감지되지 않음 | `POST /v1/refresh`를 호출하거나 서비스를 다시 로드합니다. |
