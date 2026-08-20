---
title: "JSON 메타데이터 전송"
description: "메타데이터 송신 UDP JSON 통신 규약"
sidebar_position: 3
slug: /develop-apps/advanced-concepts/metadata_sender
---

# JSON 메타데이터 전송

외부 뷰어, 레코더 또는 서비스가 UDP를 통해 UTF-8 JSON 메타데이터를 수신할 때 `MetadataSender`를 사용합니다. Insight는 이 통신 규약을 이해하는 수신자 중 하나입니다.

## 통신 규약

- 기본 호스트: `127.0.0.1`
- 기본 메타데이터 포트: `9100`
- 채널 포트 규칙: `metadata_port_base + channel`
- 기본 전송 모드: 논블로킹(`MSG_DONTWAIT`)
- 페이로드 인코딩: UTF-8 JSON 텍스트
- 필수 최상위 필드: `type`, `data`
- 최대 논리적 페이로드: 65,507바이트

`MetadataSender`는 각 UDP 페이로드를 1200바이트 이하로 유지합니다. 최대 1200바이트까지의 JSON 페이로드는 변경되지 않은 하나의 데이터그램으로 유지됩니다. 더 큰 페이로드는 다음 12바이트 이진 헤더를 사용하여 청크로 분할됩니다.

| 바이트 | 크기 | 값 |
|---|---|---|
| 0 | 1 | 매직 바이트 `0x4e` |
| 1 | 1 | 프로토콜 버전 `0x01` |
| 2 | 8 | 부호 없는 64비트 빅 엔디언 정수로서의 메시지 ID |
| 10 | 1 | 0부터 시작하는 청크 인덱스 |
| 11 | 1 | 총 청크 수 |

각 청크는 최대 1188 JSON 바이트를 포함합니다. 수신자는 JSON을 파싱하기 전에 동일한 송신자 주소와 메시지 ID를 가진 청크를 청크 인덱스 순서대로 재조립합니다. UDP 전송은 최선을 다하는 방식입니다. 송신자는 실패한 청크를 다시 전송하지 않으며, `send_raw_json(...)` 또는 `send_metadata(...)`는 첫 번째 로컬 전송 실패 후 `false`를 반환합니다.

수신자는 변경되지 않은 JSON 데이터그램과 버전이 지정된 청크를 모두 수신해야 합니다. 청크 재조립 기능이 있는 버전으로 Insight를 업데이트하거나 이 Neat Library 버전과 함께 업데이트합니다. 이전 Insight 버전은 최대 1200바이트까지의 페이로드를 계속 수신하지만, 더 큰 청크 페이로드를 디코딩할 수 없습니다.

Insight의 경우, 메타데이터 채널 `N`을 `9000 + N`의 비디오 UDP 스트림과 연결합니다. Insight 또는 다른 수신자가 컨테이너 포트 매핑 뒤에서 실행되는 경우, 앱에서 매핑된 호스트와 포트를 명시적으로 전달합니다.

추적, 추적 대상 및 기타 사용자 지정 메타데이터는 일반 JSON으로 전송할 수 있습니다. 뷰어 오버레이 지원은 수신자별로 다릅니다. Insight 추적 시각화는 `sima-neat/insight#8`에서 별도로 추적됩니다.

## C++

```cpp
simaai::neat::MetadataSenderOptions opt;
opt.host = "127.0.0.1";
opt.channel = 0;
opt.metadata_port_base = 9100;

std::string err;
simaai::neat::MetadataSender sender(opt, &err);

sender.send_metadata(
    "tracking",
    R"({"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]})",
    12345,
    "frame-7",
    &err);
```

`send_metadata(...)`는 `data_json`을 검증하고 다음 형식의 메시지를 생성합니다.

```json
{
  "type": "tracking",
  "timestamp": 12345,
  "frame_id": "frame-7",
  "data": {
    "tracks": [
      {
        "id": "trk-1",
        "bbox": [10, 20, 30, 40]
      }
    ]
  }
}
```

호출자가 이미 전체 최상위 페이로드를 구성한 경우에만 `send_raw_json(...)`을 사용하십시오.

```cpp
sender.send_raw_json(
    R"({"type":"object-detection","data":{"objects":[{"id":"obj_1","label":"car","confidence":0.92,"bbox":[120,80,96,64]}]}})",
    &err);
```

## 실시간 전송은 기본적으로 논블로킹 방식으로 작동합니다.

`MetadataSender`는 기본적으로 각 데이터그램에 `MSG_DONTWAIT`를 적용하므로, 로컬에서 전송 버퍼가 과부하 상태일 때 비디오 또는 추론 작업을 전송하는 스레드의 실행을 지연시킬 수 없습니다. 커널이 데이터그램을 즉시 수신할 수 없는 경우, 전송은 대기하는 대신 `false`를 반환합니다. 해당 메타데이터 패킷을 삭제된 것으로 처리하고 실시간 작업을 계속 진행합니다. UDP 전송은 보장되지 않습니다.

기본 생성자와 기본 전송 옵션은 동일합니다.

```cpp
simaai::neat::MetadataSenderSendOptions send_opt;
simaai::neat::MetadataSender sender(opt, send_opt, &err);
```

배송 시도를 명시적으로 차단하기를 원하는 발신자는 다음 옵션을 선택할 수 있습니다.

```cpp
simaai::neat::MetadataSenderSendOptions send_opt;
send_opt.nonblocking = false;
simaai::neat::MetadataSender sender(opt, send_opt, &err);
```

`stats()`를 사용하여 네트워크 혼잡과 다른 유형의 오류를 구별하고, 명시적인 차단 모드에서는 통화 지연을 감지합니다.

```cpp
const auto stats = sender.stats();
std::cerr << "sent=" << stats.datagrams_sent
          << " would_block=" << stats.would_block
          << " enobufs=" << stats.no_buffer_space
          << " max_send_ns=" << stats.max_send_duration_ns << '\n';
```

`stats()`는 데이터 전송이 진행 중일 때 안전하게 읽을 수 있습니다. 이 결과를 단일 시점의 트랜잭션 스냅샷이 아닌 동시 진단 스냅샷으로 간주하십시오.

## 파이썬

```python
import json
import pyneat

opt = pyneat.MetadataSenderOptions()
opt.host = "127.0.0.1"
opt.channel = 0
opt.metadata_port_base = 9100

sender = pyneat.MetadataSender(opt)

sender.send_metadata(
    "object-detection",
    json.dumps(
        {
            "objects": [
                {
                    "id": "obj_1",
                    "label": "car",
                    "confidence": 0.92,
                    "bbox": [120, 80, 96, 64],
                }
            ]
        }
    ),
    12345,
    "frame-7",
)

stats = sender.stats()
print(stats.datagrams_sent, stats.would_block, stats.max_send_duration_ns)
```

C++와 마찬가지로, 차단 동작이 필요한 경우에만 `send_opt.nonblocking = False`를 명시적으로 설정하고 두 번째 생성자 인수로 전달합니다.
