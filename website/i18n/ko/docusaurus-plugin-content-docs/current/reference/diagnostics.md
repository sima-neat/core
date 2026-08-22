---
title: "진단 및 디버깅"
description: "그래프 보고서 진단, 런타임 오류 코드, 그래프 지표 아티팩트를 수집합니다."
sidebar_position: 9
---

# 진단 및 디버깅

## 그래프 보고서

`GraphReport`는 구조화된 진단 정보를 기록합니다.
- 파이프라인 문자열(재현용)
- 표준 `error_code` (자동 분류)
- `repro_note` (사람이 요약한 내용 + 힌트)
- 노드 보고서 및 소유 요소 이름
- 버스 메시지 및 오류 세부 정보
- 선택 사항인 흐름/타이밍 카운터

오류가 발생하면 `NeatError`는 로그를 기록하거나 직렬화할 수 있는 `GraphReport`를 포함합니다.

## 오류 분류 체계

프레임워크 오류는 안정적인 코드 패밀리를 사용합니다.

| 오류 코드 | 의미 | 일반적인 해결 방법 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 노드 순서/구조 계약 위반 | 푸시 파이프라인의 경우 `Input()`을 먼저, 풀 파이프라인의 경우 `Output()`을 마지막에 배치하여 파이프라인이 올바르게 구성되었는지 확인합니다. |
| `misconfig.caps` | 프레임워크의 caps-override 또는 인접 노드 계약 불일치 | `caps_override`와 선언된 노드 계약을 일치시킵니다. |
| `misconfig.input_shape` | 입력 텐서/프레임/샘플의 형태 또는 데이터 유형이 모델 요구 사항과 일치하지 않습니다. | 예상되는 형태와 데이터 유형을 제공하거나 모델 전처리 설정을 구성하십시오. |
| `misconfig.runtime_abi_mismatch` | Neat과 런타임 플러그인이 호환되지 않는 ABI를 사용합니다. | 버전이 일치하는 Neat Library 및 런타임을 설치하십시오. |
| `misconfig.graph_element_name` | 사용자 정의 요소에는 고정된 노드 이름을 할당할 수 없습니다. | 사용자 정의 요소에 고정적이고 고유한 이름을 지정하십시오. |
| `misconfig.input_capacity` | 원본 이미지가 전처리 입력 용량을 초과했습니다. | `input_max_width` 또는 `input_max_height` 값을 늘리거나, 모델 단계 전에 크기를 조정하세요. |
| `misconfig.media_caps` | 인접한 GStreamer 스테이지는 호환되지 않는 미디어 캡을 요구합니다. | 형식, 해상도 및 프레임 속도를 일치시키거나 변환 단계를 삽입하십시오. |
| `misconfig.media_format` | 지원되지 않는 미디어 형식이 스테이지에 사용되었습니다. | 지원되는 형식으로 구성하거나 형식 변환을 추가하십시오. |
| `misconfig.tensor_dtype_missing` | 텐서 계약에 dtype/형식 정보가 없습니다. | 상위 계약에서 지원되는 텐서 dtype을 선언하십시오. |
| `misconfig.option_out_of_range` | 현재 텐서에 대해 선택한 스테이지 옵션이 유효하지 않습니다. | 진단 메시지에 표시된 범위 내의 값을 선택하십시오. |
| `build.parse_launch` | `gst_parse_launch` 오류는 더 이상 구체적으로 분류되지 않습니다. | 파서 컨텍스트에 대한 자세한 내용은 첨부된 보고서를 참조하십시오. |
| `build.pipeline_syntax` | 사용자 지정 GStreamer 파이프라인 구문이 유효하지 않습니다. | `gst-launch-1.0`를 사용하여 파이프라인을 수정하고 유효성을 검사하십시오. |
| `build.plugin_missing` | 필요한 GStreamer 요소 또는 코덱 플러그인이 설치되지 않았습니다. | 해당 플러그인을 설치하거나 교체한 후 `gst-inspect-1.0`를 사용하여 확인하십시오. |
| `build.property_invalid` | 요소 속성이 알 수 없거나 유효하지 않습니다. | 속성 이름과 값을 `gst-inspect-1.0`를 사용하여 확인하십시오. |
| `runtime.pull` | 더 구체적인 근본 원인 없이 풀(pull) 작업이 실패했습니다. | 첨부된 보고서와 첫 번째 상위 오류를 검토하십시오. |
| 런타임 요소가 실패했습니다. `runtime.element_failed` | 더 구체적인 매핑 없이 특정 단계가 실패했습니다. | 보고된 단계와 해당 단계의 상위 입력값을 수정하십시오. |
| `runtime.output_timeout` | 구성된 런타임 제한 시간 내에 결과가 도착하지 않았습니다. | 소스 흐름을 확인하거나 예상되는 런타임 제한 시간을 늘리십시오. |
| `runtime.unexpected_eos` | 파이프라인이 필요한 출력 전에 EOS에 도달했습니다. | 조기에 EOS에 도달한 소스를 확인하고 충분한 입력을 제공하십시오. |
| `io.parse` | JSON 또는 스테이지 구성 파싱/스키마 실패 | 구성 구문 및 필수 필드 유효성 검사 |
| `io.open` | 그래프 저장/불러오기 파일 열기/읽기/쓰기 실패 | 경로 존재 여부, 권한 및 저장 장치 상태를 확인하십시오. |
| `io.file_not_found` | 입력 파일이 존재하지 않습니다. | 경로를 수정하고 파일이 DevKit에 있는지 확인하십시오. |
| `io.permission_denied` | 파일 또는 장치를 읽을 수 없습니다. | 소유권/권한을 수정하세요. |
| `io.rtsp_connection_failed` | RTSP 소스에 연결할 수 없습니다. | URL, 연결 가능 여부, 서버 및 인증 정보를 확인하십시오. |
| `io.camera_not_found` | 요청하신 카메라를 사용할 수 없습니다. | 사용 가능한 카메라를 선택하거나 기본 카메라를 사용하십시오. |
| `io.model_not_found` | 요청하신 모델 아카이브가 존재하지 않습니다. | 모델 경로를 수정하고 설치되었는지 확인하십시오. |
| `io.source_ended` | 입력 소스가 정상적으로 종료되었습니다. | 입력을 중단하거나 더 많은 입력을 제공하십시오. |
| `codec.invalid_h264_stream` | 입력에 유효한 H.264 프레임이 없습니다. | 완전한 H.264 스트림을 제공하거나 코덱을 수정하십시오. |
| `codec.decode_failed` | 스트림을 수신한 후 디코딩에 실패했습니다. | 코덱과 입력 데이터의 무결성을 확인하십시오. |
| `codec.encode_failed` | 제공된 프레임을 인코딩할 수 없습니다. | 입력 형식, 해상도 및 인코더 설정을 확인하십시오. |
| `resource.memory_allocation_failed` | 필요한 메모리 할당에 실패했습니다. | 워크로드의 메모리 사용량을 줄이고 다른 애플리케이션 또는 파이프라인에서 사용 중인 메모리를 확보하십시오. |
| `resource.device_memory_exhausted` | 장치 DMA/CMA 할당 실패 | 동시에 처리하는 스트림 수, 해상도 또는 버퍼링을 줄이십시오. |
| `resource.output_pool_exhausted` | 모든 출력 버퍼가 사용 중입니다. | 제로 복사 출력을 해제하거나 소유된 복사본을 사용하세요. |
| `resource.buffer_too_small` | 버퍼의 크기가 선언된 페이로드보다 작습니다. | 올바른 차원/스트라이드를 지정하거나 필요한 바이트 수를 할당하십시오. |
| `resource.disk_full` | 저장 공간이 부족하여 쓰기 작업이 실패했습니다. | 여유 공간을 확보하거나 다른 저장 위치를 선택하세요. |
| `infra.dispatcher_unavailable` | 가속 런타임을 가져올 수 없습니다. | 경쟁하는 워크로드를 중지하고 호환성을 확인하십시오. DevKit |
| `infra.accelerator_execution_failed` | 가속기가 모델 단계를 실행할 수 없습니다. | 파이프라인을 다시 시작하고 동시에 실행되는 가속기 작업을 줄이십시오. |
| `DispatcherUnavailable` | `infra.dispatcher_unavailable`의 이전 버전 | 핸들러를 표준 인프라 코드로 마이그레이션 |
| `internal.plugin_failure` | 사용자 조치가 필요한 분류 없이 플러그인이 실패했습니다. | 보고서를 캡처하고 지원팀에 문의하십시오. |

`PullError.code`는 동일한 분류 체계를 사용합니다(예외 경로뿐만 아니라).
C++ 및 Python 상수 이름과 이전의 대략적인 코드와 일치하는 애플리케이션에 대한 마이그레이션 지침은 [오류 코드 목록](/reference/error-codes)를 참조하십시오.

프로덕션 메시지는 의도적으로 GStreamer 내부 정보를 생략합니다. 플러그인 디버그 상세 수준을 높이면 원시 GError 도메인/코드, 요소 팩토리, 메시지 및 구조화된 플러그인 세부 정보가 추가됩니다. URI 사용자 정보, `auth`, `playback-token`, `hdnts`, `stream-key` 및 `tkn`을 포함한 인식된 자격 증명 및 URL 비밀 매개변수는 두 가지 형식으로 저장되기 전에 삭제됩니다. 보고서에 표시되는 파이프라인 문자열, 노드 조각, 재현 명령 및 직렬화된 JSON은 내부적으로 유지되는 실행 가능한 파이프라인을 변경하지 않고 삭제됩니다.

## 프로그래밍 방식 처리

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build(input);
  simaai::neat::Sample out;
  simaai::neat::PullError perr;
  const auto st = run.pull(500, out, &perr);
  if (st == simaai::neat::PullStatus::Error) {
    if (perr.code == simaai::neat::error_codes::kMediaCaps) {
      // Fix the incompatible upstream/downstream media contract.
    } else {
      // Handle another specific code, including future codes, or report it.
    }
  }
} catch (const simaai::neat::NeatError& e) {
  if (e.report().error_code == simaai::neat::error_codes::kPluginMissing) {
    // Install or replace the missing GStreamer component.
  }
}
```

## 디버깅 설정 (환경)

주요 환경 변수(자세한 내용은 [아키텍처](/develop-apps/contribute/architecture) 참조):
- `SIMA_GST_DOT_DIR`: 실패 시 그래프(graph) 형식으로 DOT 파일을 생성합니다.
- `SIMA_GST_BOUNDARY_PROBES`: 경계 흐름 카운터
- `SIMA_GST_ELEMENT_TIMINGS`: 요소별 타이밍
- `SIMA_GST_FLOW_DEBUG`: 요소별 흐름 카운터
- `SIMA_GST_ENFORCE_NAMES`: 명명 규칙 적용

실패한 명령어에 대해 편집된 원본 GStreamer 컨텍스트를 `NeatError::what()` 및 `GraphReport.repro_note`에 추가하려면, 두 변수를 모두 설정하십시오.

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug`는 Neat Library 설정이 아닙니다. 일반적인 작동 시에는 자세한 출력을 비활성화 상태로 유지하십시오. 이는 짧은 진단 실행을 위해 사용되며, 인식된 자격 증명 필드는 삭제되더라도 배포에 특정한 경로 또는 미디어 주소를 포함할 수 있습니다.

## 디버깅 워크플로우

1) 먼저 `GraphReport.error_code`를 캡처하고, 오류를 분류하여 범주별로 정리합니다.
2) 구체적인 상황과 내장된 힌트를 제공하기 위해 `GraphReport.repro_note`를 캡처합니다.
3) 파이프라인 텍스트를 캡처합니다: `Graph::describe_backend()` 또는 `last_pipeline()`.
4) 구조화된 진단 정보를 캡처합니다: `MeasureReport::to_text()` 또는 `NeatError::report()`.
5) `GraphReport.bus`를 검사하여 첫 번째 터미널 `ERROR`의 원인과 자세한 내용을 확인합니다.
6) 런타임에서 문제가 발생하거나 시간이 초과되는 경우, 경계/요소 프로브를 활성화하여 문제 발생 지점을 파악합니다.

권장되는 지원 패키지:
- `error_code`
- `repro_note`
- 전체 `pipeline_string` 파이프라인
- 처음 3~5개의 터미널 버스 오류를 표시합니다(`GraphReport.bus`).
- run/validate에서 사용된 환경 변수 재정의

## 고객 그래프 성능 아티팩트

처리량/지연 시간/전력 보고를 위해 그래프 실행 JSON 내보내기를 사용하는 것이 좋습니다.

```cpp
RunOptions opt;
opt.enable_board_power();        // graph-level power when supported by the board/SOM
Run run = graph.build(opt);

// run your normal push/pull loop inside a measurement window, then:
auto report = run.start_measurement().stop();
std::cout << report.to_text();
```

내보내기는 범위를 명확하게 유지합니다.

- `run.graph_metrics.throughput_fps` 및 `run.graph_metrics.power`는 그래프 수준의 주요 지표입니다.
- `run.node_metrics[]`에는 노드/플러그인의 지연 시간만 포함되며, 노드/플러그인의 전력 소비량은 의도적으로 포함되지 않습니다.
- `latency_semantics`와 `aggregation`은 값이 프로그램 실행 기간 동안의 값인지, 아니면 특정 시간 간격 동안 측정된 값인지 알려줍니다.
- `plugin_metrics_unattributed[]`는 정확히 하나의 노드에 매핑되지 않은 커널/플러그인 행을 보존합니다.

측정된 시간 간격의 경우, `Run::start_measurement()`를 사용하고 반환된 `MeasureReport`를 `run_to_json(run, report, ...)` / `save_run_json(run, report, ...)`에 전달합니다. 측정된 시간 간격 노드의 `min_ms`/`max_ms`는 누적 최소/최대 카운터를 정확하게 뺄 수 없기 때문에 사용할 수 없도록 표시됩니다(시간 간격 내 로컬 카운터가 없으면 불가능).

참고: 현재 DVT 보드는 옵션 파이프라인 및 JSON 형식을 검증할 수 있지만, 전력량 측정값은 수치적으로 신뢰할 수 있는 것으로 간주되지 않습니다. SOM 하드웨어가 전력 관련 데이터의 검증을 위한 의도된 플랫폼입니다.

## 일반적인 문제점 → 해결 방법

| 증상 | 가능한 원인 | 해결 방법 |
| --- | --- | --- |
| `missing ... plugin` | GStreamer 플러그인을 찾을 수 없습니다. | `GST_PLUGIN_PATH`를 확인하고, `gst-inspect-1.0 <plugin>`을 실행하십시오. |
| `appsink 'mysink' not found` | 터미널이 없습니다. `Output()` | `Output`이 실행/빌드 파이프라인의 마지막 노드인지 확인하십시오. |
| `caps_override is set; renegotiation disabled` | Caps 잠금 | `caps_override`를 제거하거나 입력된 Caps Lock 설정을 유지 |
| `tensor caps change not supported` | 런타임 시 텐서의 형태/데이터 유형 변경 | 텐서의 형태/데이터 유형을 안정적으로 유지 (재협상 없음) |

구조화된 플러그인 오류 및 문제 해결을 위한 팁은 다음을 참조하십시오.
[문제 해결](/reference/troubleshooting).
