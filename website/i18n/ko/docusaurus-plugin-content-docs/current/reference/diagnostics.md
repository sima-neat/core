---
title: "진단 및 디버깅"
description: "GraphReport 진단, 런타임 오류 코드, 그래프 지표 아티팩트를 수집합니다"
sidebar_position: 9
---

# 진단 및 디버깅

## GraphReport

`GraphReport`는 다음과 같은 구조화된 진단 정보를 수집합니다.
- 파이프라인 문자열(재현용)
- 표준 `error_code`(기계적 분류용)
- `repro_note`(사람이 읽는 요약 + 힌트)
- 노드 보고서 및 소유한 요소 이름
- 버스 메시지 및 오류 세부 정보
- 선택적 흐름/타이밍 카운터

오류가 발생하면 `NeatError`에는 로그로 기록하거나
직렬화할 수 있는 `GraphReport`가 담겨 있습니다.

## 오류 분류 체계

프레임워크 오류는 다음과 같은 안정적인 코드 계열을 사용합니다.

| 오류 코드 | 의미 | 일반적인 해결 방법 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 노드 순서/형태 계약 위반 | 푸시 파이프라인에서는 `Input()`을 맨 앞에, 풀 파이프라인에서는 `Output()`을 맨 뒤에 두십시오 |
| `misconfig.caps` | 프레임워크 caps 재정의 또는 인접 노드 계약 불일치 | `caps_override`와 선언된 노드 계약을 일치시키십시오 |
| `misconfig.input_shape` | 입력 텐서/프레임/샘플의 형태 또는 데이터 유형이 모델 계약과 일치하지 않습니다 | 예상 형태와 데이터 유형을 제공하거나 모델 전처리를 구성하십시오 |
| `misconfig.runtime_abi_mismatch` | Neat와 런타임 플러그인이 호환되지 않는 ABI를 사용합니다 | 버전이 일치하는 Neat Library와 런타임을 설치하십시오 |
| `misconfig.graph_element_name` | 사용자 정의 요소에 안정적인 노드 이름을 할당할 수 없습니다 | 사용자 정의 요소에 안정적이고 고유한 이름을 지정하십시오 |
| `misconfig.input_capacity` | 원본 이미지가 전처리 입력 용량을 초과합니다 | `input_max_width` / `input_max_height`를 늘리거나, 모델 단계 전에 크기를 조정하십시오 |
| `misconfig.media_caps` | 인접한 GStreamer 단계가 서로 호환되지 않는 미디어 caps를 요구합니다 | 형식, 해상도, 프레임 속도를 일치시키거나 변환을 삽입하십시오 |
| `misconfig.media_format` | 단계가 지원되지 않는 미디어 형식을 받았습니다 | 지원되는 형식을 구성하거나 형식 변환을 삽입하십시오 |
| `misconfig.tensor_dtype_missing` | 텐서 계약에 dtype/형식이 없습니다 | 업스트림 계약에 지원되는 텐서 dtype을 선언하십시오 |
| `misconfig.option_out_of_range` | 현재 텐서에 대해 단계 옵션이 유효하지 않습니다 | 진단에 표시된 범위 안의 값을 선택하십시오 |
| `build.parse_launch` | `gst_parse_launch` 실패에 더 구체적인 분류가 없습니다 | 첨부된 보고서에서 파서 컨텍스트를 검사하십시오 |
| `build.pipeline_syntax` | 사용자 정의 GStreamer 조각의 구문이 잘못되었습니다 | 조각을 수정하고 `gst-launch-1.0`으로 검증하십시오 |
| `build.plugin_missing` | 필요한 GStreamer 요소 또는 코덱 플러그인이 설치되어 있지 않습니다 | 설치하거나 교체한 다음 `gst-inspect-1.0`으로 확인하십시오 |
| `build.property_invalid` | 요소 속성을 알 수 없거나 속성이 잘못되었습니다 | `gst-inspect-1.0`으로 속성 이름과 값을 확인하십시오 |
| `runtime.pull` | 더 구체적인 근본 원인 없이 pull이 실패했습니다 | 첨부된 보고서와 첫 번째 업스트림 오류를 검사하십시오 |
| `runtime.element_failed` | 더 구체적인 매핑 없이 단계가 실패했습니다 | 보고된 단계와 그 업스트림 입력을 수정하십시오 |
| `runtime.output_timeout` | 구성된 시간 제한 전에 출력이 도착하지 않았습니다 | 소스 흐름을 확인하거나 예상되는 시간 제한을 늘리십시오 |
| `runtime.unexpected_eos` | 필요한 출력 전에 파이프라인이 EOS에 도달했습니다 | 소스의 조기 EOS 여부를 확인하고 충분한 입력을 제공하십시오 |
| `io.parse` | JSON 또는 단계 구성의 구문 분석/스키마 실패 | 구성 구문과 필수 필드를 검증하십시오 |
| `io.open` | 그래프 저장/로드 시 파일 열기/읽기/쓰기 실패 | 경로 존재 여부, 권한, 저장소 상태를 확인하십시오 |
| `io.file_not_found` | 입력 파일이 존재하지 않습니다 | 경로를 수정하고 파일이 DevKit에 있는지 확인하십시오 |
| `io.permission_denied` | 파일 또는 장치를 읽을 수 없습니다 | 소유권/권한을 수정하십시오 |
| `io.rtsp_connection_failed` | RTSP 소스에 연결할 수 없습니다 | URL, 도달 가능성, 서버, 자격 증명을 확인하십시오 |
| `io.camera_not_found` | 요청한 카메라를 사용할 수 없습니다 | 보고된 카메라를 선택하거나 기본 카메라를 사용하십시오 |
| `io.model_not_found` | 요청한 모델 아카이브가 존재하지 않습니다 | 모델 경로를 수정하고 설치되어 있는지 확인하십시오 |
| `io.source_ended` | 입력 소스가 정상적으로 끝에 도달했습니다 | 소비를 중지하거나 더 많은 입력을 제공하십시오 |
| `io.response_too_large` | 크기가 제한된 로컬 프로토콜 응답이 크기 한도를 초과했습니다 | 서로 일치하는 클라이언트와 서비스 버전을 설치하십시오 |
| `codec.invalid_h264_stream` | 입력에 유효한 H.264 프레임이 없습니다 | 완전한 H.264 스트림을 제공하거나 코덱을 수정하십시오 |
| `codec.decode_failed` | 디코더가 스트림을 받아들인 후 실패했습니다 | 코덱과 입력 무결성을 확인하십시오 |
| `codec.encode_failed` | 인코더가 제공된 프레임을 인코딩할 수 없었습니다 | 입력 형식, 해상도, 인코더 설정을 확인하십시오 |
| `resource.memory_allocation_failed` | 필요한 메모리 할당이 실패했습니다 | 워크로드의 메모리 사용량을 줄이고 다른 애플리케이션이나 파이프라인이 사용하는 메모리를 확보하십시오 |
| `resource.device_memory_exhausted` | 장치 DMA/CMA 할당이 실패했습니다 | 동시 스트림 수, 해상도 또는 버퍼링을 줄이십시오 |
| `resource.output_pool_exhausted` | 모든 출력 버퍼가 계속 사용 중입니다 | 제로 복사 출력을 해제하거나 소유한 복사본을 사용하십시오 |
| `resource.buffer_too_small` | 버퍼가 선언된 페이로드보다 작습니다 | 크기/stride를 수정하거나 필요한 바이트를 할당하십시오 |
| `resource.disk_full` | 저장소가 가득 차서 쓰기가 실패했습니다 | 공간을 확보하거나 다른 대상 위치를 선택하십시오 |
| `infra.dispatcher_unavailable` | 가속기 런타임을 획득할 수 없습니다 | 경쟁하는 워크로드를 중지하고 DevKit 호환성을 확인하십시오 |
| `infra.accelerator_execution_failed` | 가속기가 모델 단계를 실행할 수 없었습니다 | 파이프라인을 다시 시작하고 동시에 실행되는 가속기 작업을 줄이십시오 |
| `infra.peripheral_daemon_unavailable` | SiMa Sentinel이 주변 장치 카탈로그를 제공할 수 없습니다 | `sima-cli neat install sentinel`로 Sentinel을 설치 또는 업데이트하거나 `simaai-sentinel.service`를 시작하십시오 |
| `infra.peripheral_daemon_timeout` | 주변 장치 카탈로그 요청이 제시간에 완료되지 않았습니다 | `simaai-sentinel.service`와 해당 저널을 확인한 다음 다시 시도하십시오 |
| `DispatcherUnavailable` | `infra.dispatcher_unavailable`의 레거시 표기 | 핸들러를 표준 인프라 코드로 마이그레이션하십시오 |
| `internal.plugin_failure` | 사용자가 조치할 수 있는 분류 없이 플러그인이 실패했습니다 | 보고서를 캡처하고 지원팀에 문의하십시오 |

`PullError.code`도 같은 분류 체계를 사용합니다(예외 경로에만 해당하지 않음).
C++ 및 Python 상수 이름과, 이전의 대략적인 코드와 비교하던 애플리케이션을 위한
마이그레이션 지침은 [오류 코드 카탈로그](/reference/error-codes)를 참조하십시오.

프로덕션 메시지는 의도적으로 GStreamer 내부 정보를 생략합니다. 플러그인 디버그
상세 수준에서는 원시 GError 도메인/코드, 요소 팩토리, 메시지와
구조화된 플러그인 세부 정보가 추가됩니다. URI
사용자 정보, `auth`, `playback-token`, `hdnts`, `stream-key`, `tkn`을 포함하여 인식된 자격 증명과 URL 비밀 매개변수는 어느
형식이든 저장되기 전에 가려집니다. 보고서에 표시되는 파이프라인 문자열, 노드 조각, 재현 명령,
직렬화된 JSON은 내부에 보관된 실행 가능한 파이프라인을 변경하지 않고 가려집니다.

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

## 디버그 설정(환경 변수)

주요 환경 변수(자세한 내용은 [아키텍처](/develop-apps/contribute/architecture) 참조):
- `SIMA_GST_DOT_DIR`: 실패 시 DOT 그래프 작성
- `SIMA_GST_BOUNDARY_PROBES`: 경계 흐름 카운터
- `SIMA_GST_ELEMENT_TIMINGS`: 요소별 타이밍
- `SIMA_GST_FLOW_DEBUG`: 요소별 흐름 카운터
- `SIMA_GST_ENFORCE_NAMES`: 이름 지정 계약 강제 적용

가려진 원시 GStreamer 컨텍스트를 `NeatError::what()` 및
`GraphReport.repro_note`에 추가하려면 실패하는 명령에 두 변수를 모두 설정하십시오.

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug`는 Neat Library 설정이 아닙니다. 일반 운영 중에는 상세 출력을 비활성화한
상태로 두십시오. 상세 출력은 짧은 진단 실행을 위한 것이며, 인식된 자격 증명 필드가 가려지더라도 배포별 경로나
미디어 주소를 포함할 수 있습니다.

## 디버그 워크플로

1) 먼저 `GraphReport.error_code`를 캡처하고 분류 체계에 따라 실패를 분류하십시오.
2) 구체적인 컨텍스트와 내장 힌트를 위해 `GraphReport.repro_note`를 캡처하십시오.
3) 파이프라인 텍스트 캡처: `Graph::describe_backend()` 또는 `last_pipeline()`.
4) 구조화된 진단 캡처: `MeasureReport::to_text()` 또는 `NeatError::report()`.
5) `GraphReport.bus`에서 첫 번째 최종 `ERROR`의 소스와 세부 정보를 검사하십시오.
6) 런타임이 정체되거나 시간 초과되면 경계/요소 프로브를 활성화하여 흐름이 멈춘 위치를 찾으십시오.

권장 지원 번들:
- `error_code`
- `repro_note`
- 전체 `pipeline_string`
- 처음 3~5개의 최종 버스 오류(`GraphReport.bus`)
- run/validate에 사용한 환경 변수 재정의

## 고객용 그래프 성능 아티팩트

처리량/지연 시간/전력 보고에는 그래프 실행 JSON 내보내기를 사용하는 것이 좋습니다.

```cpp
RunOptions opt;
opt.enable_board_power();        // graph-level power when supported by the board/SOM
Run run = graph.build(opt);

// run your normal push/pull loop inside a measurement window, then:
auto report = run.start_measurement().stop();
std::cout << report.to_text();
```

내보내기는 범위를 명시적으로 구분합니다.

- `run.graph_metrics.throughput_fps` 및 `run.graph_metrics.power`는 그래프 수준의 주요 지표입니다.
- `run.node_metrics[]`에는 노드/플러그인 지연 시간만 포함되며, 노드/플러그인 전력은 의도적으로 제외됩니다.
- `latency_semantics`와 `aggregation`은 값이 실행 전체 기간의 값인지 측정 창의 차이값인지 알려 줍니다.
- `plugin_metrics_unattributed[]`는 정확히 하나의 노드에 매핑할 수 없었던 커널/플러그인 행을 보존합니다.

측정 창의 경우 `Run::start_measurement()`를 사용하고 반환된 `MeasureReport`를
`run_to_json(run, report, ...)` / `save_run_json(run, report, ...)`에 전달하십시오. 측정 창의 노드
`min_ms`/`max_ms`는 사용할 수 없음으로 표시됩니다. 창 로컬 카운터 없이는 누적 최소/최대 카운터를
정확하게 뺄 수 없기 때문입니다.

전력 참고: 현재 DVT 보드로 옵션 전달 경로와 JSON 형태는 검증할 수 있지만, 전력
측정값은 수치적으로 신뢰할 수 있는 것으로 간주되지 않습니다. 전력 수치 검증에는
SOM 하드웨어를 사용하는 것이 의도된 방식입니다.

## 일반적인 실패 → 해결 방법

| 증상 | 가능한 원인 | 해결 방법 |
| --- | --- | --- |
| `missing ... plugin` | GStreamer 플러그인을 찾을 수 없음 | `GST_PLUGIN_PATH`를 확인하고 `gst-inspect-1.0 <plugin>`을 실행하십시오 |
| `appsink 'mysink' not found` | 종단 `Output()` 누락 | run/build 파이프라인에서 `Output`이 마지막 노드인지 확인하십시오 |
| `caps_override is set; renegotiation disabled` | caps 고정됨 | `caps_override`를 제거하거나 입력 caps를 고정된 상태로 유지하십시오 |
| `tensor caps change not supported` | 런타임 중 텐서 형태/dtype 변경 | 텐서 형태/dtype을 안정적으로 유지하십시오(재협상 없음) |

구조화된 플러그인 오류와 조치 가능한 힌트는
[문제 해결](/reference/troubleshooting)을 참조하십시오.

## 플랫폼 런타임 복구

Platform 3.0.0에서 사용할 수 없는 디스패처는 조사해야 할 오류입니다. 레거시 서비스를 시작하라는 요청이 아닙니다. Core는 디스패처 오류에 대응해 MLA 메모리를 초기화하거나 원격 프로세서를 재설정하지 않습니다. 레거시 복구 코드와 스크립트는 제거되었으므로 플랫폼에서 승인한 복구 절차를 사용합니다.

드라이버가 완료 여부를 알 수 없다고 보고하면 DMA 버퍼와 원래 풀 대여를 유지해야 합니다. 파일 디스크립터를 닫거나 앱을 중지하거나 서비스를 다시 시작해도 하드웨어가 메모리 접근을 멈췄다는 증거가 되지 않습니다. 오류 보고서를 수집하고 다시 시도하기 전에 플랫폼에서 승인한 복구 절차를 사용합니다.

### Core와 Internals 버전 맞추기

호환되는 B1157 Internals 패키지로 Core를 빌드하고 설치합니다. Core는 공개 C++ ABI 버전과 별도로 런타임 프로파일, 커널 소스 리비전, SDK sysroot 증명 기록을 확인합니다. 같은 공개 ABI를 가진 이전 패키지도 호환되는 대체품이 아닙니다.

설치 프로그램은 패키지를 바꾸기 전에 번들된 `neat-runtime` 프로파일과 호환되는 `neat-gst-plugins` 버전을 확인합니다. 플랫폼 검사 재정의는 런타임 버전 일치 검사를 우회하지 않습니다. 검사에 실패하면 호환되는 번들을 구해야 하며 기록을 교체하거나 이전 런타임을 강제로 설치하지 않습니다.

보드에 설치하기 전에 CVU나 하드웨어 코덱을 사용하는 앱을 중지합니다. 전체 설치 프로그램은 모든 패키지를 설치한 뒤 준비된 EV74 펌웨어를 활성화하며 해당 장치가 열려 있으면 EV74 재설정을 거부합니다. `NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=OFF`로 설정하면 펌웨어를 준비 상태로 유지하고 나중에 `sudo /usr/libexec/sima-neat-firmware/install.sh --activate`로 활성화할 수 있습니다.
