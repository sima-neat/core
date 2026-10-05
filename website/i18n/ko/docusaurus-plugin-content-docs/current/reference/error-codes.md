---
title: "오류 코드 카탈로그"
description: "안정적인 프레임워크 오류 코드, 각 코드의 발생 시점과 대응 방법"
sidebar_position: 7
---

# 오류 코드 카탈로그

Neat는 `NeatError`와 `PullError`를 통해 형식화된 실패를 보고합니다. 각 실패는 안정적인
오류 코드, 사람이 읽을 수 있는 메시지, 그리고 가능한 경우 구조화된 컨텍스트가 담긴 `GraphReport`를 제공합니다.

프로그래밍 방식의 분류에는 오류 코드를 사용하십시오. 메시지는 개발자에게 표시하십시오. 공개
상수의 전체 목록은
[`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h)에 있습니다.

## 동작 호환성이 깨지는 변경 및 마이그레이션

진단 분류 체계는 이제 구체적인 GStreamer 근본 원인을 보존합니다. 공개 메서드 시그니처는
변경되지 않았지만, 정확한 오류 문자열을 비교하는 코드는 마이그레이션이 필요할 수 있습니다.

| 이전 매칭 | 이제 반환되는 더 구체적인 코드 | 마이그레이션 |
| --- | --- | --- |
| 런타임 GStreamer 협상 오류에 대한 `misconfig.caps` | `misconfig.media_caps`, 또는 형식만 호환되지 않는 경우 `misconfig.media_format` | 미디어 코드를 처리하십시오. `misconfig.caps`는 caps 재정의와 인접 노드 계약에 대한 프레임워크 검증에만 유지하십시오. |
| 모든 `gst_parse_launch` 실패에 대한 `build.parse_launch` | `build.plugin_missing`, `build.property_invalid` 또는 `build.pipeline_syntax` | 구체적인 빌드 코드를 처리하십시오. `build.parse_launch`는 분류되지 않은 파서 실패에 대한 폴백으로 유지하십시오. |
| 전파된 버스 실패에 대한 `runtime.pull` | `misconfig.media_caps`, `io.rtsp_connection_failed` 또는 `resource.output_pool_exhausted` 같은 근본 원인 코드 | 근본 원인 코드를 처리하고 기본 분기를 유지하십시오. `runtime.pull`은 구체적인 원인이 없는 로컬 pull 실패에 대한 폴백으로 남습니다. |

문자열 리터럴을 반복하는 대신 C++ 또는 Python 상수를 사용하십시오. 더 새로운 Neat Library 빌드에서
도입된 코드에 대비해 항상 기본 경로를 유지하십시오.

## 공개 상수

두 언어 API에서 동일한 값을 사용할 수 있습니다.

| 오류 코드 | C++ | Python |
| --- | --- | --- |
| `misconfig.pipeline_shape` | `error_codes::kPipelineShape` | `pyneat.ERROR_PIPELINE_SHAPE` |
| `misconfig.caps` | `error_codes::kCaps` | `pyneat.ERROR_CAPS` |
| `misconfig.input_shape` | `error_codes::kInputShape` | `pyneat.ERROR_INPUT_SHAPE` |
| `misconfig.runtime_abi_mismatch` | `error_codes::kRuntimeAbiMismatch` | `pyneat.ERROR_RUNTIME_ABI_MISMATCH` |
| `misconfig.graph_element_name` | `error_codes::kGraphElementName` | `pyneat.ERROR_GRAPH_ELEMENT_NAME` |
| `misconfig.media_caps` | `error_codes::kMediaCaps` | `pyneat.ERROR_MEDIA_CAPS` |
| `misconfig.media_format` | `error_codes::kMediaFormat` | `pyneat.ERROR_MEDIA_FORMAT` |
| `misconfig.input_capacity` | `error_codes::kInputCapacity` | `pyneat.ERROR_INPUT_CAPACITY` |
| `misconfig.tensor_dtype_missing` | `error_codes::kTensorDtypeMissing` | `pyneat.ERROR_TENSOR_DTYPE_MISSING` |
| `misconfig.option_out_of_range` | `error_codes::kOptionOutOfRange` | `pyneat.ERROR_OPTION_OUT_OF_RANGE` |
| `build.parse_launch` | `error_codes::kParseLaunch` | `pyneat.ERROR_PARSE_LAUNCH` |
| `build.pipeline_syntax` | `error_codes::kPipelineSyntax` | `pyneat.ERROR_PIPELINE_SYNTAX` |
| `build.plugin_missing` | `error_codes::kPluginMissing` | `pyneat.ERROR_PLUGIN_MISSING` |
| `build.property_invalid` | `error_codes::kPropertyInvalid` | `pyneat.ERROR_PROPERTY_INVALID` |
| `runtime.pull` | `error_codes::kRuntimePull` | `pyneat.ERROR_RUNTIME_PULL` |
| `runtime.element_failed` | `error_codes::kRuntimeElementFailed` | `pyneat.ERROR_RUNTIME_ELEMENT_FAILED` |
| `runtime.output_timeout` | `error_codes::kOutputTimeout` | `pyneat.ERROR_OUTPUT_TIMEOUT` |
| `runtime.unexpected_eos` | `error_codes::kUnexpectedEos` | `pyneat.ERROR_UNEXPECTED_EOS` |
| `io.parse` | `error_codes::kIoParse` | `pyneat.ERROR_IO_PARSE` |
| `io.open` | `error_codes::kIoOpen` | `pyneat.ERROR_IO_OPEN` |
| `io.file_not_found` | `error_codes::kFileNotFound` | `pyneat.ERROR_FILE_NOT_FOUND` |
| `io.permission_denied` | `error_codes::kPermissionDenied` | `pyneat.ERROR_PERMISSION_DENIED` |
| `io.rtsp_connection_failed` | `error_codes::kRtspConnectionFailed` | `pyneat.ERROR_RTSP_CONNECTION_FAILED` |
| `io.camera_not_found` | `error_codes::kCameraNotFound` | `pyneat.ERROR_CAMERA_NOT_FOUND` |
| `io.model_not_found` | `error_codes::kModelNotFound` | `pyneat.ERROR_MODEL_NOT_FOUND` |
| `io.source_ended` | `error_codes::kSourceEnded` | `pyneat.ERROR_SOURCE_ENDED` |
| `io.response_too_large` | `error_codes::kResponseTooLarge` | `pyneat.ERROR_RESPONSE_TOO_LARGE` |
| `codec.invalid_h264_stream` | `error_codes::kInvalidH264Stream` | `pyneat.ERROR_INVALID_H264_STREAM` |
| `codec.decode_failed` | `error_codes::kDecodeFailed` | `pyneat.ERROR_DECODE_FAILED` |
| `codec.encode_failed` | `error_codes::kEncodeFailed` | `pyneat.ERROR_ENCODE_FAILED` |
| `resource.memory_allocation_failed` | `error_codes::kMemoryAllocationFailed` | `pyneat.ERROR_MEMORY_ALLOCATION_FAILED` |
| `resource.device_memory_exhausted` | `error_codes::kDeviceMemoryExhausted` | `pyneat.ERROR_DEVICE_MEMORY_EXHAUSTED` |
| `resource.output_pool_exhausted` | `error_codes::kOutputPoolExhausted` | `pyneat.ERROR_OUTPUT_POOL_EXHAUSTED` |
| `resource.buffer_too_small` | `error_codes::kBufferTooSmall` | `pyneat.ERROR_BUFFER_TOO_SMALL` |
| `resource.disk_full` | `error_codes::kDiskFull` | `pyneat.ERROR_DISK_FULL` |
| `infra.dispatcher_unavailable` | `error_codes::kDispatcherUnavailable` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE` |
| `infra.accelerator_execution_failed` | `error_codes::kAcceleratorExecutionFailed` | `pyneat.ERROR_ACCELERATOR_EXECUTION_FAILED` |
| `infra.peripheral_daemon_unavailable` | `error_codes::kPeripheralDaemonUnavailable` | `pyneat.ERROR_PERIPHERAL_DAEMON_UNAVAILABLE` |
| `infra.peripheral_daemon_timeout` | `error_codes::kPeripheralDaemonTimeout` | `pyneat.ERROR_PERIPHERAL_DAEMON_TIMEOUT` |
| `DispatcherUnavailable` (레거시) | `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## 잘못된 구성

| 코드 | 발생 시점 | 조치 방법 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 그래프의 토폴로지가 잘못되었거나 입력/출력 경계가 없습니다. | 그래프 연결을 수정하고 필요한 `Input` 또는 `Output` 노드를 추가하십시오. |
| `misconfig.caps` | 프레임워크 검증 중에 caps 재정의 또는 인접 노드 계약이 호환되지 않습니다. | 선언된 형식, 크기, 속도와 인접 노드 계약을 일치시키십시오. |
| `misconfig.input_shape` | 입력 텐서가 예상 형태 또는 데이터 유형과 일치하지 않습니다. | 예상 입력을 제공하거나 모델 옵션을 통해 모델 전처리를 구성하십시오. |
| `misconfig.runtime_abi_mismatch` | Neat와 설치된 런타임 플러그인이 호환되지 않는 ABI를 사용합니다. | 서로 일치하는 Neat Library 빌드와 런타임 플러그인 빌드를 설치하십시오. |
| `misconfig.graph_element_name` | 사용자 정의 조각에 안정적인 노드 이름을 할당할 수 없는 요소가 있습니다. | 사용자 정의 요소에 안정적이고 고유한 이름을 지정하십시오. |
| `misconfig.media_caps` | 연결된 GStreamer 단계가 서로 호환되지 않는 미디어 caps를 요구합니다. | 단계를 일치시키거나 필요한 변환, 크기 조정 또는 속도 변환 노드를 삽입하십시오. |
| `misconfig.media_format` | 연결된 단계가 서로 호환되지 않는 미디어 형식을 요구합니다. | 공통 형식을 구성하거나 명시적인 형식 변환을 추가하십시오. |
| `misconfig.input_capacity` | 원본 이미지가 구성된 전처리 입력 용량을 초과합니다. | `input_max_width`와 `input_max_height`를 늘리거나, 모델 단계 전에 원본의 크기를 조정하십시오. |
| `misconfig.tensor_dtype_missing` | 텐서 계약에 데이터 유형 또는 형식이 빠져 있습니다. | 업스트림 텐서 계약에 지원되는 데이터 유형을 선언하십시오. |
| `misconfig.option_out_of_range` | 현재 입력 계약에 대해 옵션이 유효하지 않습니다. | 진단에 표시된 범위 안의 값으로 옵션을 설정하십시오. |

## 빌드 실패

| 코드 | 발생 시점 | 조치 방법 |
| --- | --- | --- |
| `build.parse_launch` | GStreamer가 생성된 파이프라인을 빌드할 수 없습니다. | 사용자 정의 조각, 요소 속성 및 플러그인 사용 가능 여부를 확인하십시오. |
| `build.pipeline_syntax` | 사용자 정의 GStreamer 조각의 구문이 잘못되었습니다. | 조각을 수정하고 `gst-launch-1.0`으로 검증하십시오. |
| `build.plugin_missing` | 필요한 GStreamer 요소 또는 코덱 플러그인을 사용할 수 없습니다. | 구성 요소를 설치하거나 교체한 다음 `gst-inspect-1.0`으로 확인하십시오. |
| `build.property_invalid` | 요소 속성 이름 또는 값이 잘못되었습니다. | `gst-inspect-1.0 <element>`로 속성을 확인하십시오. |

## 런타임 실패

| 코드 | 발생 시점 | 조치 방법 |
| --- | --- | --- |
| `runtime.pull` | 더 구체적인 코드 없이 pull 작업이 실패합니다. | 첨부된 보고서와 첫 번째 업스트림 오류를 검사하십시오. |
| `runtime.element_failed` | 파이프라인 단계가 더 구체적인 분류 없이 중지됩니다. | 보고된 단계의 구성과 그 업스트림 입력을 수정하십시오. |
| `runtime.output_timeout` | 구성된 대기 시간이 만료되기 전에 출력이 도착하지 않습니다. | 소스 흐름과 백프레셔를 확인하거나, 대기가 예상되는 경우 시간 제한을 조정하십시오. |
| `runtime.unexpected_eos` | 파이프라인이 필요한 출력을 생성하기 전에 EOS에 도달합니다. | 입력에 조기 EOS가 있는지 확인하고 충분한 입력이 제공되었는지 확인하십시오. |

## I/O 실패

| 코드 | 발생 시점 | 조치 방법 |
| --- | --- | --- |
| `io.parse` | Neat가 JSON, 모델 계약 또는 단계 구성을 구문 분석할 수 없습니다. | 구성의 구문, 스키마 및 필수 필드를 검증하십시오. |
| `io.open` | Neat가 파일, 장치 또는 원격 리소스를 열 수 없습니다. | 경로 또는 주소, 권한, 리소스 가용성을 확인하십시오. |
| `io.file_not_found` | 입력 파일이 존재하지 않습니다. | 경로를 수정하고 파일이 DevKit에 있는지 확인하십시오. |
| `io.permission_denied` | 필요한 접근 권한으로 파일이나 장치를 열 수 없습니다. | 보고된 리소스의 소유권 또는 권한을 수정하십시오. |
| `io.rtsp_connection_failed` | Neat가 RTSP 소스에 연결할 수 없습니다. | URL, 서버, 네트워크 도달 가능성 및 자격 증명을 확인하십시오. |
| `io.camera_not_found` | 요청한 카메라를 사용할 수 없습니다. | 사용 가능한 카메라를 선택하거나 기본 카메라를 사용하십시오. |
| `io.model_not_found` | 요청한 모델 아카이브가 존재하지 않습니다. | 모델 경로를 수정하고 아카이브가 설치되어 있는지 확인하십시오. |
| `io.source_ended` | 입력 소스가 정상적인 끝에 도달합니다. | 해당 소스의 소비를 중지하거나, 애플리케이션에 더 많은 데이터가 필요하면 추가 입력을 제공하십시오. |
| `io.response_too_large` | 크기가 제한된 로컬 프로토콜 응답이 문서화된 크기 한도를 초과합니다. | 카탈로그 크기를 줄이거나 서로 일치하는 클라이언트와 서비스 버전을 설치하십시오. |

## 파이프라인 구체화 실패

| 코드 | 발생 시점 | 조치 방법 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 파이프라인 토폴로지가 잘못되었거나, GStreamer 구성 후 최종 요소 이름이 중복되거나 모호하거나 누락되었습니다. | 모든 명시적 요소에 해당 구체화 세그먼트 안에서 고유한 짧은 이름을 지정하십시오. `name=` 선언과 이름 있는 패드 참조를 동기화된 상태로 유지하십시오. |
| `build.parse_launch` | 구문, 플러그인 또는 속성이 잘못되어 GStreamer가 최종 launch 문자열을 구문 분석하거나 구성할 수 없습니다. | `GraphReport::pipeline_string`을 검사하고, `gst-launch-1.0`으로 조각을, `gst-inspect-1.0`으로 플러그인을 확인하십시오. |

이 검사는 `Graph::build()` 중에 자동으로 수행됩니다. 입력에 따라 달라지는 연결된 세그먼트의 경우,
첫 번째 입력이 세그먼트를 구체화할 때 동일한 코드와 `GraphReport`가 나타날 수 있습니다.

## 코덱 실패

| 코드 | 발생 시점 | 조치 방법 |
| --- | --- | --- |
| `codec.invalid_h264_stream` | 입력에 유효한 H.264 프레임이 없습니다. | 완전한 H.264 스트림을 제공하고 구성된 코덱을 확인하십시오. |
| `codec.decode_failed` | 디코더가 수락된 스트림을 디코딩할 수 없습니다. | 코덱을 확인하고 인코딩된 입력이 완전하며 손상되지 않았는지 점검하십시오. |
| `codec.encode_failed` | 인코더가 제공된 프레임을 인코딩할 수 없습니다. | 입력 형식, 해상도 및 인코더 설정을 확인하십시오. |

## 리소스 실패

| 코드 | 발생 시점 | 조치 방법 |
| --- | --- | --- |
| `resource.memory_allocation_failed` | 장치 관련 원인 없이 필요한 메모리 할당이 실패합니다. | 스트림 수, 해상도 또는 버퍼링을 줄이고, 다른 워크로드가 사용하는 메모리를 확보하십시오. |
| `resource.device_memory_exhausted` | 연속된 장치 DMA/CMA 메모리가 소진되었습니다. | 동시 스트림 수, 입력 해상도 또는 버퍼 깊이를 줄이십시오. |
| `resource.output_pool_exhausted` | 모든 출력 버퍼가 계속 사용 중입니다. | 제로 복사 출력을 즉시 해제하거나 소유한 복사본을 사용하십시오. |
| `resource.buffer_too_small` | 버퍼가 선언된 프레임 또는 텐서 페이로드보다 작습니다. | 업스트림의 크기와 stride를 수정하거나, 필요한 바이트 수를 할당하십시오. |
| `resource.disk_full` | 대상 위치의 여유 공간이 부족하여 쓰기가 실패합니다. | 공간을 확보하거나 다른 대상 위치를 선택하십시오. |

## 인프라 실패

| 코드 | 발생 시점 | 조치 방법 |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat가 가속기 런타임을 획득할 수 없습니다. | DevKit 호환성을 확인하고 가속기를 독점적으로 소유한 워크로드를 중지하십시오. |
| `infra.accelerator_execution_failed` | 가속기가 모델 단계를 실행할 수 없습니다. | 파이프라인을 다시 시작하고 동시에 실행되는 가속기 워크로드를 줄이십시오. |
| `infra.peripheral_daemon_unavailable` | 로컬 SiMa Sentinel API 소켓이 없거나 연결을 거부했거나, Sentinel의 주변 장치 검색이 비활성화되거나 중지되었거나, 설치된 Sentinel이 주변 장치 카탈로그를 제공하기에는 너무 오래되었거나, Sentinel이 예상하지 못한 다른 HTTP 상태를 반환했습니다. | `sima-cli neat install sentinel`로 Sentinel을 설치 또는 업데이트하거나 `simaai-sentinel.service`를 시작한 다음, 해당 저널을 검사하십시오. |
| `infra.peripheral_daemon_timeout` | 시간이 제한된 주변 장치 카탈로그 요청이 완료되지 않았습니다. | `simaai-sentinel.service`와 공급자 상태를 확인한 다음 다시 시도하십시오. |

## 내부 실패

| 코드 | 발생 시점 | 조치 방법 |
| --- | --- | --- |
| `internal.plugin_failure` | 사용자가 조치할 수 있는 분류 없이 Neat 플러그인이 실패합니다. | 첨부된 `GraphReport`를 캡처하여 지원팀에 실패를 보고하십시오. |

`DispatcherUnavailable`은 호환성을 위해 허용되는 레거시 표기입니다. 새 애플리케이션은
`infra.dispatcher_unavailable`과 `error_codes::kDispatcherUnavailable` 상수를 사용해야 합니다.

## 프로그래밍 방식으로 오류 처리하기

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build();
  // Push and pull application data.
} catch (const simaai::neat::NeatError& error) {
  if (error.report().error_code == simaai::neat::error_codes::kInputShape) {
    handle_input_contract_error(error.report());
  } else {
    throw;
  }
}
```

`PullError.code`도 동일한 상수를 사용합니다. `what()`을 구문 분석하거나 사람이 읽는 텍스트와 비교하지 마십시오.

## 추가 자료

- [진단 및 디버깅](/reference/diagnostics) — 프로덕션 메시지, 디버그 세부 정보 및
  `GraphReport` 수집.
- [플러그인 오류 형식](/reference/error_format) — GStreamer 플러그인 오류에 대한
  구조화된 계약.
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror) — 형식화된 예외.
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) — 구조화된 오류 컨텍스트.
