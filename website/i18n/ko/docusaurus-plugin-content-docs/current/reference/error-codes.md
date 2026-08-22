---
title: "오류 코드 목록"
description: "안정적인 프레임워크의 오류 코드, 오류 발생 시점, 그리고 대응 방법"
sidebar_position: 7
---

# 오류 코드 목록

Neat는 `NeatError` 및 `PullError`를 통해 유형 오류를 표시합니다. 각 오류는 안정적인 오류 코드, 사람이 읽을 수 있는 메시지, 그리고 가능하다면 구조화된 컨텍스트를 포함하는 `GraphReport`를 제공합니다.

오류 코드를 사용하여 프로그래밍 방식으로 오류를 분류합니다. 메시지는 개발자에게 표시합니다. 모든 공개 상수의 전체 집합은 [`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h)에 있습니다.

## 동작 방식의 변경 사항 및 마이그레이션

진단 분류 체계는 이제 특정 GStreamer 근본 원인을 보존합니다. 공개 메서드 서명은 변경되지 않았지만, 정확한 오류 문자열을 비교하는 코드는 수정해야 할 수 있습니다.

| 이전 매칭 | 이제 더 구체적인 코드가 반환됨 | 마이그레이션 |
| --- | --- | --- |
| 런타임 GStreamer 협상 오류, 또는 `misconfig.caps` 또는 `misconfig.media_format` (형식만 호환되지 않는 경우)에 대한 미디어 코드 처리 | `misconfig.media_caps` | 캡 오버라이드 및 인접 노드 계약의 프레임워크 검증에 대해서만 `misconfig.caps`를 유지합니다. |
| 모든 `gst_parse_launch` 실패, 즉 `build.plugin_missing`, `build.property_invalid` 또는 `build.pipeline_syntax` 오류에 대해 `build.parse_launch`를 사용합니다. | 특정 빌드 코드를 처리합니다. 분류되지 않은 파서 오류의 경우 `build.parse_launch`를 기본값으로 사용합니다. | |
| 전파된 버스 오류에 대한 `runtime.pull` | 근본 원인 코드(예: `misconfig.media_caps`, `io.rtsp_connection_failed` 또는 `resource.output_pool_exhausted`) | 근본 원인 코드를 처리하고 기본 분기를 유지합니다. `runtime.pull`은 특정 원인이 없는 로컬 풀 오류에 대한 대체 방법으로 유지됩니다. |

문자열 리터럴을 반복해서 사용하는 대신 C++ 또는 Python 상수를 사용하세요. 최신 Neat Library 빌드에서 추가된 코드에 대해서는 항상 기본 경로를 유지하세요.

## 공용 상수

동일한 값이 두 가지 언어 API에서 모두 제공됩니다.

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
| `DispatcherUnavailable` (레거시) | `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## 잘못된 구성

| 코드 | 발생 시점 | 해결 방법 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 그래프의 토폴로지가 유효하지 않거나 입력/출력 경계가 누락되었습니다. | 그래프 연결을 수정하고 필요한 `Input` 또는 `Output` 노드를 추가하십시오. |
| `misconfig.caps` | 프레임워크 검증 중에 캡스 오버라이드 또는 인접 노드 계약이 호환되지 않습니다. | 선언된 형식, 크기, 비율 및 인접 노드 계약을 일치시키십시오. |
| `misconfig.input_shape` | 입력 텐서가 예상되는 형태 또는 데이터 유형과 일치하지 않습니다. | 예상되는 입력을 제공하거나 모델 옵션을 통해 모델 전처리를 구성하십시오. |
| `misconfig.runtime_abi_mismatch` | Neat과 설치된 런타임 플러그인이 호환되지 않는 ABI를 사용합니다. | 호환되는 Neat Library 및 런타임 플러그인 버전을 설치하십시오. |
| `misconfig.graph_element_name` | 사용자 정의 조각에 안정적인 노드 이름을 할당할 수 없는 요소가 포함되어 있습니다. | 사용자 정의 요소에 안정적이고 고유한 이름을 지정하십시오. |
| `misconfig.media_caps` | 연결된 GStreamer 단계는 호환되지 않는 미디어 캡을 요구합니다. | 단계를 조정하거나 필요한 변환, 크기 조정 또는 비율 변환 노드를 삽입하십시오. |
| `misconfig.media_format` | 연결된 스테이지에서 호환되지 않는 미디어 형식이 필요합니다. | 공통 형식을 구성하거나 명시적인 형식 변환을 추가하십시오. |
| `misconfig.input_capacity` | 원본 이미지가 구성된 전처리 입력 용량을 초과합니다. | `input_max_width` 및 `input_max_height` 값을 늘리거나, 모델 단계 전에 원본 이미지의 크기를 조정하십시오. |
| `misconfig.tensor_dtype_missing` | 텐서 계약에서 데이터 유형 또는 형식이 누락되었습니다. | 상위 텐서 계약에서 지원되는 데이터 유형을 선언하십시오. |
| `misconfig.option_out_of_range` | 현재 입력 계약에 대해 유효하지 않은 옵션입니다. | 진단에 표시된 범위 내의 값으로 옵션을 설정하십시오. |

## 빌드 실패

| 코드 | 발생 시점 | 해결 방법 |
| --- | --- | --- |
| `build.parse_launch` | GStreamer가 생성된 파이프라인을 빌드할 수 없습니다. | 사용자 정의 조각, 요소 속성 및 플러그인 가용성을 확인하십시오. |
| `build.pipeline_syntax` | 사용자 정의 GStreamer 구문이 올바르지 않습니다. | 해당 부분을 수정하고 유효성을 검증합니다. `gst-launch-1.0`. |
| `build.plugin_missing` | 필요한 GStreamer 요소 또는 코덱 플러그인을 사용할 수 없습니다. | 해당 구성 요소를 설치하거나 교체한 후 `gst-inspect-1.0`를 사용하여 확인하십시오. |
| `build.property_invalid` | 요소 속성 이름 또는 값이 유효하지 않습니다. | `gst-inspect-1.0 <element>`를 사용하여 속성을 확인하십시오. |

## 런타임 오류

| 코드 | 발생 시점 | 해결 방법 |
| --- | --- | --- |
| `runtime.pull` | 더 구체적인 오류 코드가 없으면 런타임 풀 작업이 실패합니다. | 첨부된 보고서와 첫 번째 상위 오류를 확인하십시오. |
<<번역>>
| `runtime.element_failed` | 파이프라인의 특정 단계가 더 자세한 분류 없이 중단되었습니다. | 보고된 단계 구성과 해당 단계의 상위 입력값을 수정하십시오. |
| `runtime.output_timeout` | 구성된 대기 시간이 초과되기 전에 결과가 도착하지 않았습니다. | 소스 흐름과 역압을 확인하거나, 대기가 예상되는 경우 런타임 제한 시간을 조정하십시오. |
| `runtime.unexpected_eos` | 파이프라인이 필요한 출력을 생성하기 전에 EOS(End of Stream)에 도달했습니다. | 입력 데이터에서 조기 EOS가 발생하는지 확인하고 충분한 입력 데이터가 제공되었는지 확인하십시오. |

## 입출력 오류

| 코드 | 발생 시점 | 해결 방법 |
| --- | --- | --- |
| `io.parse` | Neat JSON, 모델 계약 또는 스테이지 구성을 파싱할 수 없습니다. | 구성 구문, 스키마 및 필수 필드를 확인합니다. |
| `io.open` | Neat이(가) 파일을 열거나, 장치에 접근하거나, 원격 리소스에 연결할 수 없습니다. | 경로 또는 주소, 권한, 리소스 가용성을 확인하십시오. |
| `io.file_not_found` | 입력 파일이 존재하지 않습니다. | 경로를 수정하고 파일이 DevKit에 있는지 확인하십시오. |
| 필요한 권한으로 파일이나 장치를 열 수 없습니다. `io.permission_denied` | 보고된 리소스에 대한 소유권 또는 권한을 수정하십시오. | |
| `io.rtsp_connection_failed` | Neat이(가) RTSP 소스에 연결할 수 없습니다. | URL, 서버, 네트워크 연결 상태 및 인증 정보를 확인하십시오. |
| `io.camera_not_found` | 요청하신 카메라를 사용할 수 없습니다. | 사용 가능한 카메라를 선택하거나 기본 카메라를 사용하십시오. |
<<번역>>
| `io.model_not_found` | 요청하신 모델 아카이브가 존재하지 않습니다. | 모델 경로를 수정하고 해당 아카이브가 설치되었는지 확인하십시오. |
| `io.source_ended` | 입력 소스가 정상적으로 종료되었습니다. | 해당 소스의 데이터 소비를 중단하거나, 애플리케이션에서 더 많은 데이터를 필요로 하는 경우 추가 입력을 제공하십시오. |

## 파이프라인 실행 실패

| 코드 | 발생 시점 | 해결 방법 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 파이프라인 토폴로지가 유효하지 않거나, GStreamer 구성 후 최종 요소 이름이 중복되거나, 모호하거나, 누락되었습니다. | 각 명시적 요소에 대해 해당 요소가 속한 세그먼트 내에서 고유한 짧은 이름을 지정하십시오. `name=` 선언과 명명된 패드 참조가 동기화되도록 유지하십시오. |
| `build.parse_launch` | GStreamer는 구문, 플러그인 또는 속성이 유효하지 않아 최종 파이프라인 문자열을 구문 분석하거나 구성할 수 없습니다. | `GraphReport::pipeline_string`을 검사하고, `gst-launch-1.0`을 사용하여 해당 부분을 확인하고, `gst-inspect-1.0`을 사용하여 플러그인을 확인하십시오. |

이러한 검사는 `Graph::build()` 중에 자동으로 수행됩니다. 입력에 따라 연결되는 세그먼트의 경우, 첫 번째 입력이 세그먼트를 생성할 때 동일한 코드와 `GraphReport`가 표시될 수 있습니다.

## 코덱 오류

| 코드 | 발생 시점 | 해결 방법 |
| --- | --- | --- |
| `codec.invalid_h264_stream` | 입력에 유효한 H.264 프레임이 없습니다. | 완전한 H.264 스트림을 제공하고 구성된 코덱을 확인하십시오. |
| `codec.decode_failed` | 디코더가 수신된 스트림을 디코딩할 수 없습니다. | 코덱을 확인하고 인코딩된 입력이 완전하고 손상되지 않았는지 확인하십시오. |
| `codec.encode_failed` | 인코더가 제공된 프레임을 인코딩할 수 없습니다. | 입력 형식, 해상도 및 인코더 설정을 확인하십시오. |

## 리소스 오류

| 코드 | 발생 시점 | 해결 방법 |
| --- | --- | --- |
| `resource.memory_allocation_failed` | 특정 장치와 관련된 원인 없이 필요한 메모리 할당에 실패했습니다. | 스트림 수, 해상도 또는 버퍼링을 줄이고 다른 작업에서 사용하는 메모리를 확보하십시오. |
| `resource.device_memory_exhausted` | 장치 DMA/CMA 메모리가 부족합니다. | 동시에 실행되는 스트림 수, 입력 해상도 또는 버퍼 깊이를 줄이십시오. |
| `resource.output_pool_exhausted` | 모든 출력 버퍼가 사용 중입니다. | 제로 복사 출력 버퍼를 즉시 해제하거나 소유된 복사본을 사용하십시오. |
| `resource.buffer_too_small` | 버퍼의 크기가 선언된 프레임 또는 텐서 페이로드보다 작습니다. | 상위 수준의 차원과 스트라이드를 수정하거나, 필요한 바이트 수를 할당하십시오. |
| `resource.disk_full` | 대상 위치에 사용 가능한 여유 공간이 부족하여 쓰기 작업이 실패했습니다. | 여유 공간을 확보하거나 다른 대상 위치를 선택하십시오. |

## 인프라 장애

| 코드 | 발생 시점 | 해결 방법 |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat이(가) 가속기 런타임에 접근할 수 없습니다. | DevKit과의 호환성을 확인하고 가속기를 독점적으로 사용하는 워크로드를 중지하십시오. |
| `infra.accelerator_execution_failed` | 가속기가 모델 단계를 실행할 수 없습니다. | 파이프라인을 다시 시작하고 동시에 실행되는 가속기 작업량을 줄이십시오. |

## 내부 오류

| 코드 | 발생 시점 | 해결 방법 |
| --- | --- | --- |
<<번역>>
| `internal.plugin_failure` | 사용자가 조치를 취할 수 있는 분류 없이 Neat 플러그인이 실패했습니다. | 첨부된 `GraphReport`를 캡처하여 지원팀에 오류를 보고합니다. |

`DispatcherUnavailable`는 호환성을 위해 허용되는 이전 버전의 철자입니다. 새로운 애플리케이션은 `infra.dispatcher_unavailable` 및 `error_codes::kDispatcherUnavailable` 상수를 사용해야 합니다.

## 프로그래밍 방식으로 오류를 처리합니다.

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

`PullError.code`는 동일한 상수를 사용합니다. `what()`을 파싱하거나 사람이 읽을 수 있는 텍스트와 비교하지 마십시오.

## 추가 정보

- [진단 및 디버깅](/reference/diagnostics) — 운영 환경 메시지, 디버그 정보 등
  `GraphReport` 컬렉션.
- [플러그인 오류 형식](/reference/error_format) — GStreamer 플러그인의 구조화된 오류 형식
  오류.
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror) — 발생한 예외 유형입니다.
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) — 구조화된 오류 정보.
