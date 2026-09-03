---
title: "환경 변수"
description: "런타임 및 빌더 환경 변수"
sidebar_position: 6
slug: /reference/environment-variables
---

# 환경 변수

이 페이지는 런타임, 빌더 및 SDK 도구에서 사용되는 환경 변수를 정리하여 보여줍니다. 대부분은 디버깅/진단용 설정이며, 대부분의 사용자는 문제 해결이 필요한 경우가 아니라면 무시해도 됩니다.

전체 감지된 목록은 [환경 변수 목록](./inventory)에서 확인할 수 있습니다.

> 참고: 일부 조정 가능한 매개변수는 내부용 또는 테스트용으로만 사용되며 변경될 수 있습니다. 여기에는 해당 매개변수가 포함되어 있습니다.

> 현재 코드 경로에서 사용되고 있기 때문입니다.

## SDK 웹 액세스

- `NFS_SERVER_HOST_IP=<address>` — 원격 주소를 구성하는 데 사용되는 SDK 호스트 주소입니다.
  Insight 및 브라우저 기반 VS Code URL입니다. 일반적으로 `sima-cli sdk setup`에서 제공됩니다.
- `CONTAINER_HOST_IP=<address>` — 원격 호스트 주소에 대한 이전 버전의 대체 방식입니다.
  `NFS_SERVER_HOST_IP` 호스트에 접속할 수 없는 경우.
- `OPENVSCODE_SERVER_HTTPS_PORT=<port>` — 브라우저 기반 VS Code에 대한 HTTPS 포트입니다.
  설정되지 않은 경우, `neat`는 VS Code URL을 표시하지 않습니다.
- `OPENVSCODE_SERVER_TOKEN=<token>` — 브라우저 기반 VS Code 액세스 토큰입니다.
  sima-cli에서 생성된 값은 URL에 안전하게 사용될 수 있습니다. 이 토큰이 포함된 URL은 자격 증명으로 취급하고 공유하지 마십시오.
- `OPENVSCODE_WORKSPACE=<path>` — 브라우저 기반 VS Code에서 열린 작업 공간
  (기본 `/workspace`).
- `OPENVSCODE_SERVER_WITHOUT_TOKEN=1` — 브라우저 기반 VS Code를 실행하고 광고합니다.
  토큰 인증 없이 사용합니다. 신뢰할 수 있는 로컬 환경에서만 사용하고, 서비스에 신뢰할 수 없는 네트워크에서 접근할 수 있는 경우 절대 활성화하지 마십시오.

## 통합 디버그 프로필 (권장)

- `SIMA_DEBUG_PROFILE=<components>` — 일반적인 진단 기능을 위한 통합 디버그 활성화 스위치입니다.
  - 구성 요소: `pipeline`, `graph`, `gst`, `appsink`, `inputstream` 또는 `all`.
  - 여러 구성 요소를 쉼표 또는 공백으로 구분할 수 있습니다(예: `pipeline,gst,inputstream`).
- `SIMA_DEBUG_LEVEL=<0..3>` — 통합 프로필에서 사용되는 디버그 상세 수준(기본값: `1`).
  - `0`: 비활성화됨
  - `1`: 핵심 디버그 로그
  - `2`: 자세한 진단 정보/버퍼 수준 추적
  - `3`: 최대 상세 정보 출력

기존의 변수별 디버그 설정은 여전히 작동하며, 명시적으로 설정할 경우 프로필 기본값을 재정의합니다.

## 핵심 빌드/실행

- `SIMA_PIPELINE_STRING_DEBUG=1` — 빌드 시 최종 gst-launch 문자열을 출력합니다.
- `SIMA_PIPELINE_STATE_DEBUG=1` — 추가 상태 변경 로그.
- `SIMA_PIPELINE_TEARDOWN_DEBUG=1` — 파이프라인 해체 단계를 기록합니다.
- `SIMA_PIPELINE_DRAIN_BEFORE_TEARDOWN_MS=<ms>` — 파이프라인 정리 전에 대기하는 시간(기본값: 1500).
- `SIMA_PIPELINE_DRAIN_MIN_OUTPUTS=<n>` — 파이프라인 종료 전에 처리해야 하는 최소 출력 개수 (기본값: 1).

## GStreamer 초기화 + 억제

- `SIMA_ALLOW_GST_INIT=1` — 이미 초기화된 경우 수동으로 `gst_init`를 수행할 수 있도록 허용합니다.
- `SIMA_GST_SUPPRESS_JSON_WARNINGS=0/1` — JSON 경고 메시지 표시를 억제합니다(기본값: true).
- `SIMA_GST_SUPPRESS_GOBJECT_ASSERTS=0/1` — GLib 어설션 로그를 억제합니다(기본값: true).
- `SIMA_GST_SUPPRESS_DEVICE_LOGS=0/1` — 장치 로그를 비활성화합니다(기본값: true).

## GStreamer 시간 초과

- `SIMA_STATE_CHANGE_TIMEOUT_MS=<ms>` — 파이프라인 상태 변경 시간 초과(기본값: 15000ms).
- `SIMA_GST_TEARDOWN_TIMEOUT_MS=<ms>` — 정리 시간 초과(기본값: 2000).
- `SIMA_GST_TEARDOWN_REAPER_MS=<ms>` — 왓치독 정리 시간(기본값: 250).
- `SIMA_GST_TEARDOWN_ASYNC=1` — 비동기 정리.
- `SIMA_GST_POLL_SLICE_MS=<ms>` — 앱 싱크 풀 요청에 대한 폴링 간격(기본값: 200).
- 선호하는 API 설정:
  - `ValidateOptions.preroll_timeout_ms` — validate() 함수의 프리롤 제한 시간입니다.
  - `RunOptions.input_timeout_ms` — build()/run() 입력 모드 풀(pull) 작업 시간 제한.
- 기존 방식의 환경 변수(옵션을 전달할 수 없는 경우에만 사용):
  - `SIMA_GST_VALIDATE_TIMEOUT_MS=<ms>` — validate() 함수 실행 시간 제한(기본값: 2000/10000).
  - `SIMA_GST_RUN_INPUT_TIMEOUT_MS=<ms>` — run() 함수 입력 시간 초과(기본값: 10000).

## 진단 및 프로브

- `SIMA_GST_DOT_DIR=<dir>` — 파이프라인 오류/디버깅을 위해 DOT 그래프를 출력합니다.
- `SIMA_GST_BOUNDARY_PROBES=1` — 경계 흐름 프로브를 부착합니다.
- `SIMA_GST_STAGE_TIMINGS=1` — 스테이지 타이밍 프로브.
- `SIMA_GST_ELEMENT_TIMINGS=1` — 요소 타이밍 프로브.
- `SIMA_GST_FLOW_DEBUG=1` — 요소 흐름 프로브.
- `SIMA_GST_ENFORCE_NAMES=1` — 빌드 시 이름 규칙을 적용합니다.
- `SIMA_GST_OPTIONS_DEBUG=1` — 빌드 중에 GStreamer 옵션을 기록합니다.
- `SIMA_GST_BUFFER_DEBUG_LIMIT=<n>` — 버퍼 디버그 출력 제한.
- `SIMA_GST_DETESS_INPUT_DEBUG=1` — 디테스 입력 디버그.
- `SIMA_GST_DETESS_OUTPUT_DEBUG=1` — 디테스 출력 디버그.
- `SIMA_GST_DETESS_POOL_DEBUG=1` — 디테스 풀 디버그.
- `SIMA_GST_APPSINK_BUFFER_DEBUG=1` — 앱 싱크 버퍼 디버깅.
- `SIMA_GST_ALL_BUFFER_DEBUG=1` — 자세한 버퍼 디버깅.
- `SIMA_GST_RUN_INSERT_BOUNDARIES=1` — run() 중에 경계를 삽입합니다.
- `SIMA_GST_VALIDATE_INSERT_BOUNDARIES=1` — validate() 중에 경계선을 삽입합니다.

## 디스패처 / 런타임

- `SIMA_DISPATCHER_TRACE=1` — 디스패처 단계를 추적합니다.
- `SIMA_DISPATCHER_AUTO_RECOVER=0/1` — 자동 복구 디스패처 (기본값: true).
- `SIMA_ASYNC_TPUT_DIAG=1` — 비동기 처리량 진단.
- `SIMA_ASYNC_WARMUP=<n>` — 비동기 워밍업 프레임.
- `SIMA_PERF_POWER=1` — 성능 테스트 시 SOM PMIC 전원 레일 전력 소비량 측정 기능을 활성화합니다.
- `SIMA_PERF_POWER_INTERVAL_MS=<ms>` — 전력 샘플링 간격(기본값: 100).
- `SIMA_PULL_TIMEOUT_DIAG=0/1` — 풀 요청 시간 초과에 대한 보고서 (기본값: true).
- `SIMA_STAGE_DEBUG=1` — 스테이지 실행 디버그 로그.

## 입력 스트림 / 샘플 디버깅

- `SIMA_INPUTSTREAM_DEBUG=1` — 자세한 InputStream 로그.
- `SIMA_INPUTSTREAM_WARN=1` — InputStream 이벤트에 대한 경고.
- `SIMA_INPUTSTREAM_POLL_MS=<ms>` — InputStream 폴링 간격(기본값: 50ms).
- `SIMA_INPUTSTREAM_DOT_ON_TIMEOUT=1` — 시간 초과 시 DOT 데이터를 덤프합니다.
- `SIMA_INPUTSTREAM_META_DEBUG=1` — GstSimaMeta 세부 정보를 기록합니다.
- `SIMA_INPUTSTREAM_ALLOC_DEBUG=1` — 할당 디버그.
- `SIMA_INPUTSTREAM_PUSH_TIMING=1` — 푸시 타이밍 로그.
- `SIMA_INPUTSTREAM_PREFLIGHT_RUN=1` — InputStream에 대한 사전 실행.
- `SIMA_SAMPLE_DEBUG=1` — 샘플 변환 로그.
- `SIMA_SAMPLE_BYTES=1` — 로그 샘플 바이트 크기를 기록합니다.
- `SIMA_SAMPLE_FORCE_BUNDLE=1` — 디버깅을 위한 강제 번들 출력.
- `SIMA_NEAT_CAPS_TRACE=1` — 텐서 캡 유도 과정 추적.

## 전처리 / 감지 / 배선

- `SIMA_PREPROC_DEBUG_CONFIG=1` — 전처리 구성 연결 정보를 출력합니다.
- `SIMA_KEEP_DETESS_CONFIG=1` — detess 구성 출력값을 유지합니다.
- `SIMA_DETESS_ASSERT_ON_ZERO=1` — 0이 아닌 detess 출력에 대한 어설션.
- `SIMA_CLAMP_DETESS_NUM_BUFFERS=1` — 클램프 디텍스 버퍼 수.
- `SIMA_DISABLE_SYNC_NUMBUFFERS_CVU_MLA=1` — 동기화된 버퍼 수 제한 기능을 비활성화합니다.

## 모델(기존 환경 변수 이름 유지)
- `SIMA_MLA_NEXT_CPU=<domain>` — MLA의 next_cpu를 재정의합니다.
- `SIMA_MPK_EXTRACT_ROOT=<dir>` — 모델 아카이브를 로드하기 위한 기본 디렉터리입니다. 절대 경로로 확인됩니다.
  경로는 프로세스당 한 번만 사용되므로 추출된 JSON으로 다시 작성된 경로는 작업 디렉터리에 의존하지 않습니다. 권한: 쓰기 권한이 없으면 대체 경로를 사용하는 대신 로드가 실패합니다. 설정되지 않은 경우 기본 경로는 마운트된 NVMe 파일 시스템의 첫 번째 쓰기 가능한 후보, `/data`, `TMPDIR` 또는 작업 디렉터리가 됩니다. NVMe 후보는 데이터 마운트의 쓰기 가능한 `/dev/nvme*` 블록 장치여야 합니다. 루트, `/boot`, `/efi` 및 기타 시스템 마운트는 vfat/ISO 파일 시스템과 마찬가지로 제외됩니다. 이러한 검사는 NVMe 검색에 적용되며, `/data`, `TMPDIR` 및 작업 디렉터리 대체 경로는 일반적인 파일 시스템 위치를 유지합니다. NVMe는 용량, 예측 가능한 위치, eMMC 쓰기 방지를 위해 선호됩니다. 이는 디코딩 속도를 향상시키는 것이 아닙니다. 이 변수는 출력이 기록될 위치를 선택하며, 디코딩은 CPU에 의해 제한됩니다.

선택은 여유 공간이 아닌 쓰기 가능성에 따라 이루어집니다. `.tar.gz`는 압축 해제된 크기를 어느 방향으로든 제한하지 않으므로 디코딩하기 전에 알 수 있는 용량 요구 사항이 없습니다. 공간은 압축 해제하는 동안 각 청크별로 적용되고, 추출하기 전에 매니페스트에서 다시 적용됩니다. 따라서 적합한 NVMe는 조건 없이 사용되며, 공간이 없는 경우 `output_storage_unavailable` 오류와 함께 로드가 실패하고 eMMC로 대체되지 않습니다. 해당 파일 시스템의 여유 공간이 있거나 `SIMA_MPK_EXTRACT_ROOT`가 다른 위치를 가리키면 문제가 해결됩니다.
- `SIMA_MPK_CLEANUP_EXTRACTED=0/1` — 정상 종료 시 프로세스별로 추출된 모델 아카이브 데이터를 삭제합니다(기본값: `1`).
  정리 기능이 활성화되면 각 프로세스는 자체 `proc_<pid>` 루트 디렉터리에 압축을 풀고 종료 시 해당 디렉터리를 삭제합니다. 정리 기능이 비활성화되면 해당 프로세스 루트 디렉터리가 유지되어 검사할 수 있으며, 오래된 루트 디렉터리 정리에서 제외됩니다. 다른 프로세스에서 자동으로 검색하거나 재사용되지 않으므로 더 이상 필요하지 않을 때 수동으로 삭제해야 합니다.

`Model`은 `etc`, `lib` 및 `share`를 포함하는 이미 구성된 패키지 루트 디렉터리를 받을 수도 있습니다. 해당 디렉터리는 압축 해제 또는 복사 없이 그대로 사용되며, 호출자가 해당 디렉터리의 수명을 관리하고 모델이 사용되는 동안 변경되지 않도록 해야 합니다. `tar -xzf`로 생성된 단순 디렉터리는 구성된 패키지가 아니므로 직접 사용할 수 없습니다.
- `SIMA_MPK_EXTRACT_GC_STALE_PROC=0/1` — 오래되고 쓸모없는 항목을 제거합니다.`proc_*` 시작 시 추출 루트 (기본값) `1`).
- `SIMA_MODEL_TAR=<path>` — 예제/테스트에서 사용되는 기본 모델 패키지 경로입니다.
  모델별 재정의(`SIMA_RESNET50_TAR`, `SIMA_YOLO_TAR` 등)는 여전히 우선 적용됩니다.
- `SIMA_MPK_EXTRACT_MIN_FREE_BYTES=<bytes>` — 스테이징 중에 확보해 두는 최소 여유 공간
  모델 아카이브 추출 중(기본값 16MiB).
- `TMPDIR=<dir>` — 위에서 언급한 기본 구성 요소의 후반 단계 후보로만 고려되며, 직접적으로 사용됩니다.
  자체 베이스를 선택하지 않는 호출자가 스테이징합니다. 이제 모델 로드는 독립적으로 여기에 스테이징되지 않습니다. 대신 압축 해제된 스냅샷과 추출된 패키지가 선택된 베이스를 공유합니다. 모든 로드(메타데이터만 검사하는 경우 포함)는 `.tar.gz` 파일을 한 번만 압축 해제하여 개인 디렉터리에 저장합니다. 추출하는 동안 해당 파일 시스템에는 압축 해제된 스냅샷, 패키지 및 `SIMA_MPK_EXTRACT_MIN_FREE_BYTES`에 대한 공간이 필요합니다. 150MB 참조 패키지의 스냅샷 크기는 약 354MB입니다. 로컬 파일 시스템을 사용하십시오. 사용 가능한 공간을 읽을 수 없는 경우 로드가 실패합니다. 네트워크 마운트가 연결이 끊어지면 이 문제가 발생할 수 있습니다. 로드가 성공하거나 실패할 때 스테이징 디렉터리가 제거됩니다. 예기치 않은 종료 또는 전원 손실이 발생하면 디렉터리가 남아 있을 수 있습니다.

`SIMA_MPK_EXTRACT_MIN_FREE_BYTES`에서 설정한 여유 공간 예약은 최선을 다하는 방식입니다. 압축 해제하는 동안 파일 시스템에서 보고하는 여유 공간을 기준으로 각 청크에 대해 확인하므로 관련 없는 동시 쓰기 작업으로 인해 파일 시스템의 공간이 부족해질 수 있습니다. 이 경우 로드가 실패하고, 쓰여진 바이트 수와 사용 중인 경로가 표시됩니다.

## RTSP / H264

- `SIMA_H264_SDP_DUMP=<path>` — H264 SDP를 파일에 저장합니다.
- `SIMA_H264_SPS_FIXUP_STREAM=<path>` — 스트림 내의 SPS를 수정합니다.

## 테스트 / 내부 훅

- `SIMA_TENSOR_MAPFAIL_DEBUG=1` — 텐서 매핑 실패를 기록합니다.
