---
title: "SIMA 플러그인 JSON 진실성 매핑"
description: "모델 파이프라인 SIMA 단계에 대한 고정된 JSON 필드 사용 맵"
sidebar_position: 2
slug: /develop-apps/contribute/sima_plugin_json_truth_map
---

# SIMA 플러그인 JSON 진실성 맵 (고정)

_최종 업데이트: 2026-02-17_

본 문서는 모델 파이프라인의 SIMA 단계에서 사용되는 JSON 필드를 고정하여 제거가 제어되고 테스트 가능하도록 합니다.

## 1. 고정 범위 및 플러그인 매트릭스

### 1.1 범위 내 (모델 파이프라인)

- `simaaiprocesscvu`는 다음과 같이 사용됨:
  - 전처리 단계(`kernel=preproc`)
  - 양자화/테셀레이션 단계(`kernel=quanttess`)
  - 디테셀레이션/디양자화를 위한 후처리 단계 래퍼(`kernel=detessdequant`는 모델 시퀀스에 있으며, 백엔드 요소는 여전히 `simaaiprocesscvu`이며, `src/nodes/sima/DetessDequant.cpp`에 있음)
- `simaaiprocessmla`
- `simaaiboxdecode` (일반 박스 디코더)
- `tmp/gst/*`의 디테셀레이션/디양자화/테셀레이션 페이로드 단계:
  - `detessdequant`
  - `detessellate`
  - `quantize`
  - `slicedequant`

### 1.2 범위 외

본 진실성 맵의 범위에서 제외되는 일반 CVU 앱 플러그인 및 사용자 정의 그래프 유틸리티(다음 포함):

- `overlay`, `genericrender`, `argmax`, `nms*`, `groupkeypoints`, `distancecalculation`, `cv_process`, `cvresize`, `fastbev*`, `PyGast-plugins/*`, 사용 중단된 플러그인 및 사용자 정의 앱/테스트 스캐폴딩.

### 1.3 적용되는 소스 트리

정적 추출은 요청된 두 트리에 모두 적용됨:

- `tmp/gst_plugins_source/gst/*`
- `tmp/gst/*`

미러 확인:

- 두 트리 모두 동일: `genericboxdecode`, `detessdequant`, `detessellate`, `quantize`, `slicedequant`
- 다름: `processcvu`, `processmla`

### 1.4 플러그인 매트릭스

| 플러그인/스테이지 | 현재 필요한 JSON 키 (현재 코드 경로) | 추론 가능한 키 | 런타임 속성 (정적 JSON이 아니어야 함) | MLA 전용 키 |
|---|---|---|---|---|
| `simaaiprocesscvu` (전처리/양자화/후처리 래퍼) | 우선 추론: 사용 가능한 경우 `ConfigManager::getBuffers()`에서 와이어링이 발생; JSON `input_buffers`/`output_memory_order`는 보조적인 용도로만 사용됨. | `input_width`/`input_height`는 그래프 200/202에 대해 캡스/런타임에서 가져올 수 있음; 와이어링 배열은 CM 메타데이터에서 합성될 수 있음. | 런타임 차원은 프레임별로 재협상됨; 프레임워크 빌드는 더 이상 스테이지별 JSON 와이어링 필드를 다시 쓰지 않음. | 양자화/테셀레이션 및 후처리 경로는 간접적으로 MLA 텐서 모양 필드(`input_depth`, `slice_*`)를 사용함. |
| `simaaiprocessmla` | `simaai__params`, `model_path`, `batch_size`; `outputs[*]`가 선호되지만 출력 모양 필드에서 세그먼트 크기를 추론할 수 있는 경우 더 이상 필수로 요구되지 않음. | 출력 세그먼트 크기는 `output_*`/`slice_*` + 데이터 유형에서 추론할 수 있음. | `input_segment_name`은 선택적인 런타임 와이어링 지원; 모델 경로는 패키지에서 파생될 수 있음. | `outputs`, `data_type`, `output_*`, `slice_*`, 양자화 매개변수 |
| `simaaiboxdecode` | 백엔드 구성 로더에서 실제로 필요: `buffers.output.size`, `memory.next_cpu`, `system.out_buf_queue`; 클래스 수 확인은 현재 구현 버전에 따라 달라짐. | `num_classes`는 `input_depth`/`slice_depth` + `num_in_tensor` + `decode_type` (새 소스 로직)에서 추론할 수 있음. | `buffers.input[*].name`은 상위 레벨에서 재구성됨; 임계값/topk는 런타임 노브인 경우가 많음. | `input_*`, `slice_*`, `data_type`, `num_in_tensor` |
| `detessdequant` (레거시 독립형 GST 요소) | `simaai__params` 및 파서 필드: `orig_img_width`, `orig_img_height`, `frame_width`, `frame_height`, `num_in_tensor`, `next_cpu`, `no_of_outbuf`, `out_sz`, `input_*`, `slice_*`, `q_scale`, `q_zp` | 플러그인에는 없음; 상위 레벨의 모양 추론은 `StageConfig`에 존재함. | 상위 래퍼 흐름에서 버퍼 이름 지정/CPU 라우팅은 런타임에 수행됨. | `input_*`, `slice_*`, `q_scale`, `q_zp`, `num_in_tensor` |
| `detessellate` 페이로드 (`tmp/gst/detessellate`) | `de_tess.*` 또는 루트/정적 계약 동등물(`input_*`, `slice_*`/`output_*`)을 허용; `buffers.input[0].offset`은 선택 사항(기본값은 0) | 텐서 수와 차원은 매니페스트 스테이지 정적 필드에서 합성할 수 있음. | 입력 이름/경로는 정적이 아닌 런타임에 와이어링되어야 함. | 모양/슬라이스 계약 |
| `quantize` 페이로드 (`tmp/gst/quantize`) | `quant_scale`, `zero_point` (JSON 보조) | 입력 요소 수는 들어오는 버퍼 크기에서 추론됨. | 이제 먼저 상위 레벨의 런타임 메타데이터(`q_scale`/`q_zp`)를 사용하고, 그 다음 JSON 보조를 사용함. | 해당 없음 (일반 양자화) |
| `slicedequant` 페이로드 (`tmp/gst/slicedequant`) | 모양 보조를 위해 섹션(`slice_dequant`/`simaai__params`/root)을 읽음; 양자화 JSON은 보조적인 용도로만 사용됨. | 먼저 런타임 메타데이터에서 양자화; 차원은 매니페스트 정적 계약에서 합성할 수 있음. | 런타임 메타데이터(`q_scale`/`q_zp`)가 선호되는 전송 방식임. | MLA 출력 모양/양자화 계약 |

### 1.5 매니페스트 컨텍스트 전송 (현재)

- 파이프라인 컨텍스트 유형: `sima.model.manifest.v1`
- ABI 호환 플러그인 액세스: `manifest_accessor_v1` (위치: `include/gst/SimaPluginStaticManifestAbi.h`)
- 스테이지 조회 키:
  - `element_name` (기본값)
  - `logical_stage_id` (설정된 경우 `stage-id` 또는 `stage_id` 파이프라인 속성에서 가져옴)
- 기존 `manifest_json` 문자열은 이전 버전과의 호환성을 위해 유지됩니다.
- 프레임워크 노드/모델 조각 빌더는 이제 SIMA 모델 경로 플러그인에 대해 `stage-id=<element-name>`을 출력하므로 추가적인 이름 변환이 적용되더라도 논리적 조회가 결정적으로 수행됩니다.

## 2. 필수 키 진실성 매핑

### 2.1 정적 추출 방법

명시적 액세스 지점에서 추출:

- `json["..."]`
- `contains("...")`
- 파서 헬퍼 (`parser_get_int`, `parser_get_double_array` 등)

주요 증거 위치:

- `tmp/gst/processcvu/gstsimaaiprocesscvu.cpp:1667`
- `tmp/gst/processmla/gstsimaaiprocessmla.cpp:579`
- `tmp/gst/genericboxdecode/payload.cpp:61`
- `tmp/gst/detessdequant/gstsimaaidetessdequant.cpp:276`
- `tmp/gst/detessellate/detessellate.cpp:361`
- `tmp/gst/quantize/payload.cpp:124`
- `tmp/gst/slicedequant/payload.cpp:57`

런타임 연결/추론 증거:

- `src/nodes/sima/Preproc.cpp:245`
- `src/nodes/sima/DetessDequant.cpp:238`
- `src/nodes/sima/SimaBoxDecode.cpp:158`
- `src/pipeline/runtime/StageConfig.cpp:296`
- `src/pipeline/runtime/StageConfig.cpp:411`

### 2.2 사용된 동적 오류 주입 방법

등록된 플러그인(`simaaiprocesscvu`, `simaaiprocessmla`, `simaaiboxdecode`, `detessdequant`)의 경우:

- 기준: `gst-launch-1.0 ... num-buffers=0`
- 한 번에 하나의 키를 변경(필드 제거)
- 시작/런타임 오류 동작 및 메시지 기록
- 참고: 동적 결과는 이 호스트에 현재 등록된 런타임 플러그인을 반영합니다.

등록되지 않은 페이로드 스테이지(`detessellate`, `quantize`, `slicedequant`)의 경우:

- 이 런타임에서 직접 사용할 수 있는 GST 요소가 없음(`gst-inspect-1.0`에서 누락된 것으로 보고)
- 따라서 동적 키 제거는 이번 패스에 대해 정적/소스 분류로 제한되었습니다.

### 2.3 범주화된 매핑 (고정)

### `simaaiprocesscvu`

- 필수:
  - CM 버퍼 추론 성공 또는 다음을 포함하는 JSON 폴백:
    - `input_buffers`
    - `output_memory_order`
    - 각 입력 `memories[*].segment_name`
    - 각 입력 `memories[*].graph_input_name`
- 선택적/기본값으로 설정 가능:
  - `graph_name`
  - `input_width`, `input_height` (선택적 JSON 차원)
- 중복/파생:
  - `input_buffers[*].name` (런타임에 연결됨)
  - 매니페스트 컨텍스트의 `sink_pad_tensor_index_map`이 이제 결정적인 다중 입력 매핑을 위해 선호됨
  - 캡/런타임의 사전 처리 차원
- 디버그 전용:
  - `debug` 스타일 필드는 실행에 필요하지 않음

동적 증거:

- CM 추론이 실패하고 JSON 연결이 누락된 경우 -> 시작 시 버스 오류로 실패
- CM 추론이 성공한 경우 -> `input_buffers`/`output_memory_order`을 생략할 수 있음
- 컨텍스트에 모델에서 관리하는 다중 입력 스테이지가 있고 `sink_pad_tensor_index_map`이 누락되었거나 모호한 경우 -> 시작 시 버스 오류로 실패

### `simaaiprocessmla`

- 필수:
  - `simaai__params`
  - `model_path`
  - `batch_size`
  - `outputs[*].name`
  - `outputs[*].size`
  - `batch_sz_model` (단, `batch_size != 1`인 경우)
- 선택/기본값 설정 가능:
  - `input_segment_name`
- 중복/파생:
  - 상위 레이어의 모델 메타데이터에서 출력 차원/유형을 추론할 수 있음
- 디버깅 전용:
  - 해당 사항 없음

동적 증거:

- `model_path` 제거 -> 처리되지 않은 `nlohmann::json` 유형 오류 (`ec=134`)
- `batch_size=2` 및 `batch_sz_model` 제거 -> 처리되지 않은 유형 오류 (`ec=134`)
- `outputs` 제거 -> `num-buffers=0`를 사용하여 시작이 가능하지만, 런타임(`num-buffers=1`)에서 SIGSEGV 스핀 경로가 발생함

### `simaaiboxdecode`

- 필수 (현재 런타임 동작):
  - `buffers.output.size`
  - `memory.next_cpu`
  - `system.out_buf_queue`
- 선택/기본값 설정 가능 (구현 버전에 따라 다름):
  - 최신 소스에서는 `num_classes`를 추론/대체할 수 있지만, 현재 런타임에서는 경고만 표시할 수 있음
  - 현재 런타임에서는 `decode_type`이 유형 불일치 경고로 저하될 수 있음
- 중복/파생:
  - `buffers.input[*].name`은 런타임에 고정됨
  - 알려진 디코딩 계열의 경우 `num_classes`는 텐서 모양(`input_depth`/`slice_depth`)에서 파생될 수 있음
- 디버깅 전용:
  - `system.debug`, `system.dump_data`

동적 증거:

- `memory.next_cpu` 제거 -> 처리되지 않은 유형 오류로 종료(`ec=134`)
- `system.out_buf_queue` 제거 -> 처리되지 않은 유형 오류로 종료(`ec=134`)
- `num_classes` 제거 -> 치명적이지 않은 `JSON type mismatch` 경고(`ec=0`)

### `detessdequant` (레거시 독립형 GST 플러그인)

- 필수:
  - `simaai__params` 객체
  - 파서 키: `orig_img_width`, `orig_img_height`, `frame_width`, `frame_height`,
    `num_in_tensor`, `next_cpu`, `no_of_outbuf`, `out_sz`,
    `input_height`, `input_width`, `input_depth`,
    `slice_height`, `slice_width`, `slice_depth`,
    `q_scale`, `q_zp`
- 선택/기본값 설정 가능:
  - 현재 코드에는 해당 사항 없음
- 중복/파생:
  - 일부 프레임/원본 크기 필드는 메타데이터 수준이며 파생될 수 있음
- 디버깅 전용:
  - `debug`, `dump_data`, `inpath`, `ibufname`, `n_request` 등

동적 증거:

- `simaai__params` 제거 -> SIGSEGV 스핀 경로 (시간 초과)
- `num_in_tensor` 제거 -> SIGSEGV 스핀 경로 (시간 초과)

### `detessellate` 페이로드

- 필수:
  - 해결된 입력/슬라이스 텐서 필드 (`de_tess.*` 또는 루트/정적 계약에서 합성된 키)
- 선택/기본값 설정 가능:
  - `buffers.input[0].offset` (기본값은 `0`)
  - `num_in_tensor` (생략된 경우 벡터 크기에서 파생됨)
- 중복/파생:
  - 모양 벡터는 매니페스트 단계의 정적 텐서에서 파생될 수 있음
- 디버깅 전용:
  - 해당 사항 없음

### `quantize` 페이로드

- 필수:
  - `quant_scale`
  - `zero_point`
- 선택/기본값 설정 가능:
  - 현재 코드에는 해당 사항 없음
- 중복/파생:
  - 텐서 요소 수는 입력 바이트 크기에서 파생됨
- 디버깅 전용:
  - 해당 사항 없음

### `slicedequant` 페이로드

- 반드시 필요:
  - 양자화 해제에 대한 양자화 매개변수(`q_scale`, `q_zp`)는 런타임 메타데이터 또는 대체 구성에서 확인됩니다.
  - 텐서 슬라이스 차원(`input_*`, `output_depth`/`slice_depth`)은 섹션/루트/정적 계약 합성에서 확인됩니다.
- 선택 사항/기본값 사용 가능:
  - 양자화 및 모양 키에 대한 스칼라 대 벡터 인코딩
- 중복/파생:
  - 양자화는 런타임 메타데이터에서 우선 적용되며, JSON은 대체 구성으로만 유지됩니다.
- 디버깅 전용:
  - 해당 없음

## 3. 제어된 제거 게이트 (이 고정된 맵에서)

모든 JSON 필드 제거에는 다음이 포함되어야 합니다.

1. 해당 필드에 대한 이 진실 맵 분류를 업데이트합니다.
2. 실패 주입 테스트 케이스를 추가/업데이트합니다.
   - 시작 실패는 명시적이어야 합니다(버스 오류), 절대 충돌해서는 안 됩니다.
3. 추론 가능/파생된 필드의 경우:
   - 먼저 추론을 구현하고,
   - 기능 제한/플러그인 옵션 대체 기능을 유지하고,
   - 여전히 필요한 경우 JSON을 마지막 대체 구성으로만 유지합니다.
4. MLA 전용 JSON을 최소화합니다.
   - 해결되지 않은 MLA 메타데이터(모양/크기/양자화 매개변수)만 유지합니다.
5. 엄격한 CI 게이트를 활성화 상태로 유지합니다.
   - `unit_sima_plugin_manifest_strict_model_pipeline_test`
   - `unit_sima_plugin_manifest_strict_fallback_test`
   - 그리고 `.github/workflows/vulcan-ci.yml`의 해당 Vulcan CI 테스트 경로.

## 4. 현재 발견된 위험

- 여러 개의 누락된 키 경로가 여전히 처리되지 않은 `nlohmann::json` 예외 또는 버스 오류 대신 SIGSEGV로 인해 중단됩니다.
- `detessdequant` 레거시 경로는 누락된 파서 키로 인해 충돌하기 쉽습니다.
- `slicedequant`는 JSON을 완전히 무시하고 컴파일된 상수를 사용합니다.
- 런타임/소스 불일치는 다시 빌드된 `.so` 파일이 `tmp/gst/*/build`에서 `deps/gst-plugins`로 복사되지 않을 때마다 다시 발생할 수 있습니다.
