# 012 파이프라인 진단 및 프로파일링

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Intermediate |
| Estimated Read Time | <10 minutes |
| Model | None |
| Labels | diagnostics, debugging, observability |

## Concept

세 가지 검사 — `graph.validate()` (그래프 유효성 검사), 하나의 측정된 `run.run()` 실행, 그리고 `MeasureReport` 진단 — 를 통해 파이프라인을 우선적으로 검사하여, 파이프라인이 올바르게 연결되었는지, 그리고 성능은 어떤지 확인합니다. 이를 통해 심층적인 디버깅을 수행하기 전에 문제를 파악할 수 있습니다.

## Walkthrough

파이프라인이 제대로 작동하지 않을 때, 즉시 요소 수준 디버깅을 시작하려는 유혹을 느낄 수 있습니다. 이 장에서는 더 쉽고 빠른 방법인 반복 가능한 문제 해결 단계를 통해 세 가지 질문에 순서대로 답하는 방법을 설명합니다. *그래프 계약이 유효한가요? 한 번의 실행이 성공했나요? 런타임 진단 결과는 무엇인가요?* 이 방법을 사용하면 몇 초 안에 대부분의 잘못된 구성 문제를 해결할 수 있으며, 문제가 더 심각해져 몇 시간이나 걸리는 디버깅 세션을 진행하기 전에 문제를 해결할 수 있습니다. 또한 이 방법은 004장에서 이미 알고 있는 동일한 최소 입력 → 출력 그래프에서 작동합니다.

이 장을 마치면 그래프 계약을 검증하고, 단일 측정 프레임을 실행하고, 파이프라인이 정상적으로 작동하는지 알려주는 측정 보고서를 출력할 수 있습니다.

### 계약 유효성 검사 {#step-validate-graph}

`validate()`는 `build()` 전에 실행되는 계약 수준 검사입니다. 이 검사는 노드 순서, 제한, 백엔드 파싱 경로를 테스트하며, 데이터를 스트리밍하지 않고 표준 `error_code`를 포함하는 보고서를 반환합니다. 빈 또는 `ok` 코드는 그래프의 구조가 유효함을 의미하며, 다른 모든 코드는 오류를 분류하여 문제의 원인을 파악하는 데 도움이 됩니다(아래 오류 분류 참조). 먼저 이 검사를 실행하면 빌드할 수 없는 그래프에서 런타임 동작을 디버깅하는 데 시간을 낭비하지 않도록 할 수 있습니다.

### 단일 측정 프레임 실행 {#step-run-with-measurement}

다음으로, `start_measurement()` 창 내에서 단일 결정적 프레임을 빌드하고 실행합니다. `output_memory = Owned`는 소유된 출력 버퍼를 요청하므로 호출 후에도 결과가 유효하게 유지됩니다. 한 번의 프레임만으로 충분합니다. 성공하면 파이프라인이 정상적으로 작동하는 것이고, 예외가 발생하면 예외에 구조화된 보고서가 포함되어 `validate()`와 동일한 방식으로 분류할 수 있습니다.

### 런타임 진단 결과 확인 {#step-read-diagnostics}

한 번의 실행 결과를 바탕으로 `MeasureReport`는 파이프라인의 상태를 요약합니다. 여기에는 카운터(`inputs_enqueued`, `outputs_pulled`, 삭제된 데이터), 전체 지연 시간, 노드 메트릭, 플러그인/커널 타이밍, 엣지 타이밍 및 선택적 전력 정보가 포함됩니다. `MeasureReport::to_text()`는 [실제 적용](#in-practice)에서 설명하는 프로브 및 DOT 그래프를 사용하기 전에 캡처하는 기본 보고서입니다.

## Run

실행하면 유효성 검사 코드와 측정 보고서가 표준 출력에 출력되는 것을 확인할 수 있습니다. **Neat 설치 루트**(`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행하고, **소스에서 빌드** 명령은 **리포지토리 루트**에서 실행합니다. 이 장에서는 모델 아카이브가 필요하지 않습니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/012_diagnose_a_pipeline/diagnose_a_pipeline.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_012_diagnose_a_pipeline
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_012_diagnose_a_pipeline
./build/tutorials-standalone/tutorial_012_diagnose_a_pipeline
```

예상 출력(카운터 값과 요약 문자열은 실행마다 다름):

```text
validate.error_code=
measure.inputs_enqueued=1 outputs_pulled=1
measure.text_size=...
[OK] 012_diagnose_a_pipeline
```

(Python 빌드는 `validate_error_code=`, `inputs_enqueued=... outputs_pulled=...` 및 `measure_text_size=...`를 출력합니다.) 이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt`를 사용하여 자신의 프로젝트에 통합하려면 (추가 폴더는 필요하지 않음) 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

구조화된 진단, 오류 분류, 디버깅 옵션, 그리고 `validate()` / `start_measurement()` / `MeasureReport`가 문제를 지적할 때 사용하는 플러그인 실패 워크플로.

### 그래프 보고서

`GraphReport`는 구조화된 진단을 캡처합니다.
- 파이프라인 문자열(재현용)
- 표준 `error_code`(기계 분류)
- `repro_note`(사람이 읽을 수 있는 요약 + 힌트)
- 노드 보고서 및 소유 요소 이름
- 버스 메시지 및 오류 세부 정보
- 선택적 흐름/타이밍 카운터

오류가 발생하면 `NeatError`는 로그하거나 직렬화할 수 있는 `GraphReport`를 포함합니다.

### 오류 분류

프레임워크 오류는 안정적인 코드 패밀리를 사용합니다.

| 오류 코드 | 의미 | 일반적인 해결 방법 |
|---|---|---|
| `misconfig.pipeline_shape` | 노드 순서/모양 계약 위반 | 푸시 파이프라인의 경우 `Input()`을 먼저, 풀 파이프라인의 경우 `Output()`을 마지막에 배치하여 확인 |
| `misconfig.caps` | 프레임워크 캡스 오버라이드 또는 인접 노드 계약 불일치 | `caps_override` 및 선언된 노드 계약을 일치시킴 |
| `misconfig.media_caps` | 런타임 GStreamer 미디어 협상 불일치 | 형식, 해상도 및 프레임 속도를 일치시키거나 변환을 삽입 |
| `misconfig.input_shape` | 입력 텐서/프레임/샘플 모양/레이아웃 불일치 | 너비/높이/깊이, 레이아웃, dtype, 스토리지를 확인 |
| `build.plugin_missing` | 필요한 GStreamer 요소 또는 코덱을 사용할 수 없음 | 설치/교체하고 `gst-inspect-1.0`으로 확인 |
| `build.property_invalid` | GStreamer 속성 이름 또는 값이 유효하지 않음 | `gst-inspect-1.0 <element>`로 확인 |
| `build.pipeline_syntax` | 사용자 지정 GStreamer 조각의 구문이 유효하지 않음 | 수정하고 `gst-launch-1.0`으로 확인 |
| `runtime.pull` | 더 구체적인 원인 없이 풀 작업 실패 | 첨부된 보고서와 첫 번째 상위 오류를 검사 |
| `io.parse` | 저장된 그래프 JSON 구문 분석/스키마 실패 | JSON 및 필수 노드 필드를 확인 |
| `io.open` | 그래프 저장/로드 파일 열기/읽기/쓰기 실패 | 경로 존재 여부, 권한 및 저장 상태를 확인 |

`PullError.code`는 동일한 분류(예외 경로만 아님)를 사용합니다.
이것은 간단한 분류 목록입니다. 이전의 대략적인 런타임 및 빌드 코드에서 마이그레이션한 내용을 포함하여 [완전한 오류 코드 카탈로그](/reference/error-codes)를 참조하십시오.

### 프로그래밍 방식 처리

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

### 디버깅 설정 (환경)

주요 환경 변수(자세한 내용은 [아키텍처](/develop-apps/contribute/architecture) 참조):
- `SIMA_GST_DOT_DIR`: 실패 시 DOT 그래프 작성
- `SIMA_GST_BOUNDARY_PROBES`: 경계 흐름 카운터
- `SIMA_GST_ELEMENT_TIMINGS`: 요소별 타이밍
- `SIMA_GST_FLOW_DEBUG`: 요소별 흐름 카운터
- `SIMA_GST_ENFORCE_NAMES`: 명명 규칙 적용

짧은 기간 동안 수정된 원본 데이터를 사용합니다. GStreamer 오류에 추가된 컨텍스트, 사용법:

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug` ...가 아닙니다. Neat Library 설정.

### 디버깅 워크플로

1. 먼저 `GraphReport.error_code`를 캡처하고 분류 체계에 따라 오류를 그룹화합니다.
2. 구체적인 상황과 기본 제공 힌트를 위해 `GraphReport.repro_note`를 캡처합니다.
3. `Graph::describe_backend()` 또는 `last_pipeline()`을 사용하여 파이프라인 텍스트를 캡처합니다.
4. `MeasureReport::to_text()` 또는 `NeatError::report()`를 사용하여 구조화된 진단 정보를 캡처합니다.
5. `GraphReport.bus`에서 첫 번째 터미널 `ERROR` 소스와 세부 정보를 확인합니다.
6. 런타임에 문제가 발생하거나 시간 초과가 발생하면 경계/요소 프로브를 활성화하여 흐름 중단 지점을 파악합니다.

권장 지원 패키지:
- `error_code`
- `repro_note`
- 전체 `pipeline_string`
- 처음 3~5개의 터미널 버스 오류(`GraphReport.bus`)
- 실행/유효성 검사 중에 사용된 환경 설정 재정의

### 일반적인 오류 → 해결 방법

| 증상 | 가능한 원인 | 해결 방법 |
| --- | --- | --- |
| `missing ... plugin` | GStreamer 플러그인을 찾을 수 없음 | 확인 `GST_PLUGIN_PATH`, 실행 `gst-inspect-1.0 <plugin>` |
| `appsink 'mysink' not found` | 연결되지 않은 터미널 `Output()` | 확인하십시오 `Output` 실행/빌드 파이프라인의 마지막 노드입니다. `caps_override is set; renegotiation disabled` | 캡 고정 해제 | 제거 `caps_override` 또는 입력 시 대문자 설정을 고정합니다. `tensor caps change not supported` | 런타임 시 텐서 모양/데이터 유형 변경 | 텐서 모양/데이터 유형을 안정적으로 유지 (재협상 없음) |

### 플러그인 오류 디버깅

플러그인이 실패하면 NEAT는 오류를 발생시킵니다. `NeatError` 누구의 메시지에 다음 내용이 포함되어 있습니까? GStreamer 오류와 구조화된 디버그 문자열입니다. 필드를 사용하여 근본 원인을 빠르게 파악하십시오.

1. **구조화된 필드를 읽습니다.** 다음을 확인하십시오. `debug` 오류 텍스트의 키/값 필드:
   - `node`: 파이프라인에서 오류가 발생한 요소의 이름
   - `config_path`: JSON 구성 파일(해당하는 경우)
   - `model_path`: 모델/패키지 경로(해당하는 경우)
   - `hint`실질적인 해결 방안 제시
   - `detail`: 누락된 키 또는 할당자 상태와 같은 추가 정보

   전체 목록은 [오류 형식 참조](/reference/error_format)를 참조하십시오.
2. **파이프라인 컨텍스트를 확인합니다.** `Graph::last_pipeline()` 또는 오류 보고서에서 파이프라인 문자열을 사용합니다.
   - `node` 이름이 파이프라인에 나타나는지 확인합니다.
   - `config_path`가 존재하고 읽을 수 있는지 확인합니다.
   - 캡 오류의 경우, 오류가 발생한 노드로 연결되는 상위 요소들을 확인합니다.
3. **일반적인 수정 사항을 적용합니다.**
   - **구성 오류**: JSON 구문, 필수 키 및 모든 모델 경로를 확인합니다.
   - **캡 오류**: 파서 요소(예: `h264parse`)를 추가하거나 수정하고, 캡에 `parsed=true`, `stream-format=byte-stream`, `alignment=au`와 같은 필수 필드가 포함되어 있는지 확인합니다.
   - **할당자 오류**: 상위 요소가 필요한 할당자 유형(시스템 대 simaai 메모리/세그먼트)을 사용하는지 확인합니다.
4. 위의 디버그 설정을 사용하여 더 많은 진단 정보를 캡처합니다(`SIMA_GST_DOT_DIR`, `SIMA_GST_FLOW_DEBUG`, `SIMA_GST_ELEMENT_TIMINGS`).

## 소스 파일
- C++: `tutorials/012_diagnose_a_pipeline/diagnose_a_pipeline.cpp`
- Python: `tutorials/012_diagnose_a_pipeline/diagnose_a_pipeline.py`
