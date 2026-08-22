# 016 처리량 및 큐 깊이 조정

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 15-20 minutes |
| Model | None |
| Labels | performance, tuning, async, queues |

## Concept

부하 상태에서 동작을 제어하는 비동기 파이프라인 설정을 조정합니다. 큐 깊이와 오버플로 정책을 조정한 다음, 실제로 어떤 일이 일어났는지 측정합니다.

## Walkthrough

성능 튜닝은 정확성 기준이 안정화된 후에만 효과가 있습니다. 이 장에서는 이를 전제로 하고, 비동기 파이프라인이 작업이 처리 속도보다 빠르게 도착할 때 어떻게 작동하는지를 결정하는 설정을 살펴봅니다. 큐 깊이를 설정하고, 해당 큐가 가득 찰 때 어떤 일이 발생하는지 선택하고, 결정적인 양의 프레임을 차단하지 않고 푸시하고, 결과를 추출하고, 어떤 프레임이 삭제되었는지, 각 프레임의 처리 시간이 얼마나 걸렸는지 알려주는 측정 보고서를 읽습니다.

결국에는 백프레셔 하에서 비동기 실행을 측정하기 위한 작동 가능한 프레임워크를 갖게 됩니다. 여기에는 큐에 추가된 횟수, 삭제된 횟수, 추출된 출력, 평균 대기 시간, 푸시 비용이 포함됩니다. 동일한 루프는 [실제 적용](#in-practice)의 휴리스틱을 기반으로 실제 파이프라인을 튜닝하는 데 사용됩니다.

### 실행 옵션 구성 {#step-configure-run-options}

`RunOptions`는 로드 하에서 비동기 동작이 결정되는 곳입니다. `queue_depth` (런타임이 수락하는 미처리 샘플의 수), `overflow_policy` (해당 큐가 가득 찰 때 발생하는 일 — `Block`, `KeepLatest`, 또는 `DropIncoming`), `output_memory = Owned` (반환된 텐서가 자체 데이터를 소유하므로 추출 후에도 유지됨)를 설정합니다. 그런 다음 `build()`를 사용하여 그래프를 `Async` 모드로 빌드합니다. 이렇게 하면 독립적인 프로듀서 및 컨슈머 측면을 갖춘 실행이 가능합니다.

**C++:** 오버플로 정책은 `--drop`에서 `simaai::neat::OverflowPolicy::{Block,KeepLatest,DropIncoming}`로 파싱됩니다. `graph.build(input, opt)`는 실행 핸들을 반환합니다.

**Python:** 정책은 `getattr(pyneat.OverflowPolicy, ...)`를 사용하여 확인됩니다. `graph.build([tensor], opt)`는 실행 핸들을 반환합니다.

### 워크로드를 푸시하고 추출 {#step-push-workload}

여기에서 큐 정책이 적용됩니다. `try_push(...)`를 좁은 루프에서 호출합니다. 이는 샘플이 수락되었는지 여부를 반환하는 비차단 푸시이므로 `DropIncoming`/`KeepLatest` 하에서 큐가 가득 찬 경우 거부된 푸시로 나타나므로 중단되는 대신 진행됩니다. 버스트 후에는 더 이상 입력이 없음을 알리기 위해 `close_input()`를 호출한 다음, 비어 있을 때까지 `pull(...)` 루프를 사용하여 컨슈머 측면을 추출합니다. `try_push`와 `close_input`를 함께 사용하고 추출 루프를 추가하는 것은 표준 비차단 비동기 패턴입니다.

### 측정 보고서 읽기 {#step-read-measurement}

런타임이 종료되면 측정 범위를 중단합니다. 보고서의 `counters` 그룹은 런타임 측면의 숫자(대기열에 추가된 입력, 삭제된 입력, 추출된 출력)를 제공하는 반면, `input`은 평균 푸시 비용 및 입력 재협상과 같은 푸시 측면의 숫자를 제공합니다. 이 두 가지를 함께 보면 대기열 깊이와 오버플로 정책이 의도한 대로 작동했는지 확인할 수 있습니다. 즉, 프레임이 삭제되었는지, 지연 시간이 증가했는지, 푸시 경로가 효율적인지 확인할 수 있습니다.

## Run

이 장에서는 모델 아카이브가 필요하지 않습니다. **Neat 설치 루트**(`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행하고, **소스에서 빌드** 명령을 **리포지토리 루트**에서 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/016_tune_throughput_and_queues/tune_throughput_and_queues.py \
  --iters 32 --queue 4 --drop block
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_016_tune_throughput_and_queues \
  --iters 32 --queue 4 --drop block
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_016_tune_throughput_and_queues
./build/tutorials-standalone/tutorial_016_tune_throughput_and_queues \
  --iters 32 --queue 4 --drop block
```

예상 출력(정확한 개수 및 시간은 호스트 및 정책에 따라 다름):

```text
inputs_enqueued=32
inputs_dropped=0
outputs_pulled=32
avg_latency_ms=0.42
avg_push_us=18.0
renegotiations=0
[OK] 016_tune_throughput_and_queues
```

(Python 빌드는 마지막 `[OK]` 줄 없이 동일한 키를 출력합니다.)

사용자 지정 `CMakeLists.txt`를 사용하여 이 장의 C++ 소스를 자체 프로젝트에 통합하려면(추가 폴더가 필요하지 않음) 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

큐 크기 조정, 삭제 정책, 사전 설정 및 출력 수명 주기 안전에 대한 실용적인 지침입니다.

### 큐 크기 조정(`queue_depth`)

휴리스틱:
- 지연 시간이 짧은 파이프라인의 경우 `queue_depth = 4–16`으로 시작합니다.
- 프로듀서가 버스팅하거나 다운스트림 요소의 지연 시간이 가변적인 경우(디코딩/MLA/후처리) 큐를 늘립니다.
- 가장 최신의 프레임이 필요한 경우(예: 실시간 카메라 미리보기) 큐를 작게 유지합니다.

### 오버플로 정책(`RunOptions::overflow_policy`)

- `Block`: 정확성을 위해 가장 안전합니다. 큐가 가득 차면 프로듀서가 대기합니다.
- `DropIncoming`: 큐에 있는 작업을 유지하고, 큐가 가득 차면 들어오는 샘플을 삭제합니다.
- `KeepLatest`: 가장 최신의 프레임을 선호하고, 가장 오래된 큐에 있는 샘플을 삭제합니다.

실시간 피드의 경우 `KeepLatest`를 사용하면 일반적으로 엔드 투 엔드 지연 시간이 가장 낮습니다.

### 사전 설정 및 재협상

`RunOptions::preset`을 사용하여 지연 시간/안전성 간의 균형을 제어합니다.
- `Realtime`: 지연 시간이 가장 짧고, 최신성을 유지하기 위한 공격적인 동작을 수행합니다.
- `Balanced`: 가능한 경우 제로 복사를 시작하고, 시작 프로브 검사를 실행하며, 안정성이 필요한 경우 복사 모드로 되돌아갑니다.
- `Reliable`: 보수적인 동작과 안정적인 출력 소유권을 제공합니다.

동적 입력의 경우 입력 모양 재협상이 자동으로 수행됩니다(위의 `renegotiations` 카운터는 재협상이 얼마나 자주 발생했는지 보고합니다).

### 출력 수명 주기(`output_memory`)

- `output_memory = Owned`: 반환된 `Tensor`가 자체 데이터를 소유합니다.
- `output_memory = ZeroCopy`: 텐서가 풀 후 재사용되는 런타임 버퍼를 참조할 수 있습니다.
- `output_memory = Auto`: 런타임이 먼저 제로 복사를 선택하고, 안정성이 필요한 경우 소유 모드로 되돌아갑니다.

현재 단계 이후에도 텐서 데이터를 유지해야 하는 경우 `clone()` 또는 `cpu().contiguous()`를 호출합니다.

### 버퍼 풀 안전

- `RunAdvancedOptions::max_input_bytes`는 입력 버퍼 할당에 대한 엄격한 상한을 설정합니다.
- 더 큰 버퍼가 필요한 경우 런타임은 명시적인 오류와 함께 빠르게 실패합니다.

입력 크기가 변경될 때 무한정 할당되는 것을 방지하기 위해 장시간 실행되는 프로세스를 보호하는 데 사용합니다.

## 소스 파일
- C++: `tutorials/016_tune_throughput_and_queues/tune_throughput_and_queues.cpp`
- Python: `tutorials/016_tune_throughput_and_queues/tune_throughput_and_queues.py`
