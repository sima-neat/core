# 004 첫 번째 그래프 만들기

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | None |
| Labels | graph, build, run, pipeline |

## Concept

수동으로 `Graph`를 구성합니다. 입력 노드, 출력 노드를 만들고, 모델은 사용하지 않은 상태로 하나의 프레임을 파이프라인에 적용합니다. 모델이 추가되기 전에 파이프라인의 기본 요소들을 개별적으로 확인합니다.

## Walkthrough

제1장에서는 세 줄의 코드로 모델을 실행합니다. 이러한 편리함은 모든 의미 있는 Neat 프로그램에서 직접 사용하는 두 부분으로 구성된 라이프사이클을 숨깁니다. 먼저 `Graph`로 파이프라인을 *설명*하고, 그런 다음 해당 설명을 실행 가능한 `Run`으로 *구축*합니다. 이 장에서는 가능한 가장 작은 파이프라인(하나의 입력 노드가 하나의 출력 노드에 연결되고, 그 사이에 모델이 없음)을 구성하고, 단일 프레임을 파이프라인에 통과시켜 해당 라이프사이클을 보여줍니다.

핵심은 개념적입니다. `Graph`는 한 번 구축하고 여러 번 실행하는 *재사용 가능한 정의*이며, 일회성 호출이 아닙니다. 이 장의 끝 부분에서는 그래프를 만들고, 실행 가능한 파이프라인으로 변환하고, 출력 텐서의 랭크를 읽어 프레임이 파이프라인을 통과했음을 증명합니다.

### 입력 {#step-configure-input} 설명

노드를 연결하기 전에 프레임이 어떻게 생겼는지 선언합니다. `InputOptions`는 이러한 계약입니다. 픽셀 `format`, `width`/`height`, 채널 `depth` 및 런타임이 각 버퍼에 타임스탬프를 찍는지 여부를 지정합니다. 이러한 옵션에서 구축된 입력 노드는 들어오는 프레임을 파이프라인에서 예상하는 모양과 비교하여 유효성을 검사합니다.

**C++:** C++는 추가적으로 `is_live = false`를 설정하여 이를 비실시간(파일/텐서) 소스로 표시합니다.

### 그래프 구성 {#step-compose-graph}

이제 구조를 구축합니다. 새 `Graph`는 빈 구성 표면이며, `add()`는 노드를 순서대로 추가합니다. 정확히 두 개의 노드(위에서 구성한 입력 노드와 빈 출력 노드)를 추가합니다. 이것이 전체 토폴로지입니다. 프레임은 입력에서 들어오고 출력에서 나가며, 그 사이에 아무것도 없습니다. 이것이 나중에 모델 또는 전처리 단계가 삽입될 위치입니다.

**C++:** 노드는 `simaai::neat::nodes::Input(...)` 및 `nodes::Output()`에서 가져옵니다.

**Python:** 노드는 `pyneat.nodes.input(...)` 및 `pyneat.nodes.output()`에서 가져옵니다.

### 파이프라인 구축 {#step-build-pipeline}

`build()`는 *설명*에서 *실행 가능*으로 전환하는 단계입니다. 추가된 노드를 구체적인 파이프라인으로 변환하고, 실제 샘플을 사용하여 입력/출력 계약의 유효성을 검사하고, 재사용 가능한 `Run` 핸들을 만듭니다. 대표 프레임을 전달하여 `build()`가 협상된 텐서 모양을 고정할 수 있도록 합니다. 다음 단계에서는 `Run::run(...)`을 사용하여 결정적인 방식으로 한 번에 하나의 호출을 수행합니다.

**C++:** 샘플 프레임은 `cv::Mat`이며, `run_opt.output_memory = Owned`는 런타임에 소유된 출력 버퍼를 반환하도록 요청합니다.

**Python:** 먼저 NumPy 배열에서 `Tensor`를 `Tensor.from_numpy(...)`를 사용하여 구체화한 다음, 이를 사용하여 구축합니다.

### 프레임을 실행하고 결과를 읽기 {#step-run-frame}

`Run`을 확보한 후 `run()`은 하나의 프레임을 파이프라인에 통과시키고 하나의 결과를 동기적으로 가져옵니다. 모델이 없으므로 출력은 입력 계약을 반영합니다. 따라서 텐서의 *랭크*를 읽는 것만으로도 프레임이 전체 과정을 완료했는지 확인할 수 있습니다. 실제 파이프라인에서는 동일한 `run()`/push/pull 표면을 사용하여 추론을 수행합니다.

**C++:** `run()`은 `TensorList`를 반환합니다. `sample.front().shape.size()`를 읽습니다.

**Python:** 텐서 입력을 사용하는 `run()`은 `TensorList`를 반환합니다. `len(outputs[0].shape)`를 읽습니다.

## Run

실행하면 출력 텐서의 순위가 표준 출력에 표시됩니다. **Neat 설치 디렉터리**(`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행하고, **저장소 루트**에서 **소스 코드를 빌드**하는 명령을 실행합니다. 이 장에서는 모델 아카이브가 필요하지 않습니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/004_build_inference_pipeline/build_inference_pipeline.py \
  --width 320 --height 240
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_004_build_inference_pipeline \
  --width 320 --height 240
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_004_build_inference_pipeline
./build/tutorials-standalone/tutorial_004_build_inference_pipeline \
  --width 320 --height 240
```

예상 결과:

```text
tensor_rank=3
[OK] 004_build_inference_pipeline
```

(Python 빌드에서는 `output_rank=...`가 출력됩니다.) 이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt`를 사용하여 자신의 프로젝트에 통합하려면 (별도의 추가 폴더는 필요하지 않음) 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

`build`/`run` 방식, 실행 모드, 푸시/풀 인터페이스, 그리고 `RunOptions`가 단일 동기 호출을 넘어서 어떻게 함께 작동하는지 설명합니다.

### 빌드 vs 실행

- `Graph::build(...)`는 파이프라인을 구성하고 푸시/풀 제어를 위한 `Run` 핸들을 반환합니다.
- `Graph::run(...)`은 간편한 동기 방식입니다. 필요하면 그래프를 빌드하고, 하나의 입력을 푸시하고, 하나의 출력을 풀합니다.

### 동기 vs 비동기

- 간단한 단일 호출의 경우 `Graph::run(...)`을 사용합니다.
- 재사용 가능한 실행기와 명시적인 `push(...)` / `pull(...)` 제어가 필요한 경우 `Graph::build(...)`를 사용합니다. 자세한 내용은 [비동기 추론 실행](/tutorials/run-inference-async)을 참조하십시오.

### 푸시/풀 API

`Run`은 다음을 제공합니다.
- 입력용: `push(...)` / `try_push(...)` (`cv::Mat`, `Tensor`, 또는 `Sample`).
- 출력용: `pull(...)`, `pull_tensor(...)`, `pull_tensor_or_throw(...)`.

출력 메타데이터(타임스탬프, 스트림 ID)가 필요한 경우 `pull()`을 사용하여 `Sample`을 가져옵니다. 텐서 페이로드만 필요한 경우 `pull_tensor()`를 사용합니다.

### RunOptions (간단한 API)

일반적인 설정:
- `preset`: 지연 시간/안전성 프로필(`Realtime`, `Balanced`, `Reliable`).
- `queue_depth`: 런타임 큐 깊이.
- `overflow_policy`: 큐 오버플로 동작(`Block`, `KeepLatest`, `DropIncoming`).
- `output_memory`: 출력 소유권 정책(`Auto`, `ZeroCopy`, `Owned`).
- `on_input_drop`: 삭제된 입력 이벤트에 대한 콜백 훅.

큐 깊이, 오버플로 및 부하 상태에서의 측정에 대해서는 [처리량 및 큐 깊이 조정](/tutorials/tune-throughput-and-queues)을 참조하십시오.

### RunAdvancedOptions (고급 API)

고급 설정은 `RunOptions::advanced`를 통해 선택적으로 사용할 수 있습니다.
- `advanced.max_input_bytes`: 입력 버퍼 증가 제한.
- `advanced.copy_input`: 방어적인 입력 복사 강제.

`Run::start_measurement()`를 사용하여 단일 측정 창에서 지연 시간, 처리량, 입력 카운터, 플러그인/에지 타이밍 및 선택적 보드 PMIC 전력 텔레메트리를 검사합니다.

보드 전력을 포함하려면 코드에서 활성화하고(환경 변수는 필요하지 않음) 측정 보고서에서 읽습니다.

```cpp
simaai::neat::RunOptions run_opt;
run_opt.enable_board_power(); // default 100 ms sampling, auto-detects built-in profile
auto run = graph.build(inputs, run_opt);
auto scope = run.start_measurement();
run.push(inputs);
(void)run.pull_tensors(5000);
auto report = scope.stop();
```

```python
run_opt = neat.RunOptions()
run_opt.enable_board_power()  # default 100 ms sampling, auto-detects built-in profile
run = graph.build(tensor, run_opt)
scope = run.start_measurement()
run.push(tensor)
_ = run.pull_tensors(5000)
report = scope.stop()
```

`Model::build(run_opt)`, `Model::build(route_opt, run_opt)` 및 `Graph::build(run_opt)`는 동일한 런타임 옵션을 기본 `Run`에 전달하므로, 파이프라인별로 중복되는 레일 샘플링 대신 그래프 수준의 보드 전력 모니터 하나를 사용합니다. 특정 기본 프로필을 강제로 적용해야 하는 경우, 보드별 헬퍼를 계속 사용할 수 있습니다: `enable_modalix_som_power()`, `enable_modalix_dvt_power()`.

## 소스 파일
- C++: `tutorials/004_build_inference_pipeline/build_inference_pipeline.cpp`
- Python: `tutorials/004_build_inference_pipeline/build_inference_pipeline.py`
