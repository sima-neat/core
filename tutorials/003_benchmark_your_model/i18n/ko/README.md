# 003 모델 성능을 평가해 보세요

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5-10 minutes |
| Model | resnet_50 |
| Labels | benchmark, synthetic, latency, throughput, power |

## Concept

컴파일된 모델을 실행하여 결정론적 합성 텐서를 사용하고, `Model::benchmark()`에서 반환된 지연 시간, 처리량, 전력 소비량, 에너지 소비량 등의 주요 지표를 출력합니다.

## Walkthrough

1장과 2장에서는 모델을 한 번 실행하고 비동기적으로 실행하는 방법을 보여주었습니다. 이 장에서는 다음 실질적인 질문에 답합니다. "이 모델은 장치에서 얼마나 빠르게 실행됩니까?" 벤치마크 API는 의도적으로 작게 설계되었습니다. 모델을 로드하고, 측정할 샘플 수를 선택한 다음, `benchmark(...)`를 호출하고 반환된 `BenchmarkReport`를 읽습니다.

벤치마크는 모델의 `input_specs()`를 사용하여 결정적인 합성 입력을 생성합니다. 이는 빠른 모델 성능 테스트와 컴파일된 모델 변형을 비교하는 데 유용하지만 카메라 벤치마크는 아닙니다. 카메라 디코딩, 실제 전처리 가변성, 동적 입력 크기 또는 데이터 종속적 후처리 동작이 포함되지 않습니다.

### 모델 로드 {#step-load-model}

이전 모델 튜토리얼에서 사용한 동일한 컴파일된 `.tar.gz` 아카이브를 사용합니다. 벤치마크는 모델의 선언된 입력 사양에서 합성 텐서를 생성하므로 이미지는 필요하지 않습니다.

**C++:** 아카이브 경로에서 `simaai::neat::Model`을 생성합니다.

**Python:** 아카이브 경로에서 `pyneat.Model`을 생성합니다.

### 벤치마크 실행 {#step-run-benchmark}

`benchmark(samples)`를 호출합니다. API는 모델을 준비하고, `samples`개의 입력으로 단일 실행(single-flight) 지연 시간을 측정한 다음, 별도의 비동기 푸시/풀 패스에서 처리량을 측정하고, 요약을 표준 출력으로 출력한 뒤, 동일한 주요 값을 `BenchmarkReport`로 반환합니다.

처리량 패스는 최소 `samples`개의 입력을 측정하고 30초가 경과할 때까지 실행되므로 전력 및 온도 측정값이 안정적입니다. 이 하한을 변경하려면 `BenchmarkOptions::min_duration_ms`를 설정하고, `samples`개의 입력 후에 중지하는 빠른 스모크 테스트를 하려면 0으로 설정합니다.

BoxDecode로 끝나는 경로를 가진 감지 모델은 `BenchmarkOptions`를 사용할 수도 있습니다. `original_width`, `original_height` 및 `resize_mode`를 설정하여 BoxDecode가 모델 좌표에서 감지를 매핑할 때 사용하는 소스 이미지의 기하학적 구조를 설명합니다. 합성 텐서는 모델의 형태를 유지합니다.

```cpp
simaai::neat::BenchmarkOptions options;
options.num_samples = 100;
options.original_width = 1920;
options.original_height = 1080;
options.resize_mode = simaai::neat::ResizeMode::Letterbox;
auto report = model.benchmark(options);
```

Python은 `pyneat.BenchmarkOptions`를 통해 동일한 필드를 노출합니다. 원래의 두 가지 차원을 모두 설정하거나, 둘 다 생략할 수 있습니다. 생략할 경우, 벤치마크는 해결된 모델 경로에서 지오메트리를 추론합니다. 각 런타임별 벤치마크 지오메트리는 `ModelOptions`에서 더 이상 사용되지 않는 BoxDecode 지오메트리보다 우선합니다.

### 보고서 읽기 {#step-read-report}

반환되는 보고서는 대부분의 사용자가 필요로 하는 주요 필드로 시작합니다. 즉, 평균 엔드투엔드 지연 시간(밀리초), 초당 프레임 단위의 처리량, 평균 및 최대 SoM 전력(와트), `samples`개의 입력을 처리하는 데 필요한 에너지(줄)입니다. `throughput_duration_ms`는 처리량 패스가 실행된 시간을 보고합니다.

전력 값은 Neat 설치 프로그램이 설정하는 보드 텔레메트리 데몬인 Sentinel에서 제공됩니다. `report.telemetry`에는 처리량 패스 동안의 레일 전력, 다이 및 보드 온도, CPU 사용률, MLA 메모리도 포함됩니다. 각 필드에 대해서는 [모델 실행 측정](/develop-apps/development-workflow/model#power-and-board-telemetry)을 참조하십시오. Sentinel이 실행 중이 아니거나 다른 트레이스를 기록 중인 경우에도 벤치마크는 지연 시간과 처리량을 보고하고, 전력 필드를 0으로 두며, `report.telemetry.status`에 그 이유를 설정합니다.

### 고급: 레이어 통계 {#step-read-layer-stats}

레이어 프로파일링은 명시적으로 활성화하는 고급 옵션입니다. `BenchmarkOptions.include_layer_stats`의 기본값은 `false`입니다. 일반 벤치마크는 기존 주요 측정을 유지하며 레이어 프로파일링을 실행하지 않습니다. 튜토리얼 명령에 `--include-layer-stats`를 추가하거나 `BenchmarkOptions`에서 활성화합니다.

```cpp
simaai::neat::BenchmarkOptions options;
options.include_layer_stats = true;
auto report = model.benchmark(options);
```

```python
options = pyneat.BenchmarkOptions()
options.include_layer_stats = True
report = model.benchmark(options)
```

`report.layer_stats`를 순회하여 각 컴파일된 레이어의 `stage_index`, `name`, `total_execution_time_us`를 출력합니다. 컴파일러 순서와 실행 시간이 0인 no-op 항목을 유지합니다. `stage_index`는 A65를 포함한 실행 단계의 0부터 시작하는 순서입니다.

러너를 닫은 후 압축 해제된 MPK 매니페스트에서 MLA ELF를 선택하고 `<elf_stem>_stats.yaml`과 연결합니다. YAML을 변경 없이 임시 디렉터리에 복사하여 `mla-rt --connect mip,ma -vs -N --per-layer-stats <copied_yaml> <elf>`를 실행하고, 생성된 `_output.yaml`을 검증하여 보고서에 읽어 들인 뒤 임시 파일을 삭제합니다. MLA 단계별 제한 시간은 120초이며 mla-rt 기본 입력을 사용합니다. 다른 MLA 작업이 없는 DevKit과 `PATH`에서 사용 가능한 `mla-rt`가 필요합니다. 기본값인 `false`에서는 `report.layer_stats`가 비어 있고 주요 값만 출력합니다. 활성화 시 출력 예시:

```text
Layer stats (execution time in microseconds):
  stage_index=0 name=MLA_0/conv total_execution_time_us=161.95
  stage_index=0 name=MLA_0/noop total_execution_time_us=0
  stage_index=2 name=MLA_1/conv total_execution_time_us=42.5
```

MIP에서 커널로 제어를 넘기는 수정이 포함된 MLA-RT가 필요합니다. eLxr B2121의 MLA-RT `2.2.0~git202610140145.603a8c6-2121`(커밋 `603a8c6`)에서 검증했습니다. 이 수정이 있으면 `benchmark()` 반환 직후 일반 추론을 재개할 수 있으며 수동 재설정이나 대기는 필요하지 않습니다. 이전 빌드는 이후 커널을 통한 추론이 실패하는 하드웨어 상태를 남길 수 있습니다.

프로파일링 전후에 `Model.run()`을 순차적으로 호출할 수 있으며 캐시된 러너도 재사용할 수 있습니다. 레이어 통계를 요청하기 전에 진행 중인 추론을 완료하세요. 프로파일링은 `Model.run()`, 명시적 러너 또는 다른 애플리케이션의 추론과 동시에 실행하면 안 됩니다.

## Run

실행하면 `benchmark()`에서 출력된 벤치마크 요약이 표시되고, 그 뒤에 반환된 보고서에서 출력된 동일한 값이 표시됩니다. **Neat 설치 루트**(`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행합니다. 그런 다음 **리포지토리 루트**에서 **소스 코드를 기반으로 빌드** 명령을 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/003_benchmark_your_model/benchmark_your_model.py \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (build from source):**
```bash
./build.sh --all --no-doc
./build/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

예상 출력(정확한 숫자는 모델, 보드 및 현재 부하에 따라 달라짐. C++ 빌드에서는 마지막 줄에 `[OK]`를 출력함):

```text
NEAT Benchmark
Input: synthetic
Samples: 100
Compiled batch: 1
Measurement detail: e2e+throughput-only
Latency: 1.97 ms
FPS:     1457.04 inferences/s
Batch throughput: 1457.04 batches/s
Throughput window: 30.09 s, 43849 inputs
Power:         13.19 W avg, 13.81 W peak (SoM)
MLA power:     3.05 W avg, 3.25 W max
Energy:        0.91 J for 100 inputs
Die temp:      46.34 °C avg, 47.56 °C max
CPU usage:     12.38 % avg, 13.22 % max
MLA memory:    29.65 MiB
CMA used:      6035.02 MiB avg, 6044.38 MiB max
Linux memory:  682.88 MiB avg, 697.41 MiB max
Telemetry:     Sentinel, 15 samples, trace 20261014T144839.984Z-neat-benchmark-20261014t144839-905z
report_latency_ms=1.96852
report_fps=1457.04
report_avg_power_watts=13.1875
report_energy_joules=0.905085
report_peak_power_watts=13.8125
report_throughput_duration_ms=30094.5
report_telemetry_status=ok
[OK] 003_benchmark_your_model
```

이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt` 파일을 사용하여 자신의 프로젝트에 통합하려면 (별도의 추가 폴더는 필요하지 않음), 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

컴파일된 모델 아카이브에 대한 빠른 답변을 얻고 싶을 때, 즉 실행되는지, 측정된 비동기 처리량은 얼마인지, 그리고 이 대상에서 주요 보드 전력 값은 얼마인지 확인하기 위해 이 벤치마크를 사용하세요.

애플리케이션 성능을 위해 실제 파이프라인도 벤치마크합니다. 합성 모델 입력은 의도적으로 안정적이므로 카메라 흔들림, 코덱 비용, 실제 전처리, 부하 상태에서의 호스트 스케줄링 또는 다운스트림 애플리케이션 로직을 나타내지 않습니다. 수동으로 구축된 비동기 실행을 사용하여 큐 깊이 및 역압 튜닝을 수행하려면 [처리량 및 큐 깊이 조정](/tutorials/tune-throughput-and-queues)을 참조하세요.

`Model::benchmark()`는 구체적인 `input_specs()` 차원을 필요로 합니다. 입력 모양이 동적이거나 구체적이지 않은 경우, 벤치마크는 추측하는 대신 명확하게 실패합니다.

## 보고서 저장

C++ 또는 Python에서 `report.save("results/model/yolo_benchmark.yaml")`을 호출하면 정확히 해당 파일 이름으로 YAML을 저장합니다. 없는 상위 디렉터리는 생성하고 기존 파일은 원자적으로 교체합니다. 지정한 경로가 반환됩니다. 튜토리얼에서도 `--report`로 같은 경로를 지정할 수 있습니다. 저장은 명시적으로 수행하며 `benchmark()`는 보고서 파일을 자동으로 생성하지 않습니다.

저장된 보고서에는 지연 시간 백분위수, 처리량 측정 구간의 메타데이터, Sentinel 텔레메트리도 포함됩니다.

YAML에는 `latency_ms`, `fps`, `avg_power_watts`, `energy_joules`, `layer_stats`가 포함됩니다. 각 `LayerStats` 레코드는 `name`, `stage_index`, `total_execution_time_us`(마이크로초 단위의 하드웨어 `run_time`)를 가집니다. 컴파일러가 생성한 MLA 이름은 원래 ONNX 연산자를 융합하거나 분할한 결과일 수 있습니다. 실행 시간이 0인 no-op 항목도 유지됩니다. 레이어 프로파일링과 주요 측정은 별도의 실행 및 입력을 사용하므로 실행 시간을 동일하게 간주해서는 안 됩니다. `stage_index`는 MLA와 A65 실행 단계를 포함한 모델 실행 순서의 0부터 시작하는 정수이며, 아카이브 파일 순서가 아닙니다. 단일 단계 모델은 `0`을 사용하고, MLA–A65–MLA 모델의 MLA 레코드는 `0`과 `2`를 사용합니다.

파일 누락, 프로세스 실패, 시간 초과 및 잘못된 통계 결과는 예외를 발생시킵니다. 원본 입력 YAML은 유지되며 성공 및 실패 시 임시 파일을 삭제합니다.

## 소스 파일
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
