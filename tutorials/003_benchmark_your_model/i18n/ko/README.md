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

`benchmark(samples)`를 호출합니다. API는 비동기 모델 러너를 준비하고, 비동기 푸시/풀 창을 측정하고, 요약을 표준 출력으로 출력한 다음, 동일한 주요 값을 `BenchmarkReport`로 반환합니다.

샘플 수는 측정된 합성 입력의 수입니다. 더 안정적인 처리량 및 전력 값을 얻으려면 더 큰 수를 사용하고, 간단한 성능 테스트만 수행하려는 경우 더 작은 수를 사용합니다.

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

반환되는 보고서는 대부분의 사용자가 필요로 하는 주요 필드만 유지합니다. 즉, 평균 엔드투엔드 지연 시간(밀리초), 초당 프레임 단위의 처리량, 사용 가능한 경우 평균 보드 전력(와트), 사용 가능한 경우 측정된 에너지(줄)입니다.

전력 텔레메트리는 보드 지원에 따라 달라집니다. 런타임이 현재 대상에서 전력 레일을 샘플링할 수 없는 경우, 벤치마크는 여전히 지연 시간과 처리량을 보고하고 전력 필드를 0으로 둡니다.

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
./build.sh --target tutorial_003_benchmark_your_model
./build/tutorials-standalone/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

예상 출력(정확한 숫자는 모델, 보드 및 현재 부하에 따라 달라짐. C++ 빌드에서는 마지막 줄에 `[OK]`를 출력함):

```text
NEAT Benchmark
Input: synthetic
Samples: 100
Latency:      12.4 ms
FPS:          80.6
Power avg:    2.3 W
Energy:       2.8 J
report_latency_ms=12.4
report_fps=80.6
report_avg_power_watts=2.3
report_energy_joules=2.8
[OK] 003_benchmark_your_model
```

이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt` 파일을 사용하여 자신의 프로젝트에 통합하려면 (별도의 추가 폴더는 필요하지 않음), 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

컴파일된 모델 아카이브에 대한 빠른 답변을 얻고 싶을 때, 즉 실행되는지, 측정된 비동기 처리량은 얼마인지, 그리고 이 대상에서 주요 보드 전력 값은 얼마인지 확인하기 위해 이 벤치마크를 사용하세요.

애플리케이션 성능을 위해 실제 파이프라인도 벤치마크합니다. 합성 모델 입력은 의도적으로 안정적이므로 카메라 흔들림, 코덱 비용, 실제 전처리, 부하 상태에서의 호스트 스케줄링 또는 다운스트림 애플리케이션 로직을 나타내지 않습니다. 수동으로 구축된 비동기 실행을 사용하여 큐 깊이 및 역압 튜닝을 수행하려면 [처리량 및 큐 깊이 조정](/tutorials/tune-throughput-and-queues)을 참조하세요.

`Model::benchmark()`는 구체적인 `input_specs()` 차원을 필요로 합니다. 입력 모양이 동적이거나 구체적이지 않은 경우, 벤치마크는 추측하는 대신 명확하게 실패합니다.

## 소스 파일
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
