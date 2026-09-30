# 003 모델 벤치마크

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5분 |
| Model | resnet_50 |
| Labels | benchmark, synthetic, latency, throughput, power |

## Concept

`model.benchmark()`로 컴파일된 모델의 지연 시간, 처리량, 전력, 에너지를 측정합니다. 벤치마크가 합성 입력을 생성하므로 이미지나 데이터셋은 필요하지 않습니다.

## Walkthrough

### 모델 로드 {#step-load-model}

C++에서는 `simaai::neat::Model`, Python에서는 `pyneat.Model`로 컴파일된 `.tar.gz` 아카이브를 로드합니다. 모델의 `input_specs()`에는 구체적인 입력 차원이 선언되어 있어야 합니다.

### 벤치마크 실행 {#step-run-benchmark}

기본 설정을 사용하려면 `model.benchmark()`를 호출하고, 샘플 수를 지정하려면 `model.benchmark(100)`을 사용합니다. API는 모델을 워밍업하고 순차 실행의 지연 시간과 비동기 실행의 처리량을 측정한 뒤, 요약을 출력하고 `BenchmarkReport`를 반환합니다.

샘플 수를 늘리면 처리량과 전력 측정값이 더 안정적일 수 있습니다.

### 보고서 확인 {#step-read-report}

| 필드 | 의미 | 단위 |
| --- | --- | --- |
| `latency_ms` | 워밍업 후 평균 지연 시간 | ms |
| `fps` | 논리적 추론 처리량 | inferences/s |
| `avg_power_watts` | 처리량 측정 중 평균 보드 전력 | W |
| `energy_joules` | 처리량 측정 중 총 에너지 | J |

보드 전력 텔레메트리를 사용할 수 없으면 전력과 에너지 값은 0으로 유지됩니다.

Python 코드에서는 보고서를 저장할 수도 있습니다.

```python
model = pyneat.Model("model.tar.gz")
report = model.benchmark()
report.save_json("benchmark.json")
```

JSON 텍스트를 얻으려면 `report.to_json()`을, 보고서를 다시 표시하려면 `print(report)`를 사용합니다. JSON은 사용할 수 없는 측정값을 유효한 0으로 처리하지 않고 사용 불가로 명시합니다.

## Run

Python 예제와 미리 빌드된 C++ 예제는 `share/`와 `lib/`가 있는 Neat 설치 루트에서 실행합니다. 소스 빌드 명령은 저장소 루트에서 실행하십시오. 두 예제 모두 벤치마크 요약을 출력한 다음 반환된 보고서 필드를 출력합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/003_benchmark_your_model/benchmark_your_model.py \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++(미리 빌드됨):**
```bash
./lib/sima-neat/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++(소스에서 빌드):**
```bash
./build.sh --target tutorial_003_benchmark_your_model
./build/tutorials-standalone/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

다음은 보고서 필드의 예시이며, 수치는 설명을 위한 값입니다.

```text
report_latency_ms=12.4
report_fps=80.6
report_avg_power_watts=2.3
report_energy_joules=2.8
```

사용자 정의 C++ 프로젝트에 대해서는 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

같은 모델 설정과 장치에서 결과를 비교하십시오. 저장된 보고서와 함께 모델 아카이브와 샘플 수도 기록합니다.

합성 입력은 카메라의 타이밍이나 데이터에 따라 달라지는 처리를 재현하지 않습니다. 실제 애플리케이션 파이프라인은 별도로 측정하십시오. 비동기 큐 조정에 대해서는 [처리량 및 큐 깊이 조정](/tutorials/tune-throughput-and-queues)을 참조하십시오.

BoxDecode의 원본 이미지 크기와 기타 벤치마크 옵션에 대해서는 [모델 실행 측정](/develop-apps/development-workflow/model#measure-model-execution)을 참조하십시오.

## 소스 파일
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
