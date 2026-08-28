# 002 비동기적으로 추론 실행

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 10-15 minutes |
| Model | resnet_50 |
| Labels | async, push-pull, throughput, runtime |

## Concept

생산자 스레드에서 모델에 데이터를 입력하는 동시에 다른 스레드에서 예측 결과를 받아와 입력과 출력을 분리하여 실제 처리량을 높입니다. 001장의 ResNet과 동일한 경로를 사용하며, 이제는 비동기 방식으로 작동합니다.

## Walkthrough

제1장에서는 단일 동기 호출을 사용하여 모델을 실행했습니다. 즉, 하나의 프레임을 전달하고 결과가 반환될 때까지 기다립니다. 이는 간단하지만 컴퓨팅 자원을 낭비합니다. 입력 데이터를 생성하는 스레드와 출력 데이터를 소비하는 스레드가 동일하므로 두 작업이 동시에 수행될 수 없습니다. 이 장에서는 동일한 ResNet-50 모델을 사용하되, 두 작업을 분리하여 처리량을 높이는 파이프라인으로 만듭니다.

메커니즘은 비동기 `Run`입니다. 모델을 `Async` 모드에서 `Graph`로 `build()`한 다음, 두 개의 독립적인 호출(생성자에서 `push(...)`, 소비자에서 `pull(...)`)을 사용하여 실행합니다. 결과적으로 생성자 스레드는 런타임이 허용하는 한 최대한 빠르게 프레임을 제공하고, 메인 스레드는 예측 결과를 가져오며, 마지막으로 `pushed=N pulled=N` 라인을 통해 데이터 손실이 없었음을 확인합니다.

### 모델 로드 {#step-load-model}

제1장과 동일하게 아카이브에서 `Model`을 생성하는 것으로 시작하지만, 여기서는 `include_input` 및 `include_output`이 설정된 `RouteOptions`도 선언합니다. 이러한 플래그는 모델이 그래프에 포함될 때 자체 입력 및 출력 경계를 노출하도록 지시하므로, 주변 파이프라인이 프레임을 입력하고 텐서를 출력할 수 있습니다.

### 비동기 파이프라인 구축 {#step-build-async}

`Model`은 직접 push/pull 방식으로 실행할 수 없습니다. `Run`을 사용해야 합니다. 모델을 `graph.add(model.graph(route_opt))`를 통해 새 `Graph`로 래핑한 다음, 대표 프레임을 사용하여 `build(...)`합니다. 샘플 프레임을 전달하면 `build()`가 구체적인 텐서 모양을 미리 협상할 수 있습니다. 반환된 `Run`은 두 스레드가 공유할 핸들입니다.

### 생성자에서 프레임 전송 {#step-push-frames}

생성자의 유일한 작업은 입력을 제공하는 것입니다. 준비된 프레임을 반복하고 각 프레임에 대해 `push(...)`를 호출한 다음, 더 이상 프레임이 없음을 알리기 위해 `close_input()`를 호출하는 스레드를 생성합니다. 이 신호는 소비자가 언제 중지해야 하는지 알 수 있도록 합니다. 생성자가 독립적으로 실행되므로 다음 프레임을 보내기 전에 결과를 기다리지 않습니다.

**C++:** `std::thread`가 루프를 실행합니다. 원자 `pushed` 카운터와 `producer_done` 플래그가 업데이트되므로 메인 스레드는 잠금 없이 진행 상황을 관찰할 수 있습니다.

**Python:** `threading.Thread`인 `frame_producer`가 루프를 실행합니다. 소비자는 나중에 `thread.is_alive()`를 확인하여 완료를 감지합니다.

### 소비자에서 결과 가져오기 {#step-pull-results}

메인 스레드는 결과를 소비합니다. `pull(timeout_ms=2000)`를 호출하는 루프를 실행합니다. 이 함수는 사용 가능한 다음 출력을 반환하거나, 지정된 시간 내에 결과가 없으면 아무것도 반환하지 않습니다. 빈 결과가 반환되면 생성자가 완료되었는지 확인합니다. 완료되었으면 루프를 중지하고, 그렇지 않으면 계속 기다립니다. 각 실제 결과는 상위 1개 클래스 인덱스로 축소되고 출력됩니다. 루프가 끝나면 생성자 스레드를 종료하고 `pushed == pulled`가 올바르게 수행되었는지 확인합니다.

**C++:** `pull()`은 `optional<Sample>`을 반환합니다. 바이트를 읽기 전에 `tensors_from_sample(...)`을 사용하여 텐서를 추출합니다.

**Python:** `pull()`은 `Sample` 또는 `None`을 반환합니다. `sample.tensor.to_numpy()`를 사용하여 배열을 가져온 다음 `argmax`를 적용합니다.

## Run

실행하면 각 프레임마다 하나의 `top1=` 줄이 표시되고, 그 뒤에 푸시/풀 집계 결과가 나타납니다. **Neat 설치 디렉터리**(`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행합니다. 그런 다음 **저장소 루트**에서 **소스 코드를 빌드**하는 명령을 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/002_run_inference_async/run_inference_async.py \
  --model /tmp/resnet_50.tar.gz --n 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_002_run_inference_async \
  --model /tmp/resnet_50.tar.gz --n 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_002_run_inference_async
./build/tutorials-standalone/tutorial_002_run_inference_async \
  --model /tmp/resnet_50.tar.gz --n 4
```

예상 출력(정확한 인덱스는 이미지에 따라 다름. C++ 빌드는 `pushed=...` 필드를 추가하고, Python 빌드는 `pulled=...`만 출력함):

```text
top1=285
top1=285
top1=285
top1=285
pushed=4 pulled=4
[OK] 002_run_inference_async
```

이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt` 파일을 사용하여 자신의 프로젝트에 통합하려면 (별도의 추가 폴더는 필요하지 않음), 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

이 장에서는 비동기 푸시/풀 표면을 사용합니다. 동일한 모델을 결정론적 합성 입력으로 측정하려면 [모델 성능 평가](/tutorials/benchmark-your-model)를 계속 진행하십시오. 전체 빌드 대 실행 및 동기 대 비동기 모델과 함께 완전한 `RunOptions` 표면에 대한 내용은 [첫 번째 그래프 빌드](/tutorials/build-inference-pipeline)를 참조하십시오. 큐 깊이, 오버플로 정책 및 부하 상태에서의 측정에 대한 내용은 [처리량 및 큐 깊이 조정](/tutorials/tune-throughput-and-queues)을 참조하십시오.

## 소스 파일
- C++: `tutorials/002_run_inference_async/run_inference_async.cpp`
- Python: `tutorials/002_run_inference_async/run_inference_async.py`
