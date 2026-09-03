# 017 프로덕션 환경에 적합한 파이프라인 구축

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | resnet_50 |
| Labels | production, reliability, deployment |

## Concept

이전 장에서 배운 내용을 바탕으로 프로덕션 스타일의 실행 루프를 구성합니다. 여기에는 명시적인 모델 옵션, 명시적인 라우트 옵션, 명시적인 실행 옵션, 그리고 하나의 비동기 푸시/풀 루프가 포함됩니다. 이는 완전한 제품 프레임워크가 아니라, 필요에 따라 조정하여 사용할 수 있는 안정적인 기본 구조입니다.

## Walkthrough

이것은 핵심적인 마지막 장입니다. 지금까지는 한 번에 하나의 개념만 다루었지만, 이제 이 모든 개념이 하나로 모여 실제 배포 코드에 적용할 수 있는 단일 청사진을 형성합니다. 템플릿의 전체적인 목적은 기본 설정에서 암시적으로 처리되는 세 가지 사항을 명시적으로 드러내는 것입니다. 즉, 모델의 입력 범위(계약 위반 시 빌드 시간에 오류가 발생하고, 파이프라인 중간에 오류가 발생하지 않도록 함), 단계 이름 지정(여러 모델이 동일한 프로세스를 공유할 때 진단 정보가 읽기 쉽도록 함), 큐 정책(부하 상태에서 동작을 예측 가능하게 하고, 불확실하게 만들지 않도록 함)입니다.

구조는 다음과 같습니다. 실행 옵션을 구성하고, 모델을 구성하고 로드한 다음, 러너를 빌드하고, 제한된 비동기 루프를 사용하여 실행합니다. 마지막에는 프로덕션 기본 설정을 사용하고 성공적인 출력을 계산하는 푸시/풀 루프를 통해 비동기 파이프라인을 실행하는 `Runner`가 생성됩니다. 이는 동일한 애플리케이션의 여러 모델에 걸쳐 표준화할 런타임 골격입니다.

### 실행 옵션 구성 {#step-configure-run-options}

이것은 프로덕션 런타임의 기본 설정입니다. `queue_depth = 8`은 작은 제한된 버퍼를 제공합니다. `overflow_policy = Block`은 프로듀서가 프레임을 자동으로 삭제하는 대신 대기하도록 하여 손실이 발생할 경우 안전한 선택을 합니다. `output_memory = Owned`는 반환된 텐서가 풀 이후에도 유지되도록 합니다. 이러한 설정을 명시적으로 지정하는 것은 기본 설정에 의존하는 대신 부하 상태에서의 동작을 예측 가능하게 만듭니다.

### 모델 구성 및 로드 {#step-configure-model}

여기서는 모델에 입력 계약을 명시적으로 적용합니다. `preprocess.input_max_width/height/depth`를 프레임의 차원으로 설정하면 일치하지 않는 입력이 빌드 시간에 명확한 계약 오류와 함께 실패하므로 나중에 혼란스러운 런타임 오류가 발생하는 대신 오류를 방지할 수 있습니다. `name_suffix = "_prod"`는 이 모델의 단계를 태그하여 다중 모델 애플리케이션에서 진단 정보에 표시될 수 있도록 합니다. 그런 다음 아카이브 경로와 이러한 옵션에서 `Model`을 구성합니다.

**C++:** `Model::Options`는 또한 모델이 예상하는 전처리(`InputKind::Image`, RGB 색상 변환 및 `has_explicit_stats = true`를 사용한 ImageNet 정규화)를 명시적으로 지정합니다. 이는 C++ 경로가 아카이브 기본 설정에 의존하는 대신 전처리를 미리 선언하기 때문입니다.

**Python:** `ModelOptions`는 `mopt.preprocess.*` 내에서 이미지 전처리, 입력 범위, ImageNet 정규화 및 접미사를 설정합니다.

### 러너 빌드 {#step-build-runner}

`ModelRouteOptions` (C++) `Model::RouteOptions`)는 경로에 포함될 경계선을 선택합니다. `include_input` 그리고 `include_output` 여기에서도 사실이고, 동일하게 적용됩니다. `_prod` 접미사를 추가하여 경로의 요소가 모델의 명명 규칙과 일치하도록 합니다. 그런 다음 다음을 호출합니다. `model.build(sample, route_options, run_options)`: 한 번의 통화로 연결되는 경로 `Model` 바로 실행 가능한 상태로 `Runner`경로와 실행 옵션을 모두 기본 파이프라인으로 전달합니다. 대표 샘플을 사용하면 빌드 프로세스가 협상된 형태를 고정할 수 있습니다.

**C++:** 샘플은 `TensorList` 다음과 같이 제작됨 `Tensor::from_cv_mat(rgb, ..., TensorMemory::EV74)`입력을 장치에 적합한 메모리에 배치합니다.

**Python:** 샘플은 하나의 요소를 포함하는 목록입니다. `Tensor` ~에서 `Tensor.from_numpy(...)`.

### 생산 프로세스 개선 {#step-run-loop}

이것은 실제 서비스가 실행되는 루프입니다. 각 반복마다 `push(...)` 입력 — 부울 값을 반환하는지 확인하여 거부된 푸시(아래) `Block`일시적인 상태이므로 잘못 계산하는 대신 적절하게 처리합니다. `pull(...)` 제한된 시간 내에 실행하고, 성공적으로 완료된 횟수를 기록합니다. 루프가 끝난 후에는 `close()` 코드를 깔끔하게 정리합니다. 이 푸시-부울/타임아웃 대기/명시적 닫기 패턴은 안정적인 비동기 기본 구조입니다. 실제 입력 및 출력 처리 부분을 대체하면 구조는 동일하게 유지됩니다.

## Run

이 장에서는 모델 아카이브가 필요합니다.`resnet_50`). 다음 명령어를 실행합니다. **Python** 및 **C++ (사전 빌드)**Neat root 디렉터리(해당 디렉터리에 포함된)를 설치합니다. `share/` 그리고 `lib/`); 소스 코드를 기반으로 빌드하는 명령어를 **저장소의 최상위 디렉터리**에서 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/017_build_production_pipeline/build_production_pipeline.py \
  --model /tmp/resnet_50.tar.gz --iters 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_017_build_production_pipeline \
  --model /tmp/resnet_50.tar.gz --iters 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_017_build_production_pipeline
./build/tutorials-standalone/tutorial_017_build_production_pipeline \
  --model /tmp/resnet_50.tar.gz --iters 4
```

예상 결과:

```text
outputs=4
[OK] 017_build_production_pipeline
```

(Python 빌드에서는 다음이 출력됩니다.) `iters=4 ok=4`.)

이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt`를 사용하여 자신의 프로젝트에 통합하려면(추가 폴더는 필요하지 않음), 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## 소스 파일
- C++: `tutorials/017_build_production_pipeline/build_production_pipeline.cpp`
- 파이썬: `tutorials/017_build_production_pipeline/build_production_pipeline.py`
