# 005 모델 옵션 구성

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | yolo_v8s |
| Labels | model-options, configuration, contracts |

## Concept

`ModelOptions`는 입력 데이터, 모델의 파이프라인 단계, 그리고 출력 디코딩 간의 관계를 정의하는 구조체입니다. 기본 동작에서 벗어나 새로운 동작을 구현할 때 가장 먼저 참조해야 하는 곳입니다.

## Walkthrough

001장에서는 합리적인 기본값을 사용하여 모델을 로드했습니다. 실제 모델, 특히 YOLOv8과 같은 감지 모델은 사용자가 해당 모델의 계약을 *선언*해야 합니다. 즉, 입력되는 픽셀 형식과 크기, 정규화 방법, 그리고 원시 네트워크 출력이 필터링된 박스로 변환되는 방식을 지정해야 합니다. `ModelOptions` 다음은 런타임에 해결된 옵션에서 계약을 검사한 후, YOLOv8 모델을 처음부터 끝까지 구성하는 내용입니다. 마지막에는 입력, 전처리 및 후처리를 설정하고, 해결된 모델을 다시 읽어볼 수 있습니다. `input_specs()`/`output_specs()`/`metadata`, 그리고 구성된 모델을 통해 하나의 결정론적 프레임을 실행합니다.

### 입력 및 전처리 선언 {#step-set-input-preproc}

첫 번째 블록에서는 프레임이 어떻게 생겼는지와 네트워크에 맞게 프레임을 준비하는 방법을 설명합니다. `format` (`BGR` 여기)와 `input_max_width`/`height`/`depth` 범위는 런타임에 검증하는 입력 계약을 설정하고 버퍼 크기를 조정합니다. 정규화 필드는 모델이 학습된 각 채널의 평균값과 표준편차를 제공하므로, 원본 픽셀은 네트워크가 예상하는 범위로 조정됩니다.

**C++:** 필드는 다음 위치에 있습니다. `opt.preprocess.*`: `kind = InputKind::Image`, `color_convert.input_format = PreprocessColorFormat::BGR`그리고 `normalize.enable = AutoFlag::On` 함께 `mean`/`stddev` 처럼 `std::array<float, 3>`.

**Python:** 필드는 다음 경로에 있습니다. `opt.preprocess.*`: `kind = pyneat.InputKind.Image`, `color_convert.input_format = pyneat.PreprocessColorFormat.BGR`그리고 `normalize.enable = pyneat.AutoFlag.On` 함께 `mean`/`stddev` 목록.

### 후처리 선언 {#step-set-postproc}

두 번째 블록은 검출기의 출력을 형성합니다. `decode_type` YOLOv8 박스 디코딩 경로를 선택하고, `score_threshold`, `nms_iou_threshold`그리고 `top_k` 원시 감지 결과를 필터링합니다. 즉, 신뢰도가 낮은 감지 결과는 제거하고, 겹치는 감지 결과는 병합하며, 최종적으로 남을 감지 결과의 수를 제한합니다. `boxdecode_original_width`/`boxdecode_original_height` 디코더가 정규화된 좌표를 픽셀에 다시 매핑하는 데 필요한 소스 프레임의 기하학적 정보를 제공하고, `name_suffix` 생성된 단계 이름을 안정화하여 다른 단계와 결합할 때 파이프라인 그래프가 읽기 쉬운 상태를 유지합니다.

**C++:** `decode_type = BoxDecodeType::YoloV8`; 기하학 필드는 다음과 같습니다. `boxdecode_original_width`/`boxdecode_original_height`.

**Python:** `decode_type = pyneat.BoxDecodeType.YoloV8`; 기하학 필드는 다음과 같습니다. `boxdecode_original_width`/`boxdecode_original_height`.

### 해결된 계약을 불러오고 검사합니다. {#step-load-and-inspect}

구축하는 과정 `Model` 이러한 옵션을 사용하면 계약 내용을 아카이브와 비교하여 충돌을 해결할 수 있습니다. 그런 다음 다시 읽어봅니다. `input_specs()` 그리고 `output_specs()` 협상된 텐서 제약 조건을 보고하고, `metadata()` 아카이브에 내장된 키/값 계약을 노출합니다. 로드 후 이러한 항목을 검사하면 런타임에서 사용자의 옵션을 수락했는지 확인할 수 있으며, 사용자가 작업할 구체적인 형식을 알 수 있습니다.

**C++:** 사양은 다음과 같습니다. `TensorConstraint` 값; 구체적인 형태를 출력합니다.

**Python:** 형태를 출력합니다. `input_specs()[0]` 그리고 `output_specs()[0]`그리고 `len(model.metadata())`.

### 한 프레임 실행 {#step-run-inference}

마지막으로 하나를 종합합니다. `640×640` BGR 프레임을 구성된 모델에 적용하고, 전체 계약이 처음부터 끝까지 제대로 실행되는지 확인한 후 결과물의 개수를 출력합니다.

**C++:** 프레임은 `cv::Mat`; `run()` 반환합니다. `TensorList` 누구의 `size()` 저희는 다음과 같이 인쇄합니다. `outputs=`.

**Python:** 프레임은 다음과 같이 래핑됩니다. `Tensor` 을 통해 `Tensor.from_numpy(...)`; `run()` 반환합니다. `TensorList`따라서 길이를 출력합니다.

## Run

실행하면 해결된 스펙 모양, 메타데이터 키 개수, 그리고 결과 합계가 표시됩니다. **Neat 설치 디렉터리**(`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행하고, **리포지토리 루트**에서 **소스 코드를 빌드**하는 명령을 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/005_configure_model_options/configure_model_options.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_005_configure_model_options
./build/tutorials-standalone/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

예상 출력(모양 및 키 개수는 모델 아카이브에 따라 달라짐. C++ 빌드는 상세한 사양 정보를 출력하고, `outputs=`를 출력하며, Python 빌드는 모양과 `output_count=`를 출력함):

```text
input_specs[0]: shape=[640,640,3]
output_specs[0]: shape=[]
metadata_keys=8
outputs=1
[OK] 005_configure_model_options
```

이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt` 파일을 사용하여 자신의 프로젝트에 통합하려면 (별도의 추가 폴더는 필요하지 않음), 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## In Practice

### 상세 출력 수준 사전 설정

프레임워크 빌드/실행 메시지는 `GraphOptions`, `Model::Options` 및 `Model::RouteOptions`에서 `VerboseOptions`를 사용하여 제어됩니다.

현재 개발 기본값: `VerboseOptions::debug_all()`. 출력을 줄이려면 `production()` 또는 `quiet()`를 명시적으로 호출합니다.

| 사전 설정 | 사용 목적 |
|---|---|
| `VerboseOptions::quiet()` | 프레임워크 진행 상황 및 상세 출력을 억제합니다. |
| `VerboseOptions::production()` | 깔끔한 단계 진행 상황만 표시합니다. |
| `VerboseOptions::debug_plugins()` | 프로덕션 UX를 유지하면서 플러그인 및 GStreamer 관련 정보를 표시합니다. |
| `VerboseOptions::debug_all()` | 모든 항목에 대해 완전한 상세/세부 출력을 강제합니다. |

런타임 큐/처리량 조정에 대해서는 [처리량 및 큐 깊이 조정](/tutorials/tune-throughput-and-queues)을 참조하십시오.

## 소스 파일
- C++: `tutorials/005_configure_model_options/configure_model_options.cpp`
- Python: `tutorials/005_configure_model_options/configure_model_options.py`
