# 006 추론 전에 이미지 전처리

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | resnet_50 |
| Labels | preprocessing, normalization, image |

## Concept

전처리 단계를 구성합니다. 즉, 형식, 차원, 채널별 정규화를 설정하여 원본 이미지 입력이 모델이 학습된 텐서와 정확히 일치하도록 합니다. 정확한 전처리는 작동하는 모델과 제대로 작동하지 않는 모델을 구분하는 중요한 요소입니다.

## Walkthrough

컴파일된 모델은 특정 형태와 값 범위의 입력을 예상합니다. 즉, 고정된 색상 순서, 고정된 차원, 그리고 모델이 학습된 방식에 따른 정규화 레시피입니다. 전처리 단계는 원본 디코딩된 이미지를 받아 정확히 해당 텐서로 변환합니다. 전처리를 잘못 설정하면 모델은 여전히 실행되지만, 의미 없는 결과를 반환합니다. 따라서 배포된 모델이 "오류가 있는 것처럼" 보일 때 가장 먼저 확인해야 할 사항은 전처리입니다.

이 장에서는 가장 많이 사용하는 전처리 제어(색상 형식, 입력/출력 차원, 크기 조정 동작 및 채널별 `mean`/`stddev` 정규화)를 구성한 다음, 전체 모델을 통해 단일 결정적 텐서를 실행하기 전에 모델의 전처리 그래프를 검사합니다. 이 장을 마치면 전체 전처리 계약을 선언하고, 모델에 연결하고, 구성된 경로가 존재하는지 확인합니다.

### 전처리 계약 구성 {#step-configure-preproc}

이 옵션들은 전처리 단계에서 적용되는 계약을 선언합니다. `format` (또는 `color_convert.input_format`)는 입력 시 색상 순서를 고정합니다. `input_max_*` 필드는 런타임에서 허용할 동적 입력을 제한합니다. 크기 조정/출력 차원은 추론에 사용되는 텐서의 크기를 설정합니다. `normalize`와 채널별 `mean`/`stddev` 상수는 값 스케일링을 적용합니다. 정규화 상수는 모델의 학습 시 레시피와 일치해야 합니다. 불일치하는 통계는 낮은 신뢰도의 출력의 가장 흔한 원인입니다.

**C++:** 필드는 `Model::Options::preprocess` 아래에 있습니다. `color_convert.input_format`은 `PreprocessColorFormat` 열거형을 사용하고, `normalize.enable`은 `AutoFlag`이며, `normalize.mean` / `normalize.stddev`는 `std::array<float, 3>`입니다.

**Python:** 필드는 `ModelOptions.preprocess` 아래에 있습니다. `color_convert.input_format`은 `PreprocessColorFormat` 열거형을 사용하고, `normalize.enable`은 `AutoFlag`이며, 상수는 `normalize.mean` / `normalize.stddev`에 할당된 목록입니다.

### 모델 구축 {#step-load-model}

아카이브 경로와 옵션을 사용하여 `Model`을 구성하면 전처리 계약이 로드된 모델에 연결됩니다. 이후 모델은 전처리 정의를 포함하므로, 해당 모델에서 파생된 모든 단계 또는 실행은 동일한 레시피를 재사용합니다.

### 전처리 독립적으로 검사 {#step-inspect-preproc}

이 장에서는 전체 모델을 실행하기 전에 전처리 단편을 검사하여 경로가 존재하는지 확인한 다음, 후속 단계를 디버깅할 수 있습니다.

**C++:** `stages::Preproc(frames, model)`은 전처리 단계를 독립적으로 실행하고 전처리된 `Tensor`를 직접 반환합니다. `pre.shape.size()` (랭크)와 `pre.dtype`을 읽어 계약이 제대로 적용되었는지 확인합니다.

**Python:** `model.preprocess()`는 전처리 `Graph` 조각을 반환하므로, 구성된 경로를 검사하기 위해 `describe()`를 출력합니다. 이후 `model.run([tensor])`가 전체 경로를 실행하고 출력 횟수를 보고합니다.

## Run

**Neat 설치 루트** ( `share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++ (미리 빌드된 버전)** 명령을 실행하고, **소스에서 빌드** 명령은 **리포지토리 루트**에서 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/006_preprocess_images/preprocess_images.py \
  --model /tmp/resnet_50.tar.gz --size 224
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_006_preprocess_images \
  --model /tmp/resnet_50.tar.gz --size 224
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_006_preprocess_images
./build/tutorials-standalone/tutorial_006_preprocess_images \
  --model /tmp/resnet_50.tar.gz --size 224
```

예상 출력 (C++ 빌드는 전처리된 텐서의 랭크와 dtype 열거형을 출력합니다):

```text
preproc_rank=3
preproc_dtype=1
[OK] 006_preprocess_images
```

(Python 빌드는 `preproc_graph=ready`, 그래프 설명 및 `output_count=...`를 출력합니다.) 사용자 지정 `CMakeLists.txt` (추가 폴더 불필요)를 사용하여 이 장의 C++ 소스를 자체 프로젝트에 통합하는 방법은 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## 소스 파일
- C++: `tutorials/006_preprocess_images/preprocess_images.cpp`
- Python: `tutorials/006_preprocess_images/preprocess_images.py`
