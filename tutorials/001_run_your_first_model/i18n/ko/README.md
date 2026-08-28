# 001 첫 번째 모델 실행하기

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | <5 minutes |
| Model | resnet_50 |
| Labels | model, inference, foundations |

## Concept

컴파일된 ResNet-50 모델 아카이브를 로드하고, 이미지를 입력한 후, 가장 가능성이 높은 클래스를 확인합니다. 이는 "모델 아카이브가 있습니다"에서 "예측 결과를 얻었습니다"에 이르는 가장 빠른 경로입니다.

## Walkthrough

이것은 시작 장입니다. 목표는 가능한 한 가장 간단한 엔드투엔드 추론을 수행하는 것입니다. 컴파일된 모델을 가져와 이미지 하나를 입력하고 예측된 클래스 인덱스를 출력합니다. 그래프, 스레드, 스트리밍은 필요 없습니다. 모든 Neat 프로그램이 구축되는 세 가지 호출만 있으면 됩니다.

*컴파일된 모델*은 배포 가능한 `.tar.gz` 아카이브이며, 여기에는 MPK 추론 계약이 포함됩니다. 즉, 모델 아티팩트와 런타임 메타데이터가 포함되어 있으며, Neat은 이를 사용하여 대상 장치에서 실행합니다. 직접 아카이브를 압축 해제하거나 단계를 연결할 필요가 없습니다. Neat에 아카이브를 지정하고, 입력을 제공하고, 출력을 읽기만 하면 됩니다. 결국 세 줄의 코드로 추론을 실행하고 `top1=` 클래스 인덱스를 출력할 수 있습니다.

### 모델 로드 {#step-load-model}

첫 번째 줄은 디스크의 경로를 활성 상태의 실행 가능한 `Model`로 변환합니다. 구성은 아카이브를 로드하고 실행을 위해 준비합니다.

**C++:** 두 번째 인수로 `build_options(size)`를 전달하여 이 모델이 예상하는 입력 계약을 선언합니다. 즉, RGB 색상, `224×224` 크기, 그리고 ResNet-50이 학습된 ImageNet 정규화입니다. 여기에 선언하면 런타임이 원본 이미지를 모델이 원하는 텐서로 변환하는 방법을 알 수 있습니다.

**Python:** `pyneat.Model`을 구성할 때 동일한 계약을 `build_options(size)`를 통해 전달합니다.

### 입력 준비 {#step-prepare-input}

다음으로 분류할 정확히 하나의 이미지를 생성합니다. `--image`를 전달하면 이미지를 읽고, `224×224` 크기로 조정하고, RGB로 변환하여 입력 계약과 일치시킵니다. 그렇지 않으면 단색 회색 프레임을 합성하여 전체 로드 → 실행 → 읽기 경로가 계속 작동하도록 합니다. 이렇게 하면 실제로 사용할 자산이 없어도 됩니다.

**C++:** 프레임은 `cv::Mat`이며, `load_rgb(...)`를 사용하여 생성하거나 회색 자리 표시자로 생성합니다.

**Python:** 프레임은 `load_image(...)`를 사용하여 생성된 NumPy 배열이며, RGB 이미지 메타데이터와 함께 `Tensor`로 래핑됩니다.

### 추론 실행 및 결과 읽기 {#step-run-inference}

세 번째 줄은 실제 작업을 수행합니다. `run()`은 입력을 받아 `timeout_ms`와 함께 동기적으로 모델을 실행하고 출력을 반환합니다. `timeout_ms`는 대기할 최대 벽시계 시간입니다. 여기서는 `2000`ms이므로 "장치가 2초 이내에 출력을 생성하지 않으면 오류를 발생시킨다"는 의미이며, 무한정 대기하는 것이 아닙니다. ( `-1`을 전달하면 무한정 차단됩니다. 실제 코드에서는 유한한 값을 사용하는 것이 좋습니다.) 그런 다음 `argmax`를 사용하여 출력을 단일 클래스 인덱스로 줄이고 `top1=`을 출력합니다.

**C++:** `run()`은 `TensorList`를 반환합니다. `map_read()`를 통해 첫 번째 텐서의 바이트를 읽습니다.

**Python:** 텐서/이미지 입력을 사용하여 `run()`을 실행하면 `TensorList`가 반환됩니다. `outputs[0].to_numpy()`는 NumPy 배열을 `argmax`에 전달합니다.

이것이 전체 이야기입니다. 이후 장의 모든 내용(비동기, 파이프라인, 사용자 정의 그래프)은 동일한 세 가지 단계(구성, 입력, 읽기)를 기반으로 구축됩니다.

## Run

프로그램을 실행하면 예측된 클래스 인덱스가 표준 출력에 표시됩니다. **Neat 설치 디렉터리**(`share/` 및 `lib/`가 포함된 디렉터리)에서 **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행하고, **리포지토리 루트**에서 **소스 코드를 빌드**하는 명령을 실행합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/001_run_your_first_model/run_your_first_model.py \
  --model /tmp/resnet_50.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_001_run_your_first_model \
  --model /tmp/resnet_50.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_001_run_your_first_model
./build/tutorials-standalone/tutorial_001_run_your_first_model \
  --model /tmp/resnet_50.tar.gz
```

예상 결과(정확한 인덱스는 이미지에 따라 다름):

```text
top1=285
[OK] 001_run_your_first_model
```

이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt`와 함께 자신의 프로젝트에 통합하려면 (별도의 추가 폴더는 필요하지 않음) 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

처리량, 배치 처리 또는 실시간 스트리밍의 경우 002장으로 계속 진행하십시오. 참고: [모델](/develop-apps/development-workflow/model).

## In Practice

튜토리얼과 테스트가 모델 아카이브(`.tar.gz`)와 샘플 에셋을 어디에서 찾는지, 그리고 이를 로컬에 어떻게 제공하는지 설명합니다. 이는 모든 모델 기반 튜토리얼의 필수 조건입니다.

### `sima-cli`가 PATH에 포함되어 있는지 확인합니다.

일부 테스트는 비대화형 셸에서 `sima-cli`를 호출합니다. `sima-cli`를 설치한 후 한 번만 실행하십시오.

```bash
SIMA_CLI_BIN_DIR="<path-to-sima-cli-bin>"
grep -Fqx "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" ~/.bashrc || echo "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

그런 다음 확인합니다.

```bash
/bin/sh -c 'command -v sima-cli'
```

### 모델 아카이브 위치 및 환경 변수

추출/런타임 배치 설정:
- `SIMA_MPK_EXTRACT_ROOT=<dir>`은 기본 추출 디렉터리를 설정합니다.
- `SIMA_MPK_CLEANUP_EXTRACTED=0`은 프로세스 종료 후 추출된 `proc_*` 모델 데이터를 보존합니다.
- `SIMA_MPK_EXTRACT_GC_STALE_PROC=0`은 시작 시 사용하지 않는 `proc_*` 정리 작업을 비활성화합니다.

#### ResNet50

검색 순서:
1. `SIMA_RESNET50_TAR` (모델별 재정의)
2. `SIMA_MODEL_TAR` (모델 아카이브 테스트/예제에 대한 공유 대체)
3. `tmp/resnet_50.tar.gz`
4. 발견된 경우 `tmp/`로 이동된 로컬 파일: `resnet_50.tar.gz`, `resnet-50.tar.gz`

다운로드 (`sima-cli`가 사용 가능한 경우):
```bash
sima-cli modelzoo get resnet_50
```

### 샘플 이미지

튜토리얼/테스트에 사용되는 기본 이미지 후보:
- `tmp/coco_sample.jpg` (파일이 없는 경우 다운로드)
- `test.jpg`
- `tests/assets/preproc_dynamic/ilena_488.jpg`

다음과 같이 테스트에서 사용되는 COCO 이미지 URL을 변경할 수 있습니다.
```bash
SIMA_COCO_URL=<custom_url>
```

### 테스트 파일 다운로드 위치

테스트 및 예제는 일반적으로 다운로드한 파일을 리포지토리 루트의 `tmp/`에 저장합니다. 필요한 파일이 없는 경우 튜토리얼은 정상적으로 건너뜁니다.

### 파일 관련 문제 해결

- 튜토리얼에서 `SKIP: missing ...` 메시지가 출력되면 해당 파일을 제공하거나 플래그를 전달합니다(예: `--model <path>`, `--image <path>`).
- `sima-cli`를 사용할 수 없는 경우, 환경 변수를 설정하여 로컬 모델 아카이브를 가리키도록 합니다.

## 소스 파일
- C++: `tutorials/001_run_your_first_model/run_your_first_model.cpp`
- Python: `tutorials/001_run_your_first_model/run_your_first_model.py`
