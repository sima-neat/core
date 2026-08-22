# 024 PCIe를 통해 첫 번째 모델 실행

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, inference, tensor, image, detection |

## Concept

PCIe 호스트 API는 모델에 사용할 수 있는 텐서 또는 디코딩된 이미지 픽셀을 허용합니다. 텐서 모드는 전처리를 호스트에서 수행합니다. 이미지 모드는 원본 픽셀을 보내고 Modalix 카드가 크기 조정, 색상 변환 및 정규화를 수행하도록 합니다. 박스 디코드를 추가하면 이미지 입력은 유지되지만 6개의 원시 YOLO 출력 텐서가 단일 컴팩트한 감지 목록으로 대체됩니다.

## Walkthrough

동일한 YOLOv8s 아카이브와 640x480 거리 장면을 사용하여 세 개의 독립적인 프로그램을 실행합니다. 각 프로그램은 하나의 모드를 시연하고, 큐 0을 동기적으로 사용하며, 하나의 모델을 닫습니다. 이렇게 하면 각 예제를 복사하여 사용할 수 있을 만큼 짧게 유지할 수 있습니다.

### 모델에 사용할 수 있는 텐서 실행 {#step-tensor-mode}

호스트는 이미지를 모델에서 보고한 `[640, 640, 3]` 입력에 맞게 조정하고, BGR을 RGB로 변환하고, 픽셀을 `[0, 1]`로 조정합니다. `Model.run()`은 해당 FP32 텐서를 보내고 카드 측 이미지 전처리를 수행하지 않으며 6개의 원시 YOLO 출력 경로를 모두 출력합니다.

### 전처리를 카드로 이동 {#step-image-mode}

`preprocess.kind`를 `Image`로 설정하고, 들어오는 픽셀을 BGR로 식별하고, `COCO_YOLO` 프리셋을 선택합니다. 이제 호스트는 디코딩된 픽셀을 보내고 카드는 레터박스 크기 조정, BGR-to-RGB 변환 및 정규화를 수행합니다. 프로그램은 6개의 원시 출력 경로 이름과 모양을 출력하므로 텐서 모드와 비교할 수 있습니다.

### 카드의 감지 디코딩 {#step-decode-boxes}

`BoxDecodeType.YoloV8`, 점수 임계값, NMS 임계값 및 출력 제한을 추가합니다. 반환된 BBOX 텐서는 감지 개수를 시작으로, 고정 크기 레코드가 이어집니다. 각 레코드에는 `(x, y, width, height, score, class_id)`가 포함됩니다. 예제는 원본 이미지 좌표에서 처음 10개의 레코드를 구문 분석하고 출력합니다.

### BBOX 텐서 구문 분석 {#step-parse-boxes}

박스 디코드가 하나의 채워진 텐서를 반환했는지 확인하고, 선행 개수를 읽고, 페이로드를 초과하는 개수는 거부합니다. 나머지 각 24바이트 레코드는 출력에 사용할 수 있도록 하나의 감지로 변환됩니다.

## Run

PCIe 호스트 패키지를 설치하고 [튜토리얼 설정](/tutorials/before-you-run)에 설명된 대로 튜토리얼 번들을 다운로드합니다. 추출된 PCIe 추가 루트에서 다음 명령을 실행합니다.

```bash
sima-cli modelzoo get yolo_v8s
```

프로그램에는 이 디렉터리에 `yolo_v8s_mpk.tar.gz`가 필요합니다. Model Zoo 출력 이름과 위치는 다를 수 있습니다. 명령이 정확히 해당 경로를 생성하지 않은 경우 다운로드한 아카이브를 해당 위치에 복사하고 확인합니다.

```bash
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_tensor_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_boxdecode.py
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_024_run_tensor_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_boxdecode
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_024_run_tensor_mode
./build.sh --target tutorial_024_run_image_mode
./build.sh --target tutorial_024_run_image_boxdecode

./build/tutorials-standalone/tutorial_024_run_tensor_mode
./build/tutorials-standalone/tutorial_024_run_image_mode
./build/tutorials-standalone/tutorial_024_run_image_boxdecode
```

일치하는 C++ 및 Python 프로그램은 텐서 모드와 이미지 모드에 대해 동일한 6개의 원시 출력 계약을 출력한 다음 디코딩된 사람, 자동차 또는 기타 보이는 개체를 출력합니다.

```text
Tensor mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_tensor_mode
Image mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_image_mode
Image + boxdecode detections=...
  person score=... box=(...)
[OK] 024_run_image_boxdecode
```

기본값은 카드 0과 큐 0입니다. 다른 카드를 사용할 때만 `--card N`를 전달합니다. 해당 관리 주소는 자동으로 파생됩니다.

## In Practice

애플리케이션에서 이미 정확히 `model.info()`에서 보고하는 dtype, 모양, 레이아웃, 색상 순서 및 숫자 범위를 생성하는 경우 텐서 모드를 사용합니다. 애플리케이션에서 자연스럽게 디코딩된 픽셀을 소유하고 카드가 반복 가능한 모델 전처리를 적용하도록 하려는 경우 이미지 모드를 사용합니다. 애플리케이션에서 원시 기능 맵이 아닌 감지가 필요한 경우 박스 디코드를 활성화합니다.

모든 모드는 동일한 `pcie::Model`/`pyneatpcie.Model` 라이프사이클을 사용합니다. `ModelOptions` 및 제출된 페이로드만 변경됩니다. `push()` 및 `pull()`을 사용하여 제출과 완료를 겹치도록 [PCIe 추론을 비동기적으로 실행합니다.](/tutorials/run-pcie-inference-async)로 계속 진행합니다.

## 소스 파일

- `run_tensor_mode.cpp`
- `run_tensor_mode.py`
- `run_image_mode.cpp`
- `run_image_mode.py`
- `run_image_boxdecode.cpp`
- `run_image_boxdecode.py`
- `../assets/street-scene.png`
