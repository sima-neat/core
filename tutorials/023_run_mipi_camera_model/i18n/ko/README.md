# 023 MIPI 카메라 모델 실행

## Metadata
| Field | Value |
| --- | --- |
| Category | Cameras & Streaming |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | User-provided camera-compatible model |
| Labels | mipi, camera, live-input, model, ev74 |

## Concept

Modalix DevKit MIPI 카메라를 `CameraInput`을 통해 `Graph`에 연결하고, 실시간 `NV12` 프레임을 모델에서 관리하는 전처리 단계에 입력한 후, 모델 출력을 가져옵니다. 이는 배포된 비전 앱을 위한 직접적인 카메라 경로입니다. 카메라가 소스 데이터를 관리하고, CVU/EV74가 이미지 전처리를 담당하며, MLA가 추론을 실행하고, 애플리케이션이 결과를 사용합니다.

## Walkthrough

이 장에서는 카메라가 이미 보드 오버레이 및 libcamera를 통해 작동한다고 가정합니다. Neat는 `.dtbo` 파일을 선택하거나 ISP를 조정하지 않습니다. 대신 `libcamerasrc`가 프레임을 생성할 수 있게 되면 해당 프레임을 사용합니다. 튜토리얼을 실행하기 전에 [하드웨어 MIPI 가이드](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces) 및 GStreamer 캡 검사를 사용하여 카메라를 확인합니다.

이 튜토리얼을 2단계로 생각하십시오. 1단계는 카메라 설정입니다. 여기에는 오버레이, 드라이버, libcamera, ISP 및 정확한 캡 설정이 포함됩니다. 2단계는 Neat 그래프입니다. 여기에는 카메라 프레임을 CVU 전처리, MLA 추론, 선택적 EV74 BoxDecode로 전달하고 출력을 가져오는 작업이 포함됩니다.

### 카메라 소스 구성 {#step-configure-camera}

`CameraInputOptions`는 Neat가 `libcamerasrc`에서 요청하는 소스 캡(해상도, 프레임 속도, 형식 및 선택적 libcamera 카메라 이름)을 설명합니다. 아직 SiMaAI 제로 복사 버퍼를 지원하지 않는 현재 카메라 스택의 경우 `allow_cpu_fallback = true`를 설정합니다. `libcamerasrc`가 지원하는 경우 `--strict-zero-copy`를 통해 엄격한 제로 복사를 사용할 수 있습니다.

### 모델 경로 구성 {#step-configure-model}

모델은 카메라 프레임을 `NV12` 이미지로 인식합니다. 색상 변환, 크기 조정, 정규화, 양자화 및 테셀레이션을 위한 모델 관리 전처리를 구성합니다. 예제에서는 모델 관리 CVU 전처리를 `EV74`에 고정하여 프로덕션 그래프가 CPU 이미지 파이프라인으로 조용히 바뀌지 않도록 합니다. `--decode none`을 사용하면 경로가 MLA에서 종료되고 원시 모델 텐서가 반환됩니다. YOLO `--decode` 토큰을 사용하면 BoxDecode가 모델 관리 EV74 후처리 단계로 실행됩니다.

### 소스 소유 그래프 구성 {#step-compose-graph}

먼저 `CameraInput`을 추가한 다음 `include_input = false`를 사용하여 모델 경로를 추가합니다. 프레임은 실행 중인 파이프라인 내에서 시작되므로 공개 `Input` 노드는 없습니다. `include_output = true`는 감지 또는 텐서를 위한 풀 엔드포인트를 유지합니다.

### 출력 가져오기 {#step-pull-output}

그래프를 빌드하고 고정된 수의 출력을 가져옵니다. 타임아웃은 `--pull-timeout-ms` 전에 모델 출력이 앱에 도달하지 못했음을 의미합니다. 카메라가 중지되었거나, 캡이 협상되지 않았거나, BoxDecode와 같은 다운스트림 단계에서 역압력이 발생했을 수 있습니다. 텐서 수와 첫 번째 텐서의 모양을 출력하여 애플리케이션 로직을 추가하기 전에 데이터가 이동하는지 확인할 수 있습니다.

## Run

구성된 MIPI 카메라가 연결된 Modalix DevKit에서 이 튜토리얼을 직접 실행합니다. Neat 설치 루트에서 미리 빌드된 명령을 실행하고, 저장소 루트에서 소스 코드를 빌드하는 명령을 실행합니다. 모델 아카이브는 요청한 전처리 및 선택적 `--decode` 모드와 일치해야 합니다.

기본 풀 타임아웃은 15초입니다. 초기 실행 진단 데이터를 수집할 때 콜드 부트된 보드에서 `--pull-timeout-ms` 값을 늘립니다.

**Python:**
<ShellCommand prompt="devkit">
``
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /모델_경로/model.tar.gz --frames 5 --decode none
``
</ShellCommand>

**C++ (prebuilt):**
<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /경로/model.tar.gz --frames 5 --decode none
</ShellCommand>

지원되는 BoxDecode 경로를 사용하는 YOLO 스타일 모델의 경우, `yolov8` 또는 `yolov9seg`와 같은 디코딩 토큰을 선택하십시오.

<ShellCommand prompt="devkit">
``
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/yolo.tar.gz --frames 5 --decode yolov8
``
</ShellCommand>

<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /경로/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

**C++ (build from source):**
<ShellCommand prompt="devkit">
./build.sh --target tutorial_023_run_mipi_camera_model
</ShellCommand>

<ShellCommand prompt="devkit">
./build/tutorials-standalone/tutorial_023_run_mipi_camera_model \
  --model /모델_파일_경로/model.tar.gz --frames 5 --decode none
</ShellCommand>

예상 출력 형태는 모델과 디코딩 경로에 따라 달라집니다. 원시 MLA 출력은 일반적으로 모델별 텐서를 포함합니다.

```text
frame=0 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=1 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=2 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=3 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=4 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
[OK] 023_run_mipi_camera_model
```

지원되는 BoxDecode 경로를 사용하면 출력이 디코딩된 감지 또는 분할 텐서로 변경됩니다. 텐서 개수와 첫 번째 형태를 보편적인 계약으로 사용하는 대신 움직임 확인에 활용하세요.

`output_timeout`가 표시되면 `gst-launch-1.0`를 사용하여 카메라를 확인한 다음 `--print-backend`를 사용하여 생성된 백엔드를 검사합니다. BoxDecode 경로의 경우 모델 아카이브, `--decode` 토큰, 임곗값이 모델과 일치하는지 확인합니다.

## In Practice

생성된 GStreamer 경로를 검사해야 할 때 `--print-backend`를 사용하십시오. 프로덕션 경로에는 폴백이 활성화된 경우 `libcamerasrc`, `neatcamerabridge`, `neatprocesscvu`, `neatprocessmla`, 선택 사항인 EV74 후처리, 그리고 `appsink`가 포함되어야 합니다. 의도적으로 디버그 전용 경로를 추가하지 않은 한 `appsrc`, `ostosima`, `videoconvert` 또는 `videoscale`는 포함되어서는 안 됩니다.

## 소스 파일
- C++: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.cpp`
- Python: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py`
