# 025 PCIe 비동기 추론 실행

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, asynchronous, throughput, detection |

## Concept

동기식 `run()`은 첫 번째 추론에 이상적이지만, 다음 이미지를 제출하기 전에 각 결과가 완료될 때까지 기다립니다. `push()`를 호출하는 프로듀서와 `pull()`을 통해 데이터를 가져오는 컨슈머가 동시에 작동하면 하나의 PCIe 모델이 계속 작동합니다. 처리량은 워밍업 후 완료된 결과에서 계산해야 하며, 모델에 제공된 이미지 수만으로는 계산할 수 없습니다.

## Walkthrough

이 튜토리얼에서는 튜토리얼 024에서 사용한 YOLOv8s 이미지와 박스 디코딩 구성, 640x480 해상도의 거리 장면 이미지를 재사용합니다. 저장 및 이미지 디코딩이 PCIe 측정에 영향을 미치지 않도록 동일한 이미지를 반복적으로 제출합니다.

### 하나의 감지 모델 구성 {#step-configure-model}

이미지를 한 번 로드하고, 카드 측 COCO 전처리 및 YOLOv8 박스 디코딩을 구성한 다음, 큐 0에 하나의 `Model`을 빌드합니다. 누락된 파일이나 카드 시작 오류가 발생하면 측정 시작 전에 프로그램이 중지됩니다.

### 파이프라인 워밍업 {#step-warm-up}

시간을 측정하지 않고 몇 개의 전체 감지를 실행합니다. 워밍업은 모델 시작 및 첫 번째 버퍼 효과를 보고된 워크로드에서 제거합니다.

### 동시 제출 및 검색 {#step-push-pull}

하나의 스레드는 `push()`를 사용하여 이미지를 제출하고, 다른 스레드는 유한한 시간 제한이 있는 `pull()`을 사용하여 BBOX 출력을 검색합니다. 작은 애플리케이션 전용 FIFO는 각 순서대로 제출된 이미지의 시작 시간을 저장합니다. 거부, 시간 초과 또는 잘못된 결과가 발생하면 모델이 닫히고 다른 스레드가 깨어납니다.

이 예제는 일반적인 `Model` 흐름 제어 동작에만 의존하며, 애플리케이션에는 큐 깊이 조정이 없습니다.

### 완료된 작업 보고 {#step-report-results}

두 스레드가 모두 완료되고 모든 수락된 이미지가 검색된 후에만 시간 측정을 중지합니다. 초당 프레임 수는 완료된 출력의 수를 사용합니다. 평균 대기 시간은 각 제출 시도부터 해당 순서대로 정렬된 결과가 도착할 때까지의 시간을 측정합니다.

## Run

PCIe 호스트 패키지를 설치하고 [튜토리얼 설정](/tutorials/before-you-run)에 설명된 대로 튜토리얼 번들을 다운로드합니다. 추출된 PCIe extras 루트에서 다음 명령을 실행합니다. Windows에서는 [Windows에서 애플리케이션 실행](/getting-started/neat-library/pcie-host/#run-applications-on-windows)에 설명된 대로 튜토리얼을 설정하고 프로그램을 실행하세요.

프로그램에는 이 디렉터리의 정확한 경로 `yolo_v8s_mpk.tar.gz`가 필요합니다. 아직 없는 경우 [Model Browser](/tools/model-browser/)에서 YOLOv8s를 다운로드하고, 다운로드한 ZIP 파일에서 아카이브를 추출하여 해당 이름으로 저장하세요.

```bash
sima-cli models download --id yolov8s --variant w8a8 --output models --force
unzip -o models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

Windows에서는 PowerShell에서 모델을 다운로드하고 `Expand-Archive`로 아카이브를 추출하세요.

```powershell
sima-cli models download --id yolov8s --variant w8a8 --output models --force
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8 -Force
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

다운로드한 파일이 ZIP 파일이 아니라 `<id>_<variant>_mpk.tar.gz` 파일인 경우(Model Browser가 모델을 ZIP 파일로 패키징하기 전에 마지막으로 빌드된 모델), 압축을 풀지 말고 `models/<id>/<variant>`에서 해당 파일을 복사하세요.

Python을 실행합니다.

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/025_run_pcie_inference_async/run_pcie_inference_async.py
```

미리 빌드된 C++ 튜토리얼을 실행합니다.

```bash
./lib/sima-pcie-host/tutorials/tutorial_025_run_pcie_inference_async
```

또는 다시 빌드합니다.

```bash
./build.sh --target tutorial_025_run_pcie_inference_async
./build/tutorials-standalone/tutorial_025_run_pcie_inference_async
```

Windows에서는 튜토리얼 복사본 디렉터리의 **x64 Native Tools Command Prompt**에서 CMake로 다시 빌드하세요. `C:\opencv\build`는 [Windows에서 애플리케이션 실행](/getting-started/neat-library/pcie-host/#run-applications-on-windows)에 설명된 OpenCV 패키지의 `build` 디렉터리로 바꾸세요.

```bat
call "%ProgramFiles%\SimaPCIeHost\bin\sima-pcie-host-env.cmd"
set "OpenCV_DIR=C:\opencv\build"
cmake -S share\sima-pcie-host\tutorials -B build
cmake --build build --config Release --target tutorial_025_run_pcie_inference_async
set "PATH=%OpenCV_DIR%\x64\vc16\bin;%PATH%"
build\Release\tutorial_025_run_pcie_inference_async.exe
```

정확한 시간은 호스트 및 카드에 따라 다르지만, 두 프로그램 모두 동일한 측정 경계를 사용하고 다음을 출력합니다.

```text
completed=1000
elapsed_seconds=...
throughput_fps=...
average_latency_ms=...
total_detections=...
[OK] 025_run_pcie_inference_async
```

튜토리얼은 항상 다섯 개의 프레임으로 워밍업한 다음 1,000개의 완료된 프레임을 측정합니다. 다른 카드를 사용하는 경우에만 `--card N`을 전달합니다.

## In Practice

제출 및 검색을 균형 있게 유지합니다. 애플리케이션이 중단 없이 계속 이미지를 제출하는 경우, 일반적인 역압력이 결국 제출 속도를 늦춥니다. 전용 컨슈머를 사용하면 오류를 쉽게 처리할 수 있습니다. 유한한 시간 제한은 중단된 결과를 식별하고, 모델을 닫으면 프로듀서가 대기 중인 경우에도 큐 0이 해제됩니다.

대표적인 벤치마크를 위해 반복되는 프레임을 고정된 이미지 세트로 바꾸고 디스크 읽기를 시간 측정 영역 외부로 이동합니다. [여러 모델을 실행합니다.](/tutorials/run-multiple-models)를 사용하여 두 개의 다른 모델을 동시에 실행합니다.

## 소스 파일

- `run_pcie_inference_async.cpp`
- `run_pcie_inference_async.py`
- `../assets/street-scene.png`
