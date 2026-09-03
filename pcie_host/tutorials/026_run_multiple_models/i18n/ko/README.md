# 026 여러 모델 실행

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | resnet_50, yolo_v8s |
| Labels | PCIe, queues, concurrency, classification, detection |

## Concept

A Modalix PCIe 카드는 0번부터 3번까지의 큐를 노출합니다. 활성화된 각 큐는 `Model` 하나를 소유하고, 큐를 사용하므로 독립적인 모델은 서로 다른 모델에 할당하여 동시에 실행할 수 있습니다.
`ConnectionOptions.queue` 각 인스턴스에 대해. 이 튜토리얼에서는 전역 코디네이터를 추가하지 않고 큐 0에 ResNet-50을, 큐 1에 YOLOv8s를 사용합니다.

## Walkthrough

두 모델은 의도적으로 서로 다른 이미지를 사용합니다. ResNet-50은 선명한 래브라도어 개의 사진을 분류하는 반면, YOLOv8s는 붐비는 거리의 장면에서 사람과 자동차를 감지합니다.

### 모델별 이미지를 로드합니다. {#step-load-assets}

각 큐에 모델 아카이브와 패키징된 에셋을 모두 검증하고, 큐에 추가하기 전에 디코딩합니다. 이미지를 분리하면 각 결과가 의미를 가지게 되고, 분류 포트레이트를 객체 감지 작업에 사용하는 것을 방지할 수 있습니다.

### 각 큐에 하나의 모델을 할당합니다. {#step-assign-queues}

두 개의 일반적인 `Model` 객체. ResNet-50을 사용하여 ImageNet 이미지 전처리 작업을 큐 0에서 수행하고, YOLOv8s를 사용하여 COCO 이미지 전처리 작업과 박스 디코딩 작업을 큐 1에서 수행하도록 구성합니다. 빌드 오류가 발생하면 오류가 발생한 큐와 모델을 식별합니다. 두 번째 빌드가 실패하면 이미 빌드된 모델이 닫힙니다.

### 두 큐를 동시에 실행합니다. {#step-run-concurrently}

각 모델에 대해 별도의 호스트 스레드에서 차단 이미지 추론을 하나씩 시작합니다. 각 호출은 여전히 간단한 동기식 모델을 사용합니다. `run` 행동은 동일하지만, 통화가 겹치는 이유는 서로 다른 물리적 대기열을 대상으로 하기 때문입니다.

### 각 결과를 독립적으로 해석하십시오. {#step-read-results}

큐 0은 하나의 FP32 분류 텐서를 반환하고 가장 높은 점수를 받은 ImageNet 클래스를 출력합니다. 큐 1은 디코딩된 BBOX 레코드를 반환하고 감지 클래스, 신뢰도 및 원본 이미지 좌표를 출력합니다. 두 모델 중 하나를 닫으면 해당 모델에 할당된 큐만 해제됩니다.

## Run

PCIe 호스트 패키지를 설치하고, 설명서에 나와 있는 대로 튜토리얼 번들을 다운로드하세요.
[튜토리얼 설정](/tutorials/before-you-run)추출된 PCIe
추가 루트에서 두 모델을 모두 다운로드합니다.

```bash
sima-cli modelzoo get resnet_50
sima-cli modelzoo get yolo_v8s
```

이 프로그램은 정확한 경로를 필요로 합니다. `resnet_50_mpk.tar.gz` 그리고
`yolo_v8s_mpk.tar.gz` 이 디렉터리 안에 있습니다. 만약 Model Zoo 다른 이름이나 위치를 사용한 경우, 다운로드한 파일을 해당 위치에 복사하고 파일이 올바른지 확인합니다.

```bash
cp /absolute/path/to/downloaded-resnet-archive.tar.gz resnet_50_mpk.tar.gz
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f resnet_50_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

Python 실행:

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/026_run_multiple_models/run_multiple_models.py
```

미리 빌드된 C++ 튜토리얼을 실행합니다.

```bash
./lib/sima-pcie-host/tutorials/tutorial_026_run_multiple_models
```

또는 다시 구축합니다.

```bash
./build.sh --target tutorial_026_run_multiple_models
./build/tutorials-standalone/tutorial_026_run_multiple_models
```

문서화된 모델과 에셋을 사용하면 두 버전 모두 다음과 유사한 결과를 출력합니다.

```text
queue=0 model=resnet_50 output_shape=[1, 1000] top1=208 (Labrador retriever)
queue=1 model=yolo_v8s detections=...
  person score=... box=(...)
[OK] 026_run_multiple_models
```

튜토리얼에서는 ResNet-50을 큐 0에, YOLOv8s를 큐 1에 할당하도록 의도적으로 설정합니다.
통과 `--card N` 다른 카드를 사용할 때만 해당됩니다.

## In Practice

큐 할당은 애플리케이션 리소스 결정입니다. 두 개의 활성 모델은 동일한 물리적 큐를 가질 수 없습니다. 작업을 시작하기 전에 모델을 구축하고, 오류 발생 시 특정 큐를 보고하며, 정상 및 오류 경로 모두에서 성공적으로 구축된 모든 모델을 닫습니다. 분리합니다. `Model` 인스턴스는 결과를 보관하고 오류를 격리하면서도 분석하기 쉽도록 유지합니다.

배포 진단을 위해 다음 단계를 진행합니다.
[PCIe 모델 워크플로](/develop-apps/development-workflow/pcie-model/) 그리고
[문제 해결 안내서](/reference/troubleshooting/).

## 소스 파일

- `run_multiple_models.cpp`
- `run_multiple_models.py`
- `../assets/labrador.jpg`
- `../assets/street-scene.png`
