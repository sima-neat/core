---
title: "튜토리얼 설정"
description: "실행 환경을 선택하고, 튜토리얼을 다운로드하고, 모델 아카이브를 준비하세요."
sidebar_position: 2
slug: /tutorials/before-you-run
---

# 튜토리얼 설정

튜토리얼을 시작하기 전에 이 설정을 한 번 완료하세요. 튜토리얼 범주에 맞는 환경을 선택하세요. Neat Library와 PCIe 번들은 서로 호환되지 않습니다.

## 1. 사용할 환경을 선택하세요.

| 튜토리얼 카테고리 | 실행 중 | 파이썬 환경 |
| --- | --- | --- |
| 모델 및 추론, 그래프 및 파이프라인, 카메라 및 스트리밍, 생성형 AI | Modalix 또는 튜토리얼에서 지정된 환경인 DevKit | `~/pyneat` |
| PCIe 공동 처리 | 호스트가 Modalix PCIe 카드에 연결되었습니다. | `~/pyneatpcie` |

PCIe 튜토리얼은 SDK 컨테이너 내부나 카드 자체에서 실행되는 것이 아니라 호스트에서 실행됩니다.

## 2. Neat Library 사용법 튜토리얼을 준비합니다.

[Neat Library가 설치되었습니다.](/getting-started/neat-library/install-or-update/)인지 확인한 다음, 튜토리얼 번들을 적용하려는 디렉터리에서 다음 명령을 실행하세요.

<ShellCommand prompt="sdk|devkit">
sima-cli neat install core -t extras
cd sima-neat-*-Linux-extras
</ShellCommand>

DevKit에서 직접 실행되는 Python 튜토리얼의 경우, PyNeat을 활성화하고 다음을 통해 임포트가 제대로 되었는지 확인합니다.

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 -c "import pyneat; print('pyneat ready')"
</ShellCommand>

## 3. PCIe 튜토리얼을 설정합니다.

먼저 [PCIe 호스트 패키지를 설치하고 확인합니다.](/getting-started/neat-library/pcie-host/)를 설치하고 확인합니다.
그런 다음 호스트에서 실행되는 Ubuntu 버전에 대한 튜토리얼 패키지를 다운로드합니다.
패키지를 원하는 디렉터리에서 명령을 실행합니다.

**Ubuntu 22.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu22/amd64 -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

**Ubuntu 24.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu24/amd64 -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

PCIe PyNeat를 확인합니다.

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

## 4. 모델 아카이브를 준비합니다.

튜토리얼에서 지정된 모델을 다운로드하려면 Model Zoo를 사용하세요. 예를 들어:

<ShellCommand prompt="sdk|devkit|pcie-host">
sima-cli modelzoo get resnet_50
sima-cli modelzoo get yolo_v8s
</ShellCommand>

Neat Library 튜토리얼은 `--model`을 허용하므로 다운로드한 아카이브 파일을 직접 사용할 수 있습니다. PCIe 튜토리얼은 PCIe 추가 자료의 루트 디렉터리에 있는 고정된 파일 이름을 사용합니다.

| PCIe 튜토리얼 | 필수 모델 파일 |
| --- | --- |
| PCIe를 통해 첫 번째 모델을 실행하세요. | `yolo_v8s_mpk.tar.gz` |
| PCIe 추론을 비동기적으로 실행합니다. | `yolo_v8s_mpk.tar.gz` |
| 여러 모델을 실행합니다. | `resnet_50_mpk.tar.gz` 및 `yolo_v8s_mpk.tar.gz` |

Model Zoo의 출력 이름과 위치는 다를 수 있습니다. 필요한 경우, 다음 이름으로 아카이브 파일을 PCIe 추가 기능 루트 디렉터리에 복사하십시오.

<ShellCommand prompt="pcie-host">
cp /absolute/path/to/downloaded-resnet-archive.tar.gz resnet_50_mpk.tar.gz
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
</ShellCommand>

## 5. 경로와 예상 출력을 확인합니다.

추출된 추가 파일의 루트 디렉터리에서 튜토리얼 명령을 실행합니다. 해당 디렉터리에 다음 파일이 포함되어 있는지 확인합니다.
빌드 도우미, 미리 빌드된 C++ 프로그램, 튜토리얼 소스:

<ShellCommand prompt="sdk|pcie-host">
test -x build.sh
ls lib/*/tutorials/
ls share/*/tutorials/
</ShellCommand>

- 미리 빌드된 C++ 프로그램은 `lib/<package>/tutorials/`에 있습니다.
- C++ 및 Python 소스 코드는 `share/<package>/tutorials/`에 있습니다.
- `./build.sh --list-targets`는 다시 빌드할 수 있는 C++ 프로그램을 나열합니다.
- 성공적인 C++ 튜토리얼은 `[OK]`로 마무리됩니다. Python 튜토리얼은 간결한 내용을 출력합니다.
  예를 들어 다음과 같은 결과 `top1=...`, `completed=...`또는 `detections=...`.

튜토리얼에서 파일이 누락되었다는 메시지가 표시되면 먼저 현재 디렉터리와 모델 파일 이름을 확인하십시오. 추가 지원이 필요하면 다음을 참조하십시오. [문제 해결](/reference/troubleshooting/).
