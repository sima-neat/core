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
sima-cli neat install core@v%%release.core%% -t extras
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

Windows에서는 호스트 패키지에 튜토리얼이 이미 포함되어 있습니다. [Windows에서 애플리케이션 실행](/getting-started/neat-library/pcie-host/#run-applications-on-windows)에 설명된 대로 설정하세요.

**Ubuntu 22.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu22/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

**Ubuntu 24.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu24/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

PCIe PyNeat를 확인합니다.

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

## 4. 모델 아카이브를 준비합니다.

튜토리얼에서 사용하는 모델을 [Model Browser](/tools/model-browser/)에서 다운로드하고 각 ZIP 파일에서 아카이브를 추출하세요.

<ShellCommand prompt="sdk|devkit|pcie-host">
sima-cli models download --id resnet_50 --variant w8a8 --output models
sima-cli models download --id yolov8s --variant w8a8 --output models
unzip models/resnet_50/w8a8/*.zip resnet_50_w8a8_mpk.tar.gz
unzip models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
</ShellCommand>

다운로드한 파일이 ZIP 파일이 아니라 `<id>_<variant>_mpk.tar.gz` 파일인 경우(Model Browser가 모델을 ZIP 파일로 패키징하기 전에 마지막으로 빌드된 모델), 압축을 풀지 말고 `models/<id>/<variant>`에서 해당 파일을 복사하세요. `unzip`이 설치되어 있지 않으면 대신 `python3 -m zipfile -e models/yolov8s/w8a8/*.zip .`을 실행하고 `resnet_50`도 같은 방법으로 실행하세요. ZIP 파일의 모든 파일이 현재 디렉터리에 압축 해제됩니다.

Neat Library 튜토리얼은 `--model`을 허용하므로 추출한 아카이브 파일을 직접 사용할 수 있습니다. PCIe 튜토리얼은 PCIe 추가 자료의 루트 디렉터리에 있는 고정된 파일 이름을 사용합니다.

| PCIe 튜토리얼 | 필수 모델 파일 |
| --- | --- |
| PCIe를 통해 첫 번째 모델을 실행하세요. | `yolo_v8s_mpk.tar.gz` |
| PCIe 추론을 비동기적으로 실행합니다. | `yolo_v8s_mpk.tar.gz` |
| 여러 모델을 실행합니다. | `resnet_50_mpk.tar.gz` 및 `yolo_v8s_mpk.tar.gz` |

PCIe 튜토리얼의 경우 PCIe 추가 자료의 루트 디렉터리에서 명령을 실행하고 아카이브 이름을 변경하세요.

<ShellCommand prompt="pcie-host">
mv resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
</ShellCommand>

Windows에서는 튜토리얼 복사본 디렉터리에서 다음 PowerShell 명령을 실행하세요.

```powershell
sima-cli models download --id resnet_50 --variant w8a8 --output models
sima-cli models download --id yolov8s --variant w8a8 --output models
Expand-Archive models\resnet_50\w8a8\*.zip -DestinationPath models\resnet_50\w8a8
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8
Copy-Item models\resnet_50\w8a8\resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

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
