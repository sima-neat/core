---
title: 오프라인 설치
description: 대상 호스트에서 인터넷 연결 없이 SDK, Model Compiler 및 PCIe 패키지 설치
sidebar_position: 6
---

:::tip 오프라인 설치가 필요한 경우
대상 호스트가 SiMa.ai 서비스에서 패키지를 다운로드할 수 없는 경우 오프라인 설치를 사용하세요.
조직에서 다음과 같은 정책을 적용할 때 필요할 수 있습니다:

- 대용량 다운로드를 차단합니다.
- 패키지 레지스트리 액세스를 제한합니다.
- 다운로드한 패키지를 사용하기 전에 승인을 요구합니다.
- 개발 호스트를 인터넷에 연결된 컴퓨터와 분리합니다.
:::

인터넷에 연결된 컴퓨터에서 패키지를 다운로드하세요.
그런 다음 설치할 패키지를 대상 호스트로 전송하세요.

## 패키지 선택

| 패키지 | 용도 | 제공 상태 |
| --- | --- | --- |
| Neat SDK | SDK 개발 환경을 설정합니다. | SDK 2.1.3.0 이상 |
| Model Compiler | SDK 또는 독립 실행형 호스트에 Model Compiler를 설치합니다. | Model Compiler 2.1.3 이상 |
| Neat PCIe | 준비된 PCIe 호스트와 연결된 모든 호환 카드에 Neat를 설치합니다. | 0.6.0 이상 |
| Neat Library | 인터넷 연결 없이 core 패키지를 설치합니다. | 0.6.0 이상 |

## 시작하기 전에

- 인터넷에 연결된 컴퓨터에 `sima-cli`를 설치하세요. [환경 설치](/getting-started/dev-environment/install-the-environment/)를 참조하세요.
- 시스템과 호환되는 패키지 버전을 선택하세요. [호환성](/getting-started/compatibility/)을 참조하세요.
- 대상 호스트의 운영 체제와 프로세서 아키텍처를 확인하세요.
- 전송에 사용할 USB 드라이브 또는 내부 파일 공유를 준비하세요.

다운로드 컴퓨터와 대상 호스트는 서로 다른 프로세서 아키텍처를 사용할 수 있습니다.
**대상 호스트**에 맞는 패키지를 선택하세요.
스크립트, 메타데이터, 체크섬 및 종속성을 포함한 모든 패키지 파일을 함께 보관하세요.

## Neat SDK 설치 {#download-the-sdk-offline-package}

### 1. 패키지 다운로드

인터넷에 연결된 컴퓨터에서 해당하는 명령 하나를 실행하세요.
다음 예제는 SDK 2.1.3.0을 사용합니다.

`amd64` 대상 호스트의 경우:

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-amd64
</ShellCommand>

`arm64` 대상 호스트의 경우:

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-arm64
</ShellCommand>

### 2. 패키지 설치

1. 다운로드한 전체 디렉터리를 대상 호스트에 복사하세요.
2. 해당 디렉터리에서 터미널을 여세요.
3. 설치 프로그램을 실행하세요:

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

## Model Compiler 설치 {#download-the-model-compiler-offline-package}

### 1. 패키지 다운로드

[Model Compiler 호환성 요구 사항](/getting-started/compatibility/#model-compiler)을 확인하세요.
인터넷에 연결된 컴퓨터에서 해당하는 명령 하나를 실행하세요.
다음 예제는 Model Compiler 2.1.3을 사용합니다.

`amd64` 대상 호스트의 경우:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v2.1.3 -t offline
</ShellCommand>

`arm64` 대상 호스트의 경우:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v2.1.3 -t offline
</ShellCommand>

### 2. 패키지 설치

설치 위치를 하나 선택하세요.

**Neat SDK 내부:**

1. 다운로드한 전체 디렉터리를 SDK 컨테이너의 `/workspace`에 매핑된 호스트 작업 공간에 복사하세요.
2. SDK 셸을 여세요.
3. 패키지 디렉터리로 이동하세요. `amd64` 패키지의 경우 다음을 사용하세요:

<ShellCommand prompt="sdk">
cd /workspace/model-compiler-offline-amd64
</ShellCommand>

`arm64` 패키지의 경우 `/workspace/model-compiler-offline-arm64`를 사용하세요.

4. 설치 프로그램을 실행하세요:

<ShellCommand prompt="sdk">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

**독립 실행형 호스트:**

1. 다운로드한 전체 디렉터리를 대상 호스트에 복사하세요.
2. 해당 디렉터리에서 터미널을 여세요.
3. 설치 프로그램을 실행하세요:

<ShellCommand prompt="host">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

### 3. Model Compiler 활성화

1. 설치에 사용한 SDK 셸 또는 호스트 셸을 다시 시작하세요.
2. Model Compiler를 활성화하세요:

<ShellCommand prompt="sdk|host">
activate-model-compiler
</ShellCommand>

Model Compiler 환경을 종료하려면 다음을 실행하세요:

<ShellCommand prompt="sdk|host">
deactivate-model-compiler
</ShellCommand>

## PCIe용 Neat 설치

PCIe 패키지는 호스트와 연결된 모든 호환 카드에 Neat를 설치합니다.
이 설치에는 대상 시스템의 Docker 또는 인터넷 연결이 필요하지 않습니다.
호스트에서 각 카드로 연결되는 PCIe 네트워크가 정상적으로 작동해야 합니다.

### 1. 호스트 및 카드 준비

Neat 설치 프로그램을 실행하기 전에 다음 작업을 완료하세요:

- 호환되는 Platform 호스트 드라이버를 설치하세요.
- 각 카드에 맞는 Platform 3.0 펌웨어를 설치하세요.
- 각 카드의 PCIe 인터페이스에서 검색 기능을 설정하세요.
- 각 카드에 고유한 PCIe IPv4 주소를 할당하세요.
- 각 카드의 SSH 액세스 및 신뢰할 수 있는 SSH 호스트 키를 설정하세요.
- SSH 계정에 각 카드에서 `sudo`를 사용할 권한을 부여하세요.

Neat 설치 프로그램은 Platform 드라이버를 설치하거나 Platform 펌웨어를 변경하지 않습니다.
Platform 릴리스에 포함된 시스템 설정 지침을 사용하세요.

### 2. 패키지 다운로드

대상 호스트에 맞는 0.6.0 이상 버전을 선택하세요.
패키지는 `amd64` 또는 `arm64` 프로세서를 사용하는 Ubuntu 22.04 및 24.04 호스트를 지원합니다.

이 예제는 인터넷에 연결된 컴퓨터에서 0.6.0 버전을 다운로드합니다.
Ubuntu 24.04 `amd64` 대상 호스트의 경우 다음을 실행하세요:

<ShellCommand prompt="host">
sima-cli neat install 'core/pciehost/ubuntu24/amd64@v0.6.0' -t offline -d ./pcie-offline
</ShellCommand>

Ubuntu 22.04의 경우 `ubuntu24`를 `ubuntu22`로 바꾸세요.
`arm64` 호스트의 경우 `amd64`를 `arm64`로 바꾸세요.
이 명령은 패키지를 다운로드하고 압축을 풉니다. Neat를 설치하지는 않습니다.

### 3. 시스템 확인

1. 전체 `pcie-offline` 디렉터리를 대상 호스트에 복사하세요. 공백이 없는 경로를 사용하세요.
2. 압축을 푼 `neat-offline` 디렉터리를 여세요. 이 디렉터리에는 `install.py`가 있습니다.
3. 필요한 시스템 도구와 SSH 설정은 제공된 `INSTALL.md`를 읽어 확인하세요.
4. 검사를 실행하세요:

<ShellCommand prompt="host">
sudo python3 -B install.py --preflight-only
</ShellCommand>

설치 프로그램은 연결된 모든 카드를 자동으로 검색합니다. 카드 IP 주소를 제공하지 마세요.
검사는 임시 파일을 생성하지만 패키지를 설치하지는 않습니다.
설치 전에 보고된 오류를 수정하세요.

### 4. 설치 및 결과 확인

같은 디렉터리에서 다음 명령을 실행하세요:

<ShellCommand prompt="host">
sudo python3 -B install.py
</ShellCommand>

설치 프로그램은 패키지를 설치하기 전에 모든 카드를 확인합니다.
먼저 호스트를 설치하고 검증한 다음 각 카드를 차례로 처리합니다.
최종 요약에서 호스트와 모든 카드의 설치 및 검증이 성공했는지 확인하세요.
그런 다음 애플리케이션 인수 테스트를 실행하세요.

설치가 실패하면 표시된 경로의 진단 로그를 읽으세요.
설치 중 카드 연결이 실패하면 다시 시도하기 전에 해당 카드의 설치 프로세스가 중지되었는지 확인하세요.
설치 프로그램은 실패 후 이전 설치 상태를 자동으로 복원하지 않습니다.

## Neat Library 설치

Neat Library 오프라인 패키지는 0.6.0 이상 버전에서 사용할 수 있습니다.
Platform 3.0 이상을 사용하는 Modalix DevKit에 이 패키지를 사용하세요.

### 1. 패키지 다운로드

인터넷에 연결된 컴퓨터에서 다음 명령을 실행하세요:

<ShellCommand prompt="host">
sima-cli neat install core@v0.6.0 -t offline --download-only
</ShellCommand>

기본 패키지에는 런타임, 튜토리얼 및 테스트가 포함됩니다.
런타임만 다운로드하려면 다음을 사용하세요:

<ShellCommand prompt="host">
sima-cli neat install core/offline@v0.6.0 -t minimal --download-only
</ShellCommand>

### 2. 패키지 설치

1. 다운로드한 파일을 Modalix DevKit 또는 Neat SDK 작업 공간에 복사하세요.
2. 해당 디렉터리에서 터미널을 여세요. SDK에 설치하려면 SDK 셸을 사용하세요.
3. 설치 프로그램 아카이브의 압축을 푸세요:

<ShellCommand prompt="sdk|devkit">
tar -xzf neat-core-offline-installer.tar.gz
</ShellCommand>

4. 압축을 푼 디렉터리로 이동하세요:

<ShellCommand prompt="sdk|devkit">
cd neat-core-offline
</ShellCommand>

5. 일반 사용자로 설치 프로그램을 시작하세요. 필요한 경우 `sudo` 권한을 요청합니다.

<ShellCommand prompt="sdk|devkit">
./install.sh
</ShellCommand>

튜토리얼이나 테스트가 필요하면 별도의 추가 아카이브를 원하는 디렉터리에 압축 해제하세요.

### 패키지 옵션 및 설치 프로그램 동작

- 설치 프로그램은 설치 전에 시스템 호환성과 사용 가능한 디스크 공간을 확인합니다.
- 설치 프로그램은 포함된 종속성을 사용합니다. 필요한 종속성이 없으면 중지합니다.
- SDK에서는 Neat Library를 `/opt/toolchain/aarch64/modalix`에 복사합니다.
- `DEVKIT_SYNC_DEVKIT_IP`가 설정되어 있으면 SSH를 통해 페어링된 DevKit에도 설치합니다. 두 환경 모두 검사를 통과해야 합니다.

설치가 실패하면 표시된 경로의 로그를 읽으세요: `/tmp/neat-core-offline-*.log`.
자세한 출력을 보려면 `./install.sh --verbose`를 사용하세요.

설치 중 SSH 연결이 끊어져도 원격 설치 프로그램은 계속 실행될 수 있습니다.
다시 시도하거나 보고서에 표시된 임시 디렉터리를 삭제하기 전에 설치 프로그램이 중지되었는지 확인하세요.

## 내부에 패키지 저장 {#hosting-packages-internally}

승인된 패키지를 내부 파일 서버에 저장할 수 있습니다.
다운로드한 각 디렉터리를 온전히 유지하고 다른 패키지 버전과 별도로 보관하세요.
설치 프로그램을 실행하기 전에 전체 디렉터리를 대상 호스트로 전송하세요.
