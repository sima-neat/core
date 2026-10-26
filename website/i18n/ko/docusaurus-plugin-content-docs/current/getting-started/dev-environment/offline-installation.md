---
title: 오프라인 설치
description: 대상 호스트에서 인터넷 연결 없이 SDK 및 Model Compiler 패키지 설치
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

런타임 패키지는 [PCIe 호스트 오프라인 설치](/getting-started/neat-library/pcie-host/#offline-installation) 또는 [Neat Library 오프라인 설치](/getting-started/neat-library/install-or-update/#offline-installation)를 참조하세요.

## 패키지 선택

| 패키지 | 용도 | 제공 상태 |
| --- | --- | --- |
| Neat SDK | SDK 개발 환경을 설정합니다. | SDK 2.1.3.0 이상 |
| Model Compiler | SDK 또는 독립 실행형 호스트에 Model Compiler를 설치합니다. | Model Compiler 2.1.3 이상 |

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

SDK와 함께 Model Compiler를 설치하려면 Model Compiler 오프라인 패키지 파일을 SDK 오프라인 패키지와 같은 디렉터리에 넣으세요.
SDK 설치 프로그램이 파일을 자동으로 찾습니다.
메시지가 표시되면 로컬 패키지에서 설치하도록 선택하세요.

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

## 내부에 패키지 저장 {#hosting-packages-internally}

승인된 패키지를 내부 파일 서버에 저장할 수 있습니다.
다운로드한 각 디렉터리를 온전히 유지하고 다른 패키지 버전과 별도로 보관하세요.
설치 프로그램을 실행하기 전에 전체 디렉터리를 대상 호스트로 전송하세요.
