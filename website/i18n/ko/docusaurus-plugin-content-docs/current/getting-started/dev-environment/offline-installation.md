---
title: "오프라인 설치"
description: "제한된 네트워크 환경을 위한 SDK 및 Model Compiler 패키지를 다운로드하세요."
sidebar_position: 6
---

:::tip 오프라인 설치를 언제 사용해야 할까요?
대상 시스템이 SiMa.ai 서비스에서 직접 패키지를 다운로드할 수 없는 경우 오프라인 설치를 사용합니다. 이는 대규모 다운로드를 차단하거나, 레지스트리 접근을 제한하거나, 수동으로 아티팩트 승인을 요구하거나, 인터넷에 연결된 시스템을 개발 호스트와 분리하는 기업 네트워크에서 흔히 발생합니다.
:::

이 작업 흐름에서 인터넷에 연결된 장치를 사용하여 설치 패키지를 가져온 다음, 다운로드한 디렉터리를 USB 드라이브, 내부 파일 공유 또는 내부적으로 호스팅되는 패키지 위치를 통해 대상 호스트로 이동합니다.

## SDK 오프라인 패키지 다운로드

SiMa.ai 패키지 서비스를 사용할 수 있는 시스템에서 대상 호스트 아키텍처에 맞는 명령을 실행합니다.

`amd64` 호스트의 경우:

<ShellCommand prompt="online-machine">
sima-cli neat install sdk@v2.1.2.3 -t offline-amd64
</ShellCommand>

`arm64` 호스트의 경우:

<ShellCommand prompt="online-machine">
sima-cli neat install sdk@v2.1.2.3 -t offline-arm64
</ShellCommand>

다운로드한 디렉토리를 대상 호스트에 복사합니다. 해당 디렉토리에서 다음 명령을 실행합니다.

<ShellCommand prompt="offline-host">
bash ./install_offline_sdk.sh
</ShellCommand>

:::note
SDK 버전 2.1.2.3 이상에서는 SDK 오프라인 패키지를 지원합니다.
:::

## Model Compiler 오프라인 패키지를 다운로드하세요.

대상 환경 및 SDK 호환성 요구 사항에 맞는 Model Compiler 패키지를 다운로드합니다. 호환성에 대한 자세한 내용은 [호환성](/getting-started/compatibility/#model-compiler)를 참조하십시오.

`amd64` 호스트에서 Model Compiler 2.1.2를 사용하는 경우:

<ShellCommand prompt="online-machine">
sima-cli install -v 2.1.2 tools/model-compiler/amd64 -t offline -d ./model-compiler-offline-amd64
</ShellCommand>

`arm64` 호스트에서 Model Compiler 2.1.2를 사용하는 경우:

<ShellCommand prompt="online-machine">
sima-cli install -v 2.1.2 tools/model-compiler/arm64 -t offline -d ./model-compiler-offline-arm64
</ShellCommand>

:::note
Model Compiler의 오프라인 패키지는 Model Compiler 2.1.2 버전 이상에서 지원됩니다.
:::

Neat SDK 내에 Model Compiler를 설치하려면 다운로드한 디렉터리를 SDK 컨테이너의 `/workspace` 폴더에 매핑된 호스트 워크스페이스 폴더에 복사합니다. 그런 다음 SDK 셸을 열고 해당 `/workspace` 경로에서 설치 프로그램을 실행합니다.

<ShellCommand prompt="username@neat-sdk-latest">
cd /workspace/model-compiler-offline-amd64
bash ./install_modelsdk_wheels.sh
</ShellCommand>

ARM64 패키지를 다운로드한 경우 대신 `arm64` 디렉터리 이름을 사용하십시오.

독립 실행형 호스트 설치의 경우, 다운로드한 디렉터리를 대상 호스트에 직접 복사한 다음 해당 디렉터리에서 동일한 설치 프로그램을 실행합니다.

설치 후에는 셸 환경을 다시 로드하거나 SDK 셸을 다시 시작합니다. 그런 다음 다음 명령을 사용하여 Model Compiler를 활성화합니다.

<ShellCommand prompt="offline-host">
activate-model-compiler
</ShellCommand>

Model Compiler 환경을 종료하려면 다음 명령을 실행하세요.

<ShellCommand prompt="offline-host">
deactivate-model-compiler
</ShellCommand>

## 내부적으로 호스팅 패키지 제공

조직에서 승인된 아티팩트를 내부적으로 복제하는 경우, 내부 위치에 게시할 때 다운로드한 패키지 디렉터리를 그대로 유지하십시오. 메타데이터 파일, 설치 스크립트, 체크섬, 패키지 리소스는 함께 유지되어야 합니다.

사용자는 내부 패키지 디렉터리를 다운로드한 다음 대상 호스트에서 동일한 설치 스크립트를 실행할 수 있습니다.
