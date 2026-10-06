---
title: "개발 환경 설치"
description: "Neat SDK 컨테이너 설치 및 설정"
sidebar_position: 2
---

최신 Neat 개발 환경(이하 Neat SDK)을 설치합니다. Neat SDK는 호스트 시스템에서 컨테이너로 실행됩니다.

## 사전 요구 사항

- 호스트가 설치에 필요한 관리자/`sudo` 권한을 포함하여 [호스트 요구 사항](/getting-started/dev-environment/#host-requirements)을 충족하는지 확인합니다.
- [sima-cli 설치 가이드](/tools/sima-cli/)에 따라 호스트 시스템에 `sima-cli`를 설치합니다.
- 사용하는 플랫폼에 맞는 호스트 설정을 완료합니다.
  - [Ubuntu 호스트 참고 사항](/reference/elxr-sdk-host-setup/ubuntu)
  - [Windows 호스트 참고 사항](/reference/elxr-sdk-host-setup/windows)
  - [macOS 호스트 참고 사항](/reference/elxr-sdk-host-setup/macos)

나중에 DevKit Sync를 사용하려면 다음 항목도 필요합니다.

- Neat SDK와 호환되는 소프트웨어를 실행하는 DevKit. 자세한 내용은 [호환성 가이드](/getting-started/compatibility/)를 참조하십시오.
- NFS 트래픽이 허용되는 동일 네트워크에 연결된 호스트 시스템과 DevKit.
- DevKit IP 주소.

## 설치

현재 Neat SDK 2.1 릴리스 채널을 설치합니다.

<ShellCommand prompt="host">
sima-cli neat install sdk@release-2.1
</ShellCommand>

처음 설치할 때는 Neat SDK 컨테이너 이미지를 다운로드하므로 몇 분이 걸릴 수 있습니다. 이미지 다운로드가 완료되면 설치 프로그램이 SDK 설정을 시작하고 Modalix DevKit과 페어링할지, SDK 내부에 호환되는 Model Compiler를 설치할지 묻습니다.

DevKit 페어링을 선택한 경우 메시지가 표시되면 DevKit IP 주소를 입력합니다. 설정 과정에서 SDK 작업 공간을 구성하고 SDK 컨테이너를 시작한 뒤 DevKit Sync를 구성합니다. 페어링을 건너뛰어도 SDK 작업 공간은 생성되며 나중에 페어링할 수 있습니다.

`release-2.1` 패키지는 2.1 계열의 최신 Neat SDK 패치 릴리스를 추적합니다. 현재 릴리스는 Neat SDK 2.1.3.0이며 DevKit 소프트웨어 2.1.3과 호환됩니다.

설정 중 `sima-cli`는 호환되는 Model Compiler를 SDK 내부에 설치할지도 묻습니다. 모델을 직접 컴파일하거나 양자화한다면 수락하십시오. 별도로 버전을 선택할 필요가 없습니다. 사전 컴파일된 모델 패키지만 실행한다면 건너뛸 수 있습니다. 나중에 설치하거나 특정 패치를 고정하거나 독립 실행형 호스트를 사용하려면 [Model Compiler 설치](/getting-started/dev-environment/install-model-compiler/)와 [호환성 가이드](/getting-started/compatibility/)를 참조하십시오.

:::note 이전 SDK 릴리스는 레거시 2단계 설치 절차를 사용합니다
별도의 이미지 가져오기 및 설정 명령이 필요한 이전 SDK 릴리스에 대해서는 [2단계 SDK 설치](/reference/two-step-sdk-installation/)를 참조하십시오.
:::

설치 후 SDK 설정을 변경하려면 [SDK 구성](/getting-started/dev-environment/configure-sdk/)을 참조하십시오. 제한된 네트워크 환경에서는 [오프라인 설치](/getting-started/dev-environment/offline-installation/)를 참조하십시오.

## SDK에 접속

설정이 완료되면 터미널, Chrome 브라우저 또는 VS Code에서 SDK에 접속할 수 있습니다.

### SDK 셸 사용

다음 명령으로 Neat SDK 셸을 엽니다.

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

### Chrome 브라우저 사용

Neat Insight는 SDK 내부에서 제공되며 브라우저에서 열 수 있습니다. SDK 셸 안에서 다음을 실행합니다.

<ShellCommand prompt="sdk">
neat
</ShellCommand>

명령 출력의 **Web Access** 섹션에는 Insight 및 브라우저 기반 VS Code의 로컬·원격 URL이 표시됩니다. 로컬 접속은 SDK를 실행하는 동일한 시스템의 브라우저에서 URL을 여는 것이고, 원격 접속은 다른 시스템에서 여는 것입니다. 브라우저와 SDK가 같은 호스트에 있다면 호스트 네트워크 IP가 바뀌어도 동작하는 로컬 `127.0.0.1` URL을 권장합니다. 원격 접속에는 `NFS_SERVER_HOST_IP`에서 파생된 URL을 사용합니다. VS Code URL에는 설정된 액세스 토큰이 포함되며 구성된 작업 공간을 엽니다. 자세한 내용은 [Insight](/tools/insight/)를 참조하십시오.

### VS Code 사용

SDK Code UI를 통해 브라우저에서 VS Code를 사용할 수 있습니다. SDK 설치가 끝나면 `sima-cli`가 다음과 같은 `codeUI` URL을 출력합니다.

<ShellCommand prompt="host">
codeUI      | https://192.168.76.4:10000/?tkn=gA5CS...&folder=/workspace
</ShellCommand>

브라우저에서 이 URL을 열어 SDK 작업 공간에서 작업합니다. SDK에는 브라우저 Code UI용 Codex 및 Claude Code 확장이 미리 설치되어 있습니다.

브라우저 대신 네이티브 VS Code를 사용할 수도 있습니다. [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)를 사용해 VS Code를 SDK 컨테이너에 연결합니다.

SDK가 사전 빌드 모델과 같은 자산을 가져올 수 있도록 SDK 컨테이너 안에서 `sima-cli login`을 한 번 실행하십시오.

## 업그레이드

현재 SDK 패키지를 다시 설치하거나 업그레이드하려면 호스트에서 위의 설치 명령을 다시 실행합니다.

<ShellCommand prompt="host">
sima-cli neat install sdk@release-2.1
</ShellCommand>

기존 Neat SDK 컨테이너 내부의 Neat 라이브러리를 업데이트하려면 컨테이너 셸에서 Neat CLI를 실행합니다.

<ShellCommand prompt="sdk">
neat update
</ShellCommand>

이 명령은 현재 Neat SDK에 설치된 Neat 라이브러리 구성 요소를 업데이트합니다. 컨테이너 수준 변경이 필요한 경우 전체 컨테이너 이미지 업그레이드를 대신하지는 않습니다.

나중에 Neat SDK 컨테이너를 삭제하거나 다시 만들면 새 컨테이너 안에서 `neat update`를 다시 실행하십시오.

## 제거

설치된 SDK 컨테이너를 제거하려면 다음을 실행합니다.

<ShellCommand prompt="host">
sima-cli sdk remove
</ShellCommand>

## 다음 단계

- **SDK에서 모델을 컴파일합니까?** DevKit 없이 SDK 안에서 완전히 실행되는 [모델 컴파일](/compile-a-model/)로 이동하십시오.
- **DevKit을 페어링합니까?** SDK 작업 공간을 공유하고 하드웨어에서 `dk` 명령을 실행하려면 [DevKit Sync](/getting-started/dev-environment/devkit-sync/)를 설정하십시오.
