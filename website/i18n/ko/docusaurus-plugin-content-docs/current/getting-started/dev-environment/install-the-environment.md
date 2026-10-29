---
title: "SDK 설치"
description: "Palette SDK 컨테이너 설치 및 설정"
sidebar_position: 2
---

최신 Palette SDK를 설치합니다. Palette SDK는 호스트 시스템에서 컨테이너로 실행됩니다.

## 사전 요구 사항

- 호스트가 설치에 필요한 관리자/`sudo` 권한을 포함하여 [호스트 요구 사항](/getting-started/dev-environment/#host-requirements)을 충족하는지 확인합니다.
- [sima-cli 설치 가이드](/tools/sima-cli/)에 따라 호스트 시스템에 `sima-cli`를 설치합니다.
- 사용하는 플랫폼에 맞는 호스트 설정을 완료합니다.
  - [Ubuntu 호스트 참고 사항](/reference/elxr-sdk-host-setup/ubuntu)
  - [Windows 호스트 참고 사항](/reference/elxr-sdk-host-setup/windows)
  - [macOS 호스트 참고 사항](/reference/elxr-sdk-host-setup/macos)

나중에 DevKit Sync를 사용하려면 다음 항목도 필요합니다.

- Palette SDK와 호환되는 소프트웨어를 실행하는 DevKit. 자세한 내용은 [호환성 가이드](/getting-started/compatibility/)를 참조하십시오.
- NFS 트래픽이 허용되는 동일 네트워크에 연결된 호스트 시스템과 DevKit.
- DevKit IP 주소.

## 설치

현재 Palette SDK %%release.sdk_series%% 릴리스 채널을 설치합니다.

<ShellCommand prompt="host">
sima-cli neat install sdk@%%release.sdk_channel%%
</ShellCommand>

처음 설치할 때는 Palette SDK 컨테이너 이미지를 다운로드하므로 몇 분이 걸릴 수 있습니다. 이미지 다운로드가 완료되면 설치 프로그램이 SDK 설정을 시작하고 Modalix DevKit과 페어링할지, SDK 내부에 호환되는 Model Compiler를 설치할지 묻습니다.

DevKit 페어링을 선택한 경우 메시지가 표시되면 DevKit IP 주소를 입력합니다. 설정 과정에서 SDK 작업 공간을 구성하고 SDK 컨테이너를 시작한 뒤 DevKit Sync를 구성합니다. 페어링을 건너뛰어도 SDK 작업 공간은 생성되며 나중에 페어링할 수 있습니다.

`%%release.sdk_channel%%` 패키지는 %%release.sdk_series%% 계열의 최신 Palette SDK 패치 릴리스를 추적합니다. 현재 릴리스는 Palette SDK %%release.sdk%%이며 DevKit 소프트웨어 %%release.platform%%과 호환됩니다.

설정 중 `sima-cli`는 호환되는 Model Compiler를 SDK 내부에 설치할지도 묻습니다. 모델을 직접 컴파일하거나 양자화한다면 수락하십시오. 별도로 버전을 선택할 필요가 없습니다. 사전 컴파일된 모델 패키지만 실행한다면 건너뛸 수 있습니다. 나중에 설치하거나 특정 패치를 고정하거나 독립 실행형 호스트를 사용하려면 [Model Compiler 설치](/getting-started/dev-environment/install-model-compiler/)와 [호환성 가이드](/getting-started/compatibility/)를 참조하십시오.

### 특정 SDK 버전 설치 {#install-a-specific-sdk-version}

SDK 2.1.3.0 이상에서는 게시된 `v<version>` 태그를 사용하여 이전 패치 릴리스를 포함한 특정 버전을 설치할 수 있습니다. 예를 들어 호스트에서 다음 명령을 실행하십시오.

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0
</ShellCommand>

`v` 접두사는 필수입니다. 특정 버전 태그에 `release-<full-version>`을 사용하지 마십시오. `release-<series>` 채널은 해당 시리즈의 최신 게시 릴리스를 추적합니다. 동일한 SDK 패키지를 선택하면 두 방식 모두 동일한 설치 및 SDK 설정 단계를 실행합니다.

[SDK 릴리스](https://github.com/sima-neat/sdk/releases)에서 게시된 태그를 확인하고, [호환성 가이드](/getting-started/compatibility/)에 따라 DevKit 소프트웨어와 호환되는 버전을 선택하십시오.

:::note 레거시 2단계 설치
SDK 2.0.0, 2.1.2.0 및 2.1.2.1에서는 이미지 가져오기와 설정에 별도의 명령이 필요합니다. 이 릴리스에만 [2단계 SDK 설치](/reference/two-step-sdk-installation/)를 사용하십시오.
:::

설치 후 SDK 설정을 변경하려면 [SDK 구성](/getting-started/dev-environment/configure-sdk/)을 참조하십시오. 제한된 네트워크 환경에서는 [오프라인 설치](/getting-started/dev-environment/offline-installation/)를 참조하십시오.

## SDK에 접속

설정이 완료되면 터미널, Chrome 브라우저 또는 VS Code에서 SDK에 접속할 수 있습니다.

### SDK 셸 사용

다음 명령으로 Palette SDK 셸을 엽니다.

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

### Chrome 브라우저 사용

Neat Insight는 SDK 내부에서 제공되며 브라우저에서 열 수 있습니다. SDK 셸 안에서 다음을 실행합니다.

<ShellCommand prompt="sdk">
neat
</ShellCommand>

`neat` 출력의 **Web Access** 섹션에는 Insight와 브라우저 기반 VS Code의 로컬 및 원격 URL이 표시됩니다. SDK 호스트에서 브라우저를 실행하는 경우 로컬 `127.0.0.1` URL을 여십시오. 다른 컴퓨터에서는 `NFS_SERVER_HOST_IP`를 기반으로 하는 원격 URL을 사용하십시오. VS Code URL은 구성된 작업 공간을 엽니다. 설치된 서버에 대해 `neat`가 표시하는 인증 정보를 사용하십시오. [SDK 웹 액세스](/reference/environment-variables/)와 [Insight](/tools/insight/)를 참조하십시오.

### VS Code 사용

SDK 셸에서 `neat`를 실행하고 SDK 호스트의 브라우저에서 **VS Code (local)** URL을 여십시오. 다른 컴퓨터에서는 **VS Code (remote)**를 사용하십시오. code-server가 포함된 이미지에서는 `neat`가 **VS Code password**도 표시합니다.

```text
VS Code (local)    https://127.0.0.1:10000/?folder=/workspace
VS Code password   <password shown by neat>
```

code-server 로그인 화면에 표시된 암호를 입력하십시오. 암호는 URL에 포함되지 않습니다.

레거시 OpenVSCode 이미지는 대신 토큰 URL을 사용합니다. `neat`가 표시하는 URL을 여십시오. SDK에는 브라우저 Code UI용 Codex 및 Claude Code 확장이 사전 설치되어 있습니다.

브라우저 대신 네이티브 VS Code를 사용할 수도 있습니다. [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)를 사용해 VS Code를 SDK 컨테이너에 연결합니다.

SDK가 사전 빌드 모델과 같은 자산을 가져올 수 있도록 SDK 컨테이너 안에서 `sima-cli login`을 한 번 실행하십시오.

## 업그레이드

현재 SDK 패키지를 다시 설치하거나 업그레이드하려면 호스트에서 위의 설치 명령을 다시 실행합니다.

<ShellCommand prompt="host">
sima-cli neat install sdk@%%release.sdk_channel%%
</ShellCommand>

기존 Palette SDK 컨테이너 내부의 Neat 라이브러리를 업데이트하려면 컨테이너 셸에서 Neat CLI를 실행합니다.

<ShellCommand prompt="sdk">
neat update
</ShellCommand>

이 명령은 현재 Palette SDK에 설치된 Neat 라이브러리 구성 요소를 업데이트합니다. 컨테이너 수준 변경이 필요한 경우 전체 컨테이너 이미지 업그레이드를 대신하지는 않습니다.

나중에 Palette SDK 컨테이너를 삭제하거나 다시 만들면 새 컨테이너 안에서 `neat update`를 다시 실행하십시오.

## 제거

설치된 SDK 컨테이너를 제거하려면 다음을 실행합니다.

<ShellCommand prompt="host">
sima-cli sdk remove
</ShellCommand>

## 다음 단계

- **SDK에서 모델을 컴파일합니까?** DevKit 없이 SDK 안에서 완전히 실행되는 [모델 컴파일](/compile-a-model/)로 이동하십시오.
- **DevKit을 페어링합니까?** SDK 작업 공간을 공유하고 하드웨어에서 `dk` 명령을 실행하려면 [DevKit Sync](/getting-started/dev-environment/devkit-sync/)를 설정하십시오.
