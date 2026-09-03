---
title: "2단계 SDK 설치"
description: "SDK 2.0.0, 2.1.2 및 2.1.2.1 버전에 대한 기존의 2단계 설치 방식"
sidebar_position: 20
---

이 참고 자료는 별도의 이미지 다운로드 및 설정 명령이 필요한 SDK 릴리스에만 적용됩니다. 이는 SDK 2.0.0, SDK 2.1.2.0 및 SDK 2.1.2.1에 해당합니다. 최신 SDK 릴리스는 [환경을 설치합니다.](/getting-started/dev-environment/install-the-environment/)에 설명된 간소화된 패키지 설치 방식을 사용합니다.

기존 SDK 설치는 두 단계로 구성됩니다.

1. SDK 컨테이너 이미지를 가져옵니다.
2. SDK 설정을 실행하여 작업 공간을 만들고, 선택적으로 DevKit와 페어링합니다.

## SDK 이미지를 가져옵니다.

호스트에서 해당 SDK 릴리스와 일치하는 이미지 설치 명령을 실행합니다.

SDK 2.0.0의 경우:

<ShellCommand prompt="host">
sima-cli install ghcr:sima-neat/sdk:v2.0.0
</ShellCommand>

SDK 2.1.2 또는 2.1.2.1을 사용하는 경우 해당 릴리스와 함께 제공되는 일치하는 2.1 이미지 태그를 사용하십시오.

처음 설치하는 경우 SDK 컨테이너 이미지를 다운로드하므로 몇 분 정도 걸릴 수 있습니다.

## SDK 설정 실행

DevKit이 호스트에서 접근 가능한 경우, 설정 중에 페어링하세요.

<ShellCommand prompt="host">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

DevKit에 아직 연결할 수 없는 경우, 페어링하지 않고 SDK 작업 공간을 설정하세요.

<ShellCommand prompt="host">
sima-cli sdk setup
</ShellCommand>

설정 과정에서 `sima-cli`는 설치된 SDK 이미지를 선택하고, 호스트 작업 공간 디렉터리를 선택하고, SDK 확장 디렉터리를 구성하도록 요청할 수 있습니다. `--devkit`를 사용하면 설정 과정에서 DevKit 연결 정보도 요청하고 DevKit 동기화를 구성합니다.

## SDK 셸을 엽니다.

설치가 완료되면 SDK 셸을 엽니다.

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

## 호환성

SDK 2.0.0은 DevKit 소프트웨어 2.0.0용으로 제작되었습니다. SDK 2.1.2 및 2.1.2.1은 DevKit 소프트웨어 2.1.2용으로 제작되었습니다. 최신 호환성 정보는 [호환성](/getting-started/compatibility/)를 참조하십시오.
