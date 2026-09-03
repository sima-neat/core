---
title: "SDK 구성"
description: "설치 후에는 Neat SDK 작업 공간, DevKit 페어링 또는 설정 옵션을 변경하십시오."
sidebar_position: 4
---

:::tip SDK 설정을 변경할 때만 여기서부터 시작하세요.
SDK 설치 명령어는 이미 SDK 이미지를 다운로드하고 설정을 완료합니다. 작업 공간 위치, DevKit 페어링, SDK 확장 또는 기타 설정 옵션과 같이 SDK 설정을 변경해야 하는 경우 이 페이지를 사용하세요.

기존의 Neat SDK 컨테이너를 구성해야 할 때 `sima-cli sdk setup`을 직접 실행합니다.

## DevKit 페어링을 사용하여 구성

DevKit이 호스트에서 접근 가능한 상태이고, DevKit 페어링을 추가하거나 업데이트하려는 경우 이 명령어를 사용하세요.

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

다음 항목들을 준비해 두세요:

- DevKit의 IP 주소
- NFS를 설치하거나 구성해야 하는 경우 호스트 관리자 비밀번호
- 메시지가 표시되면 DevKit 사용자 계정 정보 입력. 기본 사용자는 `sima`이고 기본 비밀번호는 `edgeai`입니다.

`--devkit`를 사용하면 설정을 통해 DevKit Sync를 활성화할 수 있습니다. 이 기능을 사용하면 호스트 워크스페이스를 NFS를 통해 내보내고 기본적으로 DevKit에 `/workspace`로 마운트합니다.

## DevKit 페어링 없이 구성

SDK 설정을 업데이트하되 DevKit에 아직 연결할 수 없는 경우 이 명령을 사용하세요.

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

설정 과정에서 `sima-cli`가 다음 사항을 요청할 수 있습니다.

- 호스트 작업 공간 디렉터리를 선택합니다. 특별한 작업 공간이 필요하지 않은 경우 기본값을 그대로 사용합니다.
- SDK 확장 프로그램 디렉터리를 선택합니다.
- Model Compiler를 설치할지 여부를 선택합니다.

Model Compiler는 사용자가 직접 모델을 컴파일하거나 양자화하는 데 필요하며, 미리 컴파일된 모델 패키지만 사용하는 경우에만 선택 사항입니다. 모델을 컴파일하거나 양자화할 계획이라면 여기에서 설치하고, 메시지가 표시되면 `sima-cli login`을 완료하세요.

## 다음 단계

[DevKit Sync](/getting-started/dev-environment/devkit-sync/)를 통해 작업 공간 공유, 페어링 업데이트, rsync 백업, 그리고 `dk` 명령어 세부 사항을 확인하십시오.
