---
title: "컨테이너 애플리케이션 개발"
description: "Palette SDK에서 컨테이너 이미지를 빌드하고 Modalix DevKit에서 실행합니다."
sidebar_position: 7
---

컨테이너는 애플리케이션과 애플리케이션에 필요한 라이브러리를 하나로 묶으므로, 해당 라이브러리를 DevKit 자체에 설치하지 않고도 DevKit에서 애플리케이션을 실행할 수 있습니다. Palette SDK를 사용하면 호스트에서 ARM64 컨테이너 이미지를 빌드하고, 로컬 컨테이너 레지스트리에 푸시한 다음, DevKit에서
`dk container`로 시작합니다.

## 컨테이너를 사용하는 경우

DevKit의 Arm CPU에서 실행되는 소프트웨어에 컨테이너를 사용하십시오. 예를 들면 다음과 같습니다.

- eLxr에서 제공하지 않는 라이브러리가 필요한 서비스 또는 도구.
- Ubuntu와 같은 다른 Linux 배포판용으로 빌드된 애플리케이션. 예를 들어 eLxr에서 사용할 수 있는 ROS 2 릴리스와 다른 ROS 2 릴리스용으로 빌드된 ROS 2 애플리케이션.
- 같은 라이브러리의 서로 다른 버전이 필요한 여러 애플리케이션.

:::warning 컨테이너에서는 MLA 또는 DSP를 사용할 수 없습니다
Ubuntu 컨테이너를 포함하여 컨테이너에서 MLA 또는 DSP를 사용하는 것은 지원되지 않습니다. 컨테이너의 애플리케이션은 DevKit의 Arm CPU에서만 실행됩니다. MLA 또는 DSP를 사용하는 Neat 애플리케이션은 [Neat Library](/getting-started/neat-library/)에 설명된 대로 DevKit에서 직접 실행하십시오.
:::

## 작동 방식

이 워크플로는 Palette SDK의 다른 작업과 동일한 세 위치를 사용합니다.

| 위치 | 수행되는 작업 |
| --- | --- |
| 호스트 | `sima-cli sdk setup --devkit`이 DevKit를 SDK와 페어링하고 호스트에 로컬 컨테이너 레지스트리를 설정합니다. |
| SDK 컨테이너 | `docker buildx`로 ARM64 이미지를 빌드하고 로컬 레지스트리에 푸시합니다. |
| DevKit | `dk container deploy`가 로컬 레지스트리에서 이미지를 다운로드하고 Docker로 시작합니다. |

DevKit에는 Docker가 기본 제공되지 않습니다. 처음 실행하는 `dk container` 명령이 Docker 설치를 제안합니다.

## 시작하기 전에

다음이 필요합니다.

- 호스트에 설치된 Palette SDK %%release.sdk_series%% 이상.
- 호스트에서 접근할 수 있는 플랫폼 버전 3.0의 Modalix DevKit. 버전을 확인하려면 DevKit에서 `cat /etc/buildinfo`를 실행하십시오.
- Docker를 다운로드하기 위한 DevKit의 인터넷 연결.
- Docker가 이미지와 컨테이너를 저장하는 DevKit의 `/data` 여유 공간. `/data`는 약 3 GB이므로 이미지를 작게 유지하십시오.

## 컨테이너 애플리케이션 빌드 및 실행

### 1. DevKit를 로컬 레지스트리와 페어링

호스트에서 DevKit를 SDK와 페어링합니다.

<ShellCommand prompt="host">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

설정 중에 **Set up a local container registry so the DevKit can download
images built in the SDK?** 질문이 표시되면 `y`로 응답하십시오. 페어링하면 DevKit 사용자에게 비밀번호 없는 `sudo` 권한도 부여되며, Docker 설치에 이 권한이 필요합니다. 그런 다음 새 SDK 셸을 엽니다.

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

이전에 레지스트리 없이 DevKit를 페어링했다면 설정 명령을 다시 실행하고 `y`로 응답하십시오.

### 2. DevKit에 Docker 설정

SDK 셸에서 다음을 실행합니다.

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

DevKit에 Docker를 설정할지 묻는 메시지가 표시되면 `y`로 응답하십시오. DevKit에서 Docker가 이미 작동하고 있으면 명령은 기존 설치를 유지합니다. 완료되면 `Docker is
ready on the DevKit`를 출력합니다. 설치로 변경되는 내용은
[DevKit에 Docker 설치](/reference/devkit-containers/#install)를 참조하십시오.

### 3. 이미지 빌드 및 푸시

애플리케이션용 `Dockerfile`이 들어 있는 디렉터리를 만듭니다. 예를 들어 다음은 Ubuntu 24.04에서 셸 스크립트를 실행하는
`Dockerfile`입니다.

```dockerfile
FROM ubuntu:24.04
COPY hello.sh /usr/local/bin/hello.sh
CMD ["sh", "/usr/local/bin/hello.sh"]
```

SDK 셸의 해당 디렉터리에서 ARM64용 이미지를 빌드하고 로컬 레지스트리에 푸시합니다.

<ShellCommand prompt="sdk">
docker buildx build --platform linux/arm64 -t "$SIMA_CONTAINER_REGISTRY/my-app:latest" --push .
</ShellCommand>

DevKit는 ARM64 이미지만 실행합니다. 항상
`--platform linux/arm64`로 빌드하십시오.

### 4. DevKit에서 이미지 실행

SDK 셸에서 이미지를 DevKit로 다운로드하고 시작합니다.

<ShellCommand prompt="sdk">
dk container deploy my-app:latest --detach --name my-app
</ShellCommand>

이미지 이름에 레지스트리 주소를 넣을 필요는 없습니다. `dk`가 주소를 추가합니다. 이미지 이름 뒤의 옵션은 `docker run` 옵션입니다.

컨테이너의 출력을 확인한 다음 컨테이너를 중지하고 제거합니다.

<ShellCommand prompt="sdk">
dk container logs my-app --follow
dk container stop my-app
dk container remove my-app
</ShellCommand>

## 다음 단계

- 모든 `dk container` 명령, Docker 수동 설치, Ubuntu 컨테이너 실행 및 문제 해결은
  [DevKit 컨테이너](/reference/devkit-containers/)를 참조하십시오.
- SDK와 DevKit가 파일을 공유하고 명령을 실행하는 방식은
  [DevKit Sync](/getting-started/dev-environment/devkit-sync/)를 참조하십시오.
