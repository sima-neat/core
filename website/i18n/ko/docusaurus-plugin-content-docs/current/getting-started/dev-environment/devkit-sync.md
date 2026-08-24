---
title: "DevKit Sync"
description: "Modalix DevKit와 함께 SDK 작업 공간을 공유하고 하드웨어에서 명령을 실행합니다."
sidebar_position: 3
---

:::tip 참고용으로만 사용하십시오.
SDK 설치 중에 DevKit Sync 설정을 선택적으로 진행할 수 있으며, 설치 프로그램에서 이미 해당 설정을 묻는 메시지를 표시합니다. 이 페이지를 참고하여 DevKit Sync의 작동 방식을 확인하거나, 나중에 SDK 페어링을 다른 DevKit로 변경하려는 경우에 활용하십시오.
:::

DevKit Sync는 Neat 개발 환경(Neat SDK로 지칭)을 동일 네트워크의 Modalix DevKit와 연결합니다. 이를 통해 호스트, Neat SDK 컨테이너, DevKit 전반에 걸쳐 하나의 공유 작업 공간을 제공하며, 하드웨어에서 SDK 기반 명령을 실행하기 위한 `dk` 헬퍼를 제공합니다.

![호스트, 컨테이너 및 DevKit 작업 공간 매핑](@site/../docs/images/elxr-sdk-workspaces.svg)

동일한 작업 공간이 Neat SDK 컨테이너와 DevKit에 `/workspace`로 마운트되므로, 빌드 아티팩트, 로그, 추적 정보 및 모델 파일이 각 환경에서 확인할 수 있습니다.

## DevKit Sync 구성

설치 중에 DevKit 페어링을 건너뛰었거나 나중에 변경해야 하는 경우, 호스트에서 다음 설정 명령을 실행하세요.

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

설정 중:

- 둘 이상의 SDK 이미지가 있는 경우 설치된 `sdk:v2.1-latest` 이미지를 선택합니다.
- 다른 경로가 필요하지 않은 경우 기본 `/workspace` DevKit 마운트 경로를 사용합니다.
- 호스트에서 NFS 서버를 설정하라는 메시지가 표시되면 호스트 관리자 비밀번호를 입력합니다.
- 메시지가 표시되면 DevKit 사용자 계정을 입력합니다. 기본 사용자는 `sima`이고 기본 비밀번호는 `edgeai`입니다.
- SDK 2.1.2.2부터 호스트와 DevKit 간에 NFS를 구성할 수 없는 경우 설정이 SSH를 통한 rsync로 대체될 수 있습니다.

설정이 성공하면 다음과 유사한 결과가 표시됩니다.

```text
============================================================
  DevKit Connected
============================================================
  DevKit target : sima@192.168.91.221:22
  Mounted path  : /workspace
  Host export   : 192.168.74.48:/Users/joey/workspace

  You can now run DevKit binaries from this SDK shell:
    dk /workspace/<path-to-arm64-binary> [args...]
============================================================
```

SDK 셸 내에서 `dk status`를 사용하여 페어링된 DevKit과 활성 상태의 작업 공간 동기화 방법을 확인합니다.

<ShellCommand prompt="username@neat-sdk-latest">
dk status
</ShellCommand>

## SDK에서 페어링을 업데이트하세요.

Neat SDK가 이미 설치되어 있고 다른 DevKit를 페어링하거나 DevKit의 IP 주소가 변경된 후 페어링을 업데이트해야 할 때 이 단계를 따르세요.

Neat SDK 컨테이너 내부에서 다음 명령을 실행합니다.

<ShellCommand prompt="username@neat-sdk-latest">
source devkit.sh {devkit-ip}
</ShellCommand>

사용하려는 DevKit의 IP 주소로 `{devkit-ip}`를 대체하십시오.

예시:

<ShellCommand prompt="username@neat-sdk-latest">
source devkit.sh 192.168.91.221
</ShellCommand>

## DevKit Sync 없이 SDK 구성

Neat SDK 호스트에서 DevKit에 연결할 수 없는 경우에도 페어링하지 않고 SDK 작업 공간을 구성할 수 있습니다.

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

Neat SDK 컨테이너에서 여전히 실행 파일을 빌드할 수 있지만, 테스트를 위해 해당 파일을 DevKit로 수동으로 전송해야 합니다. DevKit에서 호환되는 Neat Library 버전을 실행하고 있는지 확인하십시오.

## DevKit Sync를 사용한 파일 공유

DevKit Sync는 세 가지 환경을 연결합니다.

1. 호스트
2. Neat SDK 컨테이너
3. DevKit

`sima-cli sdk setup --devkit {devkit-ip}`는 NFS를 구성하여 동일한 작업 공간이 세 가지 환경 모두에서 사용 가능하도록 합니다.

- 호스트 워크스페이스 폴더는 호스트 NFS를 통해 내보내집니다.
- 해당 폴더는 Neat SDK 컨테이너에 `/workspace`로 마운트됩니다.
- 동일한 콘텐츠가 NFS를 통해 DevKit의 `/workspace`에 표시됩니다.
- 마운트된 폴더의 기본 이름은 `/workspace`이며, 설정 중에 변경할 수 있습니다.

이 구성은 빌드 아티팩트에 대한 직접적인 작업 흐름을 제공합니다.

- Neat SDK에서 생성된 아티팩트는 별도의 배포 단계 없이 DevKit에서 즉시 확인할 수 있습니다.
- 에이전트는 DevKit에서 앱이 실행되는 동안 생성된 로그, 출력, 추적 정보 및 기타 임시 파일에 액세스할 수 있습니다.
- 개발자와 에이전트는 동일한 작업 공간에서 동일한 파일을 검사할 수 있습니다.

Insight를 사용하면 웹 브라우저에서 작업 공간을 볼 수 있습니다. 일부 SiMa.ai 관련 모델 아카이브(예: `*.tar.gz` 모델 아티팩트)는 자동으로 최적화되어 검토가 더 용이하도록 지원됩니다.

## Rsync 대체 방안

SDK 2.1.2.2부터 DevKit Sync는 NFS 설정이 실패할 경우 대체 수단으로 SSH를 통한 rsync를 사용할 수 있습니다. 이는 SSH를 통해 DevKit에 접속할 수 있지만, 호스트 NFS 내보내기를 DevKit에서 마운트할 수 없는 네트워크 또는 호스트에서 유용합니다.

rsync 대체 모드가 활성화된 경우:

- 호스트와 SDK 컨테이너는 여전히 로컬 `/workspace` 디렉터리를 사용합니다.
- DevKit은 동기화된 원격 작업 공간을 사용하며, 일반적으로 `/workspace-rsync`를 사용합니다.
- `dk status`는 `Sync method : rsync`를 보고 로컬 및 원격 작업 공간 경로를 표시합니다.
- `dk <file> [args...]`는 SDK 작업 공간의 경로를 DevKit rsync 작업 공간에 매핑한 다음, 원격에서 명령을 실행합니다.
- `dk`가 파일을 실행하기 전에 해당 파일을 포함하는 최상위 작업 공간 폴더를 자동으로 동기화합니다. 예를 들어, `dk apps/demo.py`는 DevKit 측 복사를 실행하기 전에 `/workspace/apps` 폴더를 동기화합니다.

현재 페어링 및 동기화 방법을 확인하세요.

<ShellCommand prompt="username@neat-sdk-latest">
dk status
</ShellCommand>

현재 작업 영역 범위를 수동으로 동기화합니다.

<ShellCommand prompt="username@neat-sdk-latest">
dk sync
</ShellCommand>

특정 파일 또는 폴더 범위를 동기화합니다.

<ShellCommand prompt="username@neat-sdk-latest">
dk sync /workspace/apps
</ShellCommand>

전체 작업 공간을 동기화합니다.

<ShellCommand prompt="username@neat-sdk-latest">
dk sync --all
</ShellCommand>

rsync 폴백이 활성화된 경우, 하나의 `dk` 명령에 필요한 파일을 동일한 최상위 워크스페이스 폴더 아래에 보관합니다. 명령이 `/workspace/apps`에서 실행되는 경우, `/workspace/models`를 가리키는 인수는 자동 동기화 범위에서 제외되므로 `dk sync
/workspace/models`를 사용하여 별도로 동기화해야 합니다. 또는 필요한 파일이 동일한 최상위 폴더 아래에 있도록 프로젝트를 구성해야 합니다.

## DevKit에서 dk를 사용하여 실행

SDK에는 ARM64 실행 파일을 SDK 셸 내에서 페어링된 DevKit에서 실행하기 위한 `dk` 헬퍼, 즉 `devkit-run`이 포함되어 있습니다.

`dk`를 호출하면 SDK가 페어링된 DevKit에서 명령을 실행하고, 컨테이너의 파일 인수가 DevKit에서 올바르게 해석되도록 경로를 변환합니다.

<ShellCommand prompt="username@neat-sdk-latest">
dk <file> [args...]
</ShellCommand>

SDK 작업 공간에서 C++ 애플리케이션을 컴파일한 후, 생성된 ARM64 실행 파일을 DevKit에서 실행합니다.

<ShellCommand prompt="username@neat-sdk-latest">
dk build/sima_neat_hello
</ShellCommand>

SDK 작업 공간에 Python 스크립트를 생성하거나 복사한 후, 페어링된 DevKit에서 실행합니다.

<ShellCommand prompt="username@neat-sdk-latest">
dk hello_neat.py
</ShellCommand>

Python 스크립트의 경우, `dk`는 페어링된 DevKit에서 스크립트를 실행하고,
DevKit의 PyNeat 런타임 환경을 사용합니다. SDK는 통합된 작업 공간 및 오케스트레이션 환경으로서 유용성을 유지하지만, Python 전용 워크플로는 C++ 크로스 컴파일 도구 체인이 필요하지 않습니다.

:::note `dk`는 어디에서 왔을까요?
`dk`는 SDK 컨테이너 내의 `~/devkit-sync.rc`에 정의된 셸 함수입니다.
셸은 `~/.bashrc`를 통해 이 함수를 로드하므로, 대화형 세션에서 사용할 수 있습니다.
:::

## 다음 단계

라이브러리/런타임 자체를 설치하거나 업데이트하려면 [Neat Library](/getting-started/neat-library/)로 이동하세요.
