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

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-amd64
</ShellCommand>

`arm64` 호스트의 경우:

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-arm64
</ShellCommand>

다운로드한 디렉토리를 대상 호스트에 복사합니다. 해당 디렉토리에서 다음 명령을 실행합니다.

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

:::note
SDK 버전 2.1.3.0 이상에서는 SDK 오프라인 패키지를 지원합니다.
:::

## Model Compiler 오프라인 패키지를 다운로드하세요.

대상 환경 및 SDK 호환성 요구 사항에 맞는 Model Compiler 패키지를 다운로드합니다. 호환성에 대한 자세한 내용은 [호환성](/getting-started/compatibility/#model-compiler)를 참조하십시오.

`amd64` 호스트에서 Model Compiler 2.1.3을 사용하는 경우:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v2.1.3 -t offline
</ShellCommand>

`arm64` 호스트에서 Model Compiler 2.1.3을 사용하는 경우:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v2.1.3 -t offline
</ShellCommand>

:::note
Model Compiler의 오프라인 패키지는 Model Compiler 2.1.3 버전 이상에서 지원됩니다.
:::

Neat SDK 내에 Model Compiler를 설치하려면 다운로드한 디렉터리를 SDK 컨테이너의 `/workspace` 폴더에 매핑된 호스트 워크스페이스 폴더에 복사합니다. 그런 다음 SDK 셸을 열고 해당 `/workspace` 경로에서 설치 프로그램을 실행합니다.

<ShellCommand prompt="sdk">
cd /workspace/model-compiler-offline-amd64
bash ./install_modelsdk_wheels.sh
</ShellCommand>

ARM64 패키지를 다운로드한 경우 대신 `arm64` 디렉터리 이름을 사용하십시오.

독립 실행형 호스트 설치의 경우, 다운로드한 디렉터리를 대상 호스트에 직접 복사한 다음 해당 디렉터리에서 동일한 설치 프로그램을 실행합니다.

설치 후에는 셸 환경을 다시 로드하거나 SDK 셸을 다시 시작합니다. 그런 다음 다음 명령을 사용하여 Model Compiler를 활성화합니다.

<ShellCommand prompt="sdk|host">
activate-model-compiler
</ShellCommand>

Model Compiler 환경을 종료하려면 다음 명령을 실행하세요.

<ShellCommand prompt="sdk|host">
deactivate-model-compiler
</ShellCommand>

## 내부적으로 호스팅 패키지 제공

조직에서 승인된 아티팩트를 내부적으로 복제하는 경우, 내부 위치에 게시할 때 다운로드한 패키지 디렉터리를 그대로 유지하십시오. 메타데이터 파일, 설치 스크립트, 체크섬, 패키지 리소스는 함께 유지되어야 합니다.

사용자는 내부 패키지 디렉터리를 다운로드한 다음 대상 호스트에서 동일한 설치 스크립트를 실행할 수 있습니다.

## 오프라인 Core 패키지

인터넷에 연결된 컴퓨터에서 다운로드한 다음 준비된 Modalix DevKit에 아카이브를 복사합니다. 기본 프로필에는 런타임 번들과 튜토리얼 및 테스트 추가 패키지가 포함됩니다. 추가 패키지를 제외하려면 `-t minimal`을 사용합니다. 대상 설치 프로그램이 OS, 아키텍처 및 Platform 호환성을 확인합니다.

```bash
sima-cli neat install core/offline@v0.6.0 --download-only
sima-cli neat install core/offline@v0.6.0 -t minimal --download-only
```

`neat-core-offline-installer.tar.gz`를 추출한 후 `cd neat-core-offline && ./install.sh`를 실행합니다. 튜토리얼이나 테스트가 필요하면 별도의 추가 아카이브를 원하는 디렉터리에 추출하세요.

설치 프로그램은 Neat eLxr SDK 또는 프로비저닝된 Modalix DevKit만 허용합니다. `/etc/sdk-release` 또는 `/etc/buildinfo`를 아카이브에 포함된 패키지 매니페스트와 비교하므로 다운로드한 배포 메타데이터는 필요하지 않습니다. 다른 환경이나 호환되지 않는 Platform 버전에서는 설치 전에 중단합니다. SDK에서는 Core를 `/opt/toolchain/aarch64/modalix`에 복사합니다. `DEVKIT_SYNC_DEVKIT_IP`가 설정되어 있으면 SSH를 통해 동일한 오프라인 번들을 페어링된 DevKit에 전송합니다. 설치를 시작하기 전에 두 환경 모두 사전 검사를 통과해야 합니다. DevKit에서는 로컬로 설치합니다.

표준 SDK 컨테이너의 오프라인 설치 프로그램에는 `unshare`, 특권 모드 또는 추가 컨테이너 권한이 필요하지 않습니다. 패키지를 sysroot에 추출하며 로컬 APT `file`, `copy`, `store` 메서드만 활성화합니다. HTTP, HTTPS 및 다른 네트워크 전송은 비활성화되며 의존성이 누락되면 온라인 대체 경로 없이 실패합니다. 네이티브 DevKit 설치는 격리된 네트워크 네임스페이스를 유지합니다.

네이티브 설치 프로그램은 오프라인 APT 트랜잭션 계획에 따라 설치 또는 재설치할 Debian 패키지와 모든 Python wheel의 압축 해제 크기의 3배에 더해 메타데이터와 Python 환경을 위한 128 MiB를 확보합니다. APT가 변경하지 않는 종속 패키지는 이 설치 공간 계산에서 제외합니다. NVMe 또는 tmpfs의 번들을 포함하여 이미 배치된 아카이브는 다시 계산하지 않습니다. 공간이 부족하면 루트 파일 시스템의 사용 가능한 공간, 필요한 공간, 추가로 필요한 공간을 MiB 단위로 표시합니다.

페어링된 DevKit으로 전송할 때는 표준 DevKit에서 RAM 기반인 `/tmp` 아래의 전용 디렉터리를 사용합니다. 설치 실패가 보고되거나 전송이 완료되지 않은 경우를 포함하여 작업이 끝나면 스테이징 디렉터리를 삭제합니다. 설치 중 SSH 연결이 끊기거나 시간 초과가 발생하면 원격 설치 프로그램이 계속 실행 중일 수 있으므로 디렉터리를 유지하고 경로를 표시합니다.

일반 SDK 또는 DevKit 사용자로 `./install.sh`를 실행합니다. 실행 스크립트는 `sudo` 권한을 자동으로 요청하고 페어링, DevKit 인증, 동기화 기본 설정 및 SSH 에이전트 설정만 명시적으로 유지합니다. 페어링된 장치 배포에는 실행한 SDK 사용자의 SSH 자격 증명을 계속 사용합니다.

설치 프로그램은 전송, 호환성 확인, SDK 설치, 페어링된 DevKit 설치 및 정리 단계를 번호와 함께 표시합니다. 터미널에서는 색상을 사용하고 오래 걸리는 단계에서 15초마다 실행 중 메시지를 표시합니다. APT, 패키지 및 호환성 상세 보고서는 비공개 `/tmp/neat-core-offline-*.log` 파일에 저장하며 정확한 경로를 표시하고 실패 시에도 보존합니다. 색상 없이 출력하려면 `NO_COLOR=1 ./install.sh`를 사용합니다. 반복되는 호환성 확인은 패키지 트랜잭션을 보호하기 위한 것으로 설치 작업과 별개입니다.
