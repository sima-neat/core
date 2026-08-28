---
title: "Neat SDK"
description: "빠르고 효율적인 에이전트 기반 Neat 애플리케이션 개발을 위해 Neat SDK를 설정하세요."
sidebar_position: 1
---

Neat 개발 환경(Neat SDK로 지칭)은 대규모 Neat 애플리케이션을 구축하고 Modalix DevKit에서 이를 검증하기 위한 권장 호스트 측 작업 공간입니다. 이 환경은 빌드 도구, 모델 도구, 하드웨어 연결, 그리고 에이전트가 사용할 수 있는 소스 컨텍스트를 하나의 컨테이너화된 워크플로로 통합합니다.

SDK는 세 곳을 연결합니다.

- **호스트 머신:** SDK 컨테이너를 설치하고 실행하는 위치입니다.
- **SDK 컨테이너:** 애플리케이션을 빌드하고, 모델을 컴파일하고, 에이전트 도구를 사용하고, 공유 파일을 검사하는 위치입니다.
- **Modalix DevKit:** 컴파일된 모델 아티팩트와 Neat 애플리케이션이 하드웨어에서 실행되는 위치입니다.

DevKit Sync는 해당 위치들을 공유된 `/workspace`로 연결하여 빌드 결과물, 로그, 모델 아티팩트 및 애플리케이션 파일을 호스트, SDK 컨테이너 및 DevKit에서 수동으로 복사하는 단계 없이 확인할 수 있도록 합니다. 이 공유된 작업 공간은 SDK 워크플로우의 중심입니다.

<div class="overview-section-label">여기서 시작하세요.</div>

SDK 설치부터 시작합니다. 설정을 변경하거나, 나중에 Model Compiler를 추가하거나, DevKit Sync의 작동 방식을 이해하거나, 제한된 네트워크를 위한 오프라인 패키지를 준비해야 할 때에는 다른 SDK 관련 내용을 참고하십시오.

:::tip SDK 전용 간편 설정
SDK를 설치하고 DevKit와 페어링하지 않은 경우, 다음 두 단계를 수행하면 됩니다.
[환경을 설치하세요.](/getting-started/dev-environment/install-the-environment/), 그런 다음 [모델을 작성하세요.](/compile-a-model/) — 모델 컴파일이 SDK 내에서 완전히 실행됩니다. SDK를 구성하고, DevKit Sync를 설정하고, Model Compiler를 설치합니다(설치 과정에서 제공됨). Neat Library 및 PyNeat 페이지는 선택 사항이며, 필요할 때만 방문하거나 DevKit와 페어링한 후에 방문하십시오.
:::

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>SDK 관련 주제</h2>
    <p>SDK를 설치한 후, 설정을 변경하거나 Model Compiler를 설치하거나 페어링된 DevKit를 사용할 때 필요한 경우 선택적인 구성 옵션을 사용하세요.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/install-the-environment/"><strong>환경을 설치하세요.</strong><span>사용 중인 DevKit 소프트웨어 버전에 맞는 SDK 패키지를 설치하고 설정합니다.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/devkit-sync/"><strong>DevKit Sync</strong><span>작업 공간 공유, 페어링 업데이트, rsync 백업 방식 등을 이해합니다. <code>dk</code> 명령어 실행.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/offline-installation/"><strong>오프라인 설치</strong><span>제한된 네트워크 환경을 위한 SDK 및 Model Compiler 패키지를 다운로드하세요.</span></a></li>
    </ul>
  </section>
</div>

설치 후 작업 공간 위치 또는 DevKit 페어링과 같이 SDK 설정을 변경하려면 [SDK 구성](/getting-started/dev-environment/configure-sdk/)를 참조하십시오.

Model Compiler는 SDK 설정 중에 제공됩니다. 나중에 설치하려면 특정 버전을 고정하거나 독립 실행형 호스트를 사용하십시오. 자세한 내용은 [Model Compiler를 설치합니다.](/getting-started/dev-environment/install-model-compiler/)를 참조하십시오.

패키지를 직접 다운로드할 수 없는 호스트의 경우, 다음을 참조하십시오.
[오프라인 설치](/getting-started/dev-environment/offline-installation/).

## 포함 내용

- **크로스 컴파일 환경:** 호스트의 Linux 컨테이너에서 C++ Neat 애플리케이션을 Modalix용으로 빌드합니다.
- **DevKit Sync:** SDK를 Modalix DevKit와 연결하고 동일한 작업 공간을 양쪽에 노출합니다.
- **모델 도구:** SDK에 해당 Model Compiler를 설치합니다. ONNX 또는 GenAI 모델을 직접 컴파일하거나 양자화하려면 필수이며, 사전 컴파일된 모델 패키지만 사용하는 경우 선택 사항입니다.
- **Insight:** 브라우저에서 작업 공간 파일, 미디어 소스, 스트림 전송 및 런타임 동작을 검사합니다.
- **에이전트 준비 컨텍스트:** 현재 Neat 소스 참조 및 예제와 함께 번들된 Codex 및 Claude 기능을 사용합니다.

## 호스트 요구 사항

SDK를 설치하기 전에 호스트 시스템이 다음 최소 요구 사항을 충족하는지 확인하십시오.
SDK를 모든 지원되는 호스트에 설치하려면 관리자(`sudo`) 권한이 필요합니다. 이는 선택 사항인 DevKit 네트워킹뿐만 아니라 `sima-cli`, Docker Engine, SDK 이미지 및 NFS 패키지를 설치할 때 모두 더 높은 수준의 권한이 필요하기 때문입니다.

| 호스트 OS | CPU | RAM | 사용 가능한 디스크 공간 | 관리자 / sudo |
|---|---|---|---|---|
| Ubuntu 22.04 / 24.04 (`x86_64` 또는 `arm64`) | 최소 4코어 | 최소 16GB | 100GB | SDK 설치(`sima-cli`, Docker, SDK 이미지), NFS 설치/구성, 공유 네트워크/방화벽 설정에 필요한 `sudo` 권한 |
| Windows 11 (WSL 사용) (`x86_64`) | 최소 4코어 | 최소 16GB | 100GB | WSL 내 SDK 설치(Docker, `sima-cli`), WSL 네트워킹, NFS 방화벽 규칙에 필요한 관리자 권한 |
| macOS 15.5+ Apple Silicon (`arm64`) | 최소 4코어 | 최소 16GB | 100GB | SDK 설치(Homebrew, Colima, `sima-cli`), 전체 디스크 접근 권한(`nfsd`), 인터넷 공유에 필요한 관리자 권한 |

:::note GenAI 모델 컴파일에는 더 많은 리소스가 필요합니다.
LLiMa를 사용하여 GenAI 모델을 컴파일하는 작업은 기본 SDK보다 훨씬 더 많은 리소스를 소모합니다. 128GB의 RAM을 권장하며, 512GB의 디스크 공간을 사용하는 것이 좋고, 더 많은 코어를 사용하면 도움이 됩니다.
전체 요구 사항은 [GenAI 설정](/genai-llima/setup/)를 참조하십시오.

## 지원되는 플랫폼

| 플랫폼 | 아키텍처 | SDK | Model Compiler |
|---|---|---|---|
| Docker Engine을 통해 Ubuntu 22.04 및 24.04 | `x86_64` | 예 | 예 |
| WSL 및 Docker Engine을 통해 Windows 11 | `x86_64` | 예 | 예 |
| Docker Engine을 통해 Ubuntu 22.04 및 24.04 | `arm64` | 예 | Model Compiler 2.1.2 이상 |
| Colima를 통해 macOS 15.5 이상 | `arm64` | 예 | Model Compiler 2.1.2 이상; Neat SDK 내부에 설치 |

:::note 아키텍처 이름
`arm64`와 `aarch64`는 동일한 64비트 Arm 아키텍처입니다. macOS에서는 `arm64`로, Linux에서는 `aarch64`로 보고됩니다. 마찬가지로 `x86_64`와 `amd64`는 동일한 아키텍처입니다. 호스트(또는 SDK 내부)에서 `uname -m`을 실행하여 어떤 아키텍처를 사용하는지 확인하십시오. Model Compiler 설치 명령어는 `arm64`와 `amd64`를 사용합니다. 자세한 내용은 [Model Compiler를 설치합니다.](/getting-started/dev-environment/install-model-compiler/)를 참조하십시오.

:::note 특정 버전 설치
기본 설치는 현재 지원되는 기본값을 사용합니다. `release-2.1` 채널은 [환경을 설치하세요.](/getting-started/dev-environment/install-the-environment/) 페이지에서 항상 최신 2.1 패치 릴리스를 추적합니다. 정확한 SDK, Neat Library 또는 Model Compiler 버전을 지정하려면 [호환성 안내서](/getting-started/compatibility/)을 참조하십시오.
:::

## SDK에 포함된 도구

SDK는 페어링된 DevKit용 애플리케이션을 개발할 때 [Neat Library](/getting-started/neat-library/)를 설치하고 업데이트할 수 있는 권장 위치입니다.

터미널, VS Code 또는 브라우저에서 SDK에 액세스하려면 Neat Insight를 사용하여 [환경을 설치하세요.](/getting-started/dev-environment/install-the-environment/#access-the-sdk)를 참조하십시오.

SDK 설정 과정에서 `sima-cli`는 해당하는 Model Compiler를 자동으로 설치하도록 안내합니다. 직접 모델을 컴파일하거나 양자화하는 경우 설치하고, 미리 컴파일된 모델 패키지만 사용하는 경우에는 설치를 건너뛸 수 있습니다. 컴파일러 설정 및 사용 방법에 대해서는 [모델을 작성하세요.](/compile-a-model/)를 참조하십시오.

## 다음 단계

[환경을 설치하세요.](/getting-started/dev-environment/install-the-environment/)부터 시작하세요.
