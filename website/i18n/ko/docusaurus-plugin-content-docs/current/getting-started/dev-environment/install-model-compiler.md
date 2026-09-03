---
title: "Model Compiler를 설치합니다."
description: "Model Compiler를 Neat SDK에 설치하거나 지원되는 독립 실행형 호스트에 설치합니다."
sidebar_position: 5
---

:::tip Model Compiler를 별도로 설치할 때만 여기서부터 시작하세요.
Model Compiler는 SDK 설치 중에 선택적으로 설치할 수 있습니다. 해당 메시지를 건너뛰었거나, 최신 버전의 호환되는 Model Compiler를 설치하려는 경우, 또는 지원되는 호스트에서 SDK 외부로 Model Compiler를 설치해야 하는 경우에만 이 페이지를 사용하십시오.
:::

Model Compiler는 ONNX 모델을 양자화하고 컴파일하여 SiMa.ai의 MLA에서 실행할 수 있도록 합니다. GenAI 모델을 포함하여 사용자가 직접 모델을 컴파일하거나 양자화할 때는 **필수**이며, 미리 컴파일된 모델 패키지만 사용하는 경우에만 **선택 사항**입니다.

SDK 설치/설정 과정에서 `sima-cli`는 Neat SDK 내부에 해당 모델 컴파일러를 확장 프로그램으로 설치하도록 안내합니다. 또한 나중에 Neat SDK 컨테이너 내부 또는 지원되는 Ubuntu 호스트에 독립적으로 설치할 수도 있습니다. 지원되는 버전 조합 및 독립 실행형 호스트 요구 사항은 [호환성](/getting-started/compatibility/#model-compiler)를 참조하십시오.

## SDK 내부에 설치

SDK 설정 중에 Model Compiler를 건너뛰었다면, 나중에 Neat SDK 내에서 설치하십시오. 사용 중인 Neat SDK 컨테이너 아키텍처에 맞는 명령을 실행합니다. 확인하려면 SDK 셸 내에서 `uname -m`을 실행하십시오. `aarch64`는 `arm64` 명령을 사용해야 함을 의미하고, `x86_64`는 `amd64` 명령을 사용해야 함을 의미합니다.

`amd64` Neat SDK 컨테이너의 경우:

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli neat install model-compiler/amd64@v2.1.3
</ShellCommand>

`arm64` Neat SDK 컨테이너의 경우:

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli neat install model-compiler/arm64@v2.1.3
</ShellCommand>

설치 후에는 Neat SDK 셸 내에서 컴파일러 환경을 활성화합니다.

<ShellCommand prompt="username@neat-sdk-latest">
activate-model-compiler
</ShellCommand>

기본 Neat SDK 셸로 돌아가려면 다음 명령을 실행하세요.

<ShellCommand prompt="username@neat-sdk-latest">
deactivate-model-compiler
</ShellCommand>

## 독립형 호스트에 설치

독립 실행 방식으로 설치하는 것은 [호환성](/getting-started/compatibility/#model-compiler)에 나열된 호스트 환경에서만 지원됩니다. 지원되는 호스트 환경에서 해당 `sima-cli neat install` 명령을 실행합니다. 호스트 아키텍처를 확인하려면 `uname -m` 명령을 실행합니다. `x86_64`는 `amd64` 명령을 사용하고, `aarch64`는 `arm64` 명령을 사용합니다.

`amd64` 호스트에서 Model Compiler 2.1.3을 사용하는 경우:

<ShellCommand prompt="user-host-machine">
sima-cli neat install model-compiler/amd64@v2.1.3
</ShellCommand>

`arm64` 호스트에서 Model Compiler 2.1.3을 사용하는 경우:

<ShellCommand prompt="user-host-machine">
sima-cli neat install model-compiler/arm64@v2.1.3
</ShellCommand>

`amd64` 호스트에서 Model Compiler 2.0.0을 사용하는 경우:

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.0.0 tools/model-compiler/amd64
</ShellCommand>

## 다음 단계

[모델을 작성하세요.](/compile-a-model/)을 계속 진행하십시오.
