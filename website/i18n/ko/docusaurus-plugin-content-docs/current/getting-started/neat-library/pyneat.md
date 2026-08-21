---
title: "PyNeat을 설치하세요."
description: "사용자 지정 Python 가상 환경에 PyNeat 휠을 설치합니다."
sidebar_position: 4
---

:::note DevKit만 해당 — SDK 설치 시에는 건너
이 단계를 통해 PyNeat을 DevKit (또는 런타임 역할을 하는 독립적인 호스트)에 설정합니다. Neat SDK 내에서 작업 중인 경우 이 페이지를 건너뛰십시오.
:::

:::tip PyNeat은 이미 Neat Library와 함께 설치되어 있습니다.
PyNeat은 Neat Library와 함께 제공되며, Neat Library를 설치하면 자동으로 설치됩니다.

기본적으로 `~/pyneat`에 가상 환경으로 설치됩니다. 별도의 venv 또는 DevKit의 conda 환경과 같이 사용자 지정 가상 환경에 PyNeat을 설치하려는 경우가 아니라면 이 단계를 건너뛸 수 있습니다.
:::

아래 단계를 DevKit에서 실행합니다. 이 지침은 런타임 `.deb` 패키지를 설치하거나 업데이트하지 않으므로, 해당 Neat Library 런타임이 이미 설치된 위치에서 실행하십시오.

## 휠 다운로드

<ShellCommand prompt="devkit">
sima-cli neat install core -t pyneat
</ShellCommand>

특정 Neat Library 버전에 대한 휠 파일을 다운로드하려면 해당 버전 정보를 포함하세요.

특정 버전을 설치하려면:

<ShellCommand prompt="devkit">
sima-cli neat install core@v0.4.0 -t pyneat
</ShellCommand>

## 파이썬 환경을 설정합니다.

해당 환경의 `python3`를 사용하여 가상 환경을 생성하고 활성화합니다.

<ShellCommand prompt="devkit">
python3 -m venv ~/my-neat-env
source ~/my-neat-env/bin/activate
</ShellCommand>

## 휠 설치

<ShellCommand prompt="devkit">
pip install ./pyneat-*.whl
</ShellCommand>

지원되는 Neat Library, SDK 및 DevKit 소프트웨어 조합에 대한 자세한 내용은
[호환성 안내서](/getting-started/compatibility/)를 참조하십시오.

## 다음 단계

설치된 환경을 확인하기 위해 [Neat 명령줄 인터페이스](/getting-started/neat-library/neat-cli/)를 계속 실행합니다.
