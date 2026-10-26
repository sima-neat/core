---
title: "Neat 명령줄 인터페이스"
description: "neat 명령어를 사용하여 설치된 Neat 환경을 검사하고 업데이트합니다."
sidebar_position: 5
---

설치된 라이브러리는 `neat` 환경 명령을 제공합니다. SDK 또는 DevKit에서 실행하여 설치된 구성 요소 버전, 설치된 `sima-cli` 플레이북, 그리고 최신 아티팩트가 있는지 확인할 수 있습니다.

SDK에서 상태 출력에는 `$HOME/.insight-config/neat-port-map.json`의 Insight 호스트-포트 매핑 정보도 포함됩니다.

PyNeat 또는 Insight가 설치되어 있지 않아도 상태 보고는 계속됩니다. 버전 조회가
실패하면 해당 구성 요소는 `unknown`으로 표시되고, `neat`는 공급자의 진단을
표준 오류에 기록합니다. JSON 출력은 기존 스키마를 유지하며 사용할 수 없는
버전은 `null`로 보고합니다.

분할 설치에서는 `simaai-runner`와 `simaai-scheduler`를 표시합니다. DevKit에서는 설치된 패키지 데이터베이스, SDK에서는 캐시된 DEB 제어 메타데이터에서 버전을 읽습니다. 레거시 `neat-runtime`과 `neat-gst-plugins` 행은 해당 패키지가 있거나 분할 컴포넌트가 감지되지 않을 때 표시됩니다. JSON은 기존 `runtime`, `gstPlugins` 키를 유지하면서 `components.runner`, `components.scheduler`를 추가합니다. 없는 버전은 계속 `null`입니다.

<ShellCommand prompt="sdk|devkit">
neat
</ShellCommand>

레거시 설치의 출력 예:

```text
Neat Environment
  Mode               Neat SDK
  Sysroot            /opt/toolchain/aarch64/modalix
  Update check       online

Components
  Neat core              %%release.core%% channel=release latest=%%release.core%%
  PyNeat                 %%release.core%%
  neat-runtime           %%release.core%%
  neat-gst-plugins       %%release.core%%
  neat-insight           %%release.insight%% channel=release status=Running venv=/opt/neat-insight/venv
  Model Compiler         %%release.model_compiler%% run activate-model-compiler to activate

Exposed Ports
  Insight Web UI     https://10.0.0.22:9900

  Name               Protocol Host Port (Start) Host Port (End)
  ------------------ -------- ----------------- ---------------
  mainUI             tcp      9900              -
  metadataUDP        udp      9100              9179
  rtsp.tcp           tcp      8554              -
  videoUDP           udp      9000              9079
  videoUI            tcp      8081              -
  webRTC             udp      40000             40199
  webSSH             tcp      8022              -
```

## JSON 출력

자동화 및 도구 통합을 위해 JSON 형식을 사용하세요.

<ShellCommand prompt="sdk|devkit">
neat --json
</ShellCommand>

## 설치된 구성 요소 업데이트

감지된 채널에서 Neat Library 런타임, `neat-insight` 및 설치된 `sima-cli` 플레이북을 업데이트하려면 다음 명령을 실행하세요.

<ShellCommand prompt="sdk|devkit">
neat update
</ShellCommand>

Insight나 플레이북을 조회하거나 업데이트하지 않고 Neat Library 런타임만
업데이트하려면 다음 명령을 실행하세요.

<ShellCommand prompt="sdk|devkit">
neat update --core-only
</ShellCommand>

## 다음 단계

[모델을 작성하세요.](/compile-a-model/)을 계속 진행하여 Modalix용 모델을 준비합니다. 이 작업은 SDK 내에서 실행되며 별도의 DevKit가 필요하지 않습니다. 하드웨어에서 전체 애플리케이션을 실행하려면 [안녕하세요 Neat!](/develop-apps/hello-neat/minimal/)을 참조하십시오. 이 작업에는 페어링된 DevKit가 필요합니다.
