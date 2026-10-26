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
  Neat core              0.4.0 channel=release latest=0.4.0
  PyNeat                 0.4.0
  neat-runtime           0.4.0
  neat-gst-plugins       0.4.0
  neat-insight           0.0.6 channel=release status=Running venv=/opt/neat-insight/venv
  Model Compiler         2.1.3 run activate-model-compiler to activate

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

## PCIe 호스트 및 카드 상태

`sima-pcie-host` 패키지는 Ubuntu에도 `neat`를 설치합니다. `neat` 또는 `neat --json`으로 호스트에 설치된 패키지, PyNeatPCIe 및 각 SiMa PCIe 카드의 보드와 설치된 소프트웨어를 확인합니다. 카드는 PCI 주소와 인터페이스로 표시됩니다. PCI 주소는 현재 연결을 나타내며 영구적인 보드 일련번호가 아닙니다.

```bash
neat
neat --json
neat --offline
```

카드 조회는 `pcie-setup.sh`와 동일하게 구성된 호스트 인터페이스 `10.0.N.1`과 카드 주소 `10.0.N.2`를 사용합니다. 네트워크 연결과 신뢰할 SSH 키를 먼저 설정하세요. 이 명령은 인터페이스 설정, 호스트 키 등록, 암호 입력, 카드에서의 sudo 실행 또는 패키지 변경을 수행하지 않습니다. 인터페이스 누락, 비표준 주소, SSH 오류 및 소프트웨어 정보 조회 실패는 카드별로 표시하며 연결 가능한 카드의 버전은 계속 표시합니다.

`NEAT_PCIE_SSH_USER`는 카드 SSH 사용자(기본값 `sima`)를 선택합니다. `NEAT_PCIE_SSH_KEY`는 개인 키(기본값 `~/.ssh/sima_neat_pcie_ed25519`)를 선택하며 기존 SSH 설정과 에이전트 인증도 사용할 수 있습니다. `--offline`은 인터넷 업데이트 확인을 생략하지만 로컬 SSH로 연결된 카드를 조회합니다. PCIe 호스트 상태에서는 아티팩트 업데이트를 확인하지 않습니다. 호스트 패키지를 업데이트하려면 `sima-cli neat install core/pciehost/<distro>/<arch>@<ref>`를 사용하세요. `neat update`는 PCIe 카드를 업데이트하지 않습니다.

JSON은 `sima.neat.status.v1`을 유지하며 `components.pcieHost`, `components.pcieScheduler`, `components.pyneatpcie` 및 `pcie`를 추가합니다. `pcie`에는 호스트 소프트웨어, PCI 주소·인터페이스·주소·상태·소프트웨어·오류 필드가 있는 `cards` 배열 및 토폴로지 오류가 포함됩니다. 원격 소프트웨어 정보를 읽을 수 없으면 설치된 버전을 추정하지 않고 `null`을 반환합니다.

PyNeat 및 PyNeatPCIe 상태는 감지된 각 설치의 버전과 가상 환경 경로를 표시합니다. 별도 환경은 버전 충돌이 아닌 여러 환경으로 표시합니다. 가상 환경 외부의 패키지는 패키지 디렉터리를 대신 표시합니다. 상태 조회는 네이티브 모듈을 가져오지 않고 패키지 메타데이터를 읽습니다.

JSON의 각 Python 항목에 `installations` 배열이 추가됩니다. 각 레코드에는 `version`, `environment`, `path`(패키지 디렉터리)가 포함됩니다. 가상 환경 외부에서는 `environment`가 `null`입니다. 설치된 버전이 다르면 요약 `version`은 `null`이며, 여러 설치가 있다는 이유만으로 `error`를 설정하지 않습니다. 같은 설치를 가리키는 심볼릭 링크 경로는 한 번만 표시합니다.
