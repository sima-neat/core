---
title: "개발 프로세스"
description: "SiMa.ai의 Neat 개발 프로세스를 개략적으로 살펴보겠습니다. 설치부터 배포까지의 과정을 설명합니다."
sidebar_position: 4
---

# 개발 프로세스

이 페이지는 SiMa.ai와 Neat을 일상적으로 어떻게 사용하는지에 대한 개략적인 안내입니다. 각 단계는 자세한 내용을 담은 다른 페이지로 연결되며, 필요할 때 해당 페이지를 확인할 수 있습니다.

## 더 루프

일반적인 Neat 개발 주기는 다음과 같습니다.

1. **설치** — `sima-neat` 패키지와 선택적으로 `pyneat` Python 바인딩을 호스트 또는 장치에 설치합니다.
2. **Hello Neat 사용해 보세요** — 간단한 예제 코드를 컴파일하여 라이브러리가 제대로 연결되었는지 확인합니다.
3. **컴파일된 모델을 선택하세요.** — Neat은 컴파일된 모델 아카이브(`.tar.gz`, 일반적으로 MPK라고 함)를 사용합니다. Model Zoo에서 모델을 선택하거나 Model Compiler를 사용하여 직접 컴파일할 수 있습니다.
4. **`Model` / `Graph` / `Run` 작성** — 모델을 로드하고, 그래프를 구성한 다음, 해당 작업에 가장 적합한 최소 런타임 경로를 사용하여 실행합니다.
5. **실행 및 검사** — 입력을 제공하고, 출력을 확인하며, 동작을 검증하기 위해 `GraphReport` / `MeasureReport`를 사용합니다.
6. **튜토리얼을 통해 반복 학습** — 단일 추론에서 파이프라인, 다중 입력 모델, 다중 스트림 그래프, 그리고 프로덕션 수준의 오류 처리로 발전합니다.
7. **배포** — 대상 장치에 설치된 Neat 라이브러리와 애플리케이션을 연결합니다.

## 모델, 그래프 또는 실행을 선택하세요.

가장 작은 면적부터 시작하여 문제를 해결하십시오. 앱이 더 많은 기능을 갖추게 되면 언제든지 기능을 확장할 수 있습니다.

| 만약 필요한 것이 있다면... | 다음부터 시작하세요. | 왜 | 다음 정류장 |
| --- | --- | --- | --- |
| 컴파일된 모델을 한 번 실행합니다. | `Model.run(...)` | 아티팩트에서 출력 텐서까지의 최단 경로입니다. | [첫 번째 모델 실행](/tutorials/run-your-first-model) |
| 연결된 Modalix PCIe 카드에서 모델을 실행합니다. | `pcie::Model` / `pyneatpcie.Model` | 별도의 온카드 애플리케이션을 구축하지 않고 호스트 측에서 공동 처리 작업을 수행합니다. | [PCIe 공동 처리](/develop-apps/development-workflow/pcie-model) |
| 표준 계약서 또는 경로를 검토하십시오. | `Model` | 사양, 메타데이터, 경로 정보는 모델이 어떤 입력을 받아 어떤 출력을 생성하는지 알려줍니다. | [모델](/develop-apps/development-workflow/model) |
| 애플리케이션 흐름에 모델을 추가합니다. | `Graph` | 입력과 출력을 명명하고, 노드를 구성하며, 토폴로지를 명확하게 유지합니다. | [그래프](/develop-apps/development-workflow/graph) |
| 시간이 지남에 따라 그래프를 재사용합니다. | `graph.build(...)` → `Run` | 푸시/풀, 개방/차단 제어, 측정, 대기열 정책 기능을 제공합니다. | [그래프 실행](/develop-apps/development-workflow/pipeline) |
| 여러 스트림을 동시에 처리하거나 최대 처리량을 달성하도록 합니다. | `RunOptions` + 측정 | 처리량 최적화를 위해서는 대기열 정책, 패킷 손실 카운터, 그리고 스트림별 증거 자료가 필요합니다. | [처리량 및 큐 깊이 조정](/tutorials/tune-throughput-and-queues) |

이것이 처음으로 직접 실습하는 과정이라면, [튜토리얼](/tutorials) 사전 점검 목록부터 시작한 다음, 그래프 관련 기능을 추가하기 전에 먼저 하나의 모델을 실행해 보세요.

## 핵심 개념을 한눈에

개발 워크플로 페이지에서는 이러한 각 단계를 자세히 설명합니다. 간략하게 살펴보면 다음과 같습니다.

- [모델](/develop-apps/development-workflow/model) — 컴파일된 모델 패키지를 로드하고 실행 가능한 단위로 제공합니다.
- [PCIe 공동 처리](/develop-apps/development-workflow/pcie-model) — 호스트 애플리케이션에서 연결된 Modalix PCIe 카드에서 컴파일된 모델을 실행합니다.
- [생성형 AI 모델](/develop-apps/development-workflow/genai-model) — 생성 모델을 직접 실행하거나 HTTP를 통해 제공합니다.
- [텐서 및 샘플](/develop-apps/development-workflow/core_types) — 각 단계 간에 전달되는 데이터와 메타데이터.
- [실행 / 추론](/develop-apps/development-workflow/overview) — 동기적으로 실행(`run`)하거나 비동기적으로 실행(`push` / `pull`).
- [그래프](/develop-apps/development-workflow/graph) — 모델의 단계, 노드, 입력 및 출력을 조합하여 애플리케이션 그래프를 구성합니다.
- [그래프 실행](/develop-apps/development-workflow/pipeline) — 그래프를 실시간 `Run`에 통합한 후, 데이터를 전송하고, 가져오고, 처리하고, 측정하고, 처리량을 조정합니다.
- [노드](/develop-apps/development-workflow/node)는 그래프를 구성하는 기본적인 단위입니다.

먼저 한 페이지만 학습하려는 경우, [실행/추론 개요](/develop-apps/development-workflow/overview)부터 시작하세요. 이 페이지는 `Model`, `Graph`, 그리고 `Run`을 처음부터 끝까지 연결하여 설명합니다.

## 다음으로 어디를 갈까요?

새로운 사용자를 위한 단계별 안내:

- [Neat SDK](/getting-started/dev-environment/) — Neat SDK를 설치하고, DevKit와 페어링한 후, `dk`가 설치된 하드웨어에서 실행합니다.
- [구축하다](/develop-apps/contribute/build) — 소스 코드를 사용하여 Neat을 빌드합니다. `build.sh`를 사용합니다(기여자 워크플로).
- [안녕하세요 Neat!](/develop-apps/hello-neat/minimal) — 설치된 라이브러리와 연결되는 최소한의 CMake 애플리케이션입니다.
- [튜토리얼](/tutorials) — “첫 번째 모델”부터 “실제 파이프라인”까지 단계별로 안내하는 튜토리얼입니다.

더 깊이 있는 정보를 원할 때 참고할 자료:

- [실행 / 추론](/develop-apps/development-workflow/overview) — `Model`, `Graph`, `Run`, 노드, 그리고 입/출력(I/O)을 개념별로 나누어 설명.
- [C++ 레퍼런스](/reference/cppapi) — 설치된 헤더 파일에 대한 전체 API 인터페이스입니다.
- [파이썬 레퍼런스](/reference/pythonapi) — `pyneat` 바인딩 참조.

## 당신이 작성하는 내용과 Neat가 제공하는 내용

Neat은 런타임을 소유합니다. 여기에는 모델 로딩, 검증, 파이프라인 구성, 스케줄링, 정리, 진단 등이 포함됩니다. 사용자는 입력을 출력에 연결하고 결과에 반응하는 애플리케이션 코드를 소유합니다. 경계는 `include/`에 있는 공개 API이며, 이는 **안정적**으로 간주됩니다. 따라서 애플리케이션을 다시 작성하지 않고도 Neat을 업그레이드할 수 있습니다.

Neat의 코드를 세 줄만 기억한다면, 다음 세 줄을 기억하세요.

```cpp
simaai::neat::Model      model(mpk_path);
simaai::neat::TensorList outputs = model.run(input_tensors, /*timeout_ms=*/2000);
simaai::neat::Mapping    view = outputs[0].map_read();  // inspect the output bytes
```

이 문서에 포함된 나머지 모든 내용, 즉 그래프, 실행 핸들, 비동기 큐, 멀티스트림 앱은 핵심적인 세 줄짜리 설명의 확장된 버전입니다.
