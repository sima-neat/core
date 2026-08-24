---
title: "앱 개발"
description: "Modalix에서 SiMa.ai Neat로 AI 애플리케이션 빌드 및 실행"
sidebar_position: 1
---

# SiMa.ai와 Neat를 사용하여 앱을 개발하세요.

<LanguageContent lang="cpp">

<div className="overview-workflow-image overview-workflow-image-light">

![SiMa.ai Neat로 구성한 C++ 애플리케이션 워크플로](@site/../docs/develop-apps/images/neat-overview-workflow-cpp.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![SiMa.ai Neat로 구성한 C++ 애플리케이션 워크플로](@site/../docs/develop-apps/images/neat-overview-workflow-cpp-dark.svg)

</div>

</LanguageContent>

<LanguageContent lang="py">

<div className="overview-workflow-image overview-workflow-image-light">

![SiMa.ai Neat로 구성한 Python 애플리케이션 워크플로](@site/../docs/develop-apps/images/neat-overview-workflow-python.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![SiMa.ai Neat로 구성한 Python 애플리케이션 워크플로](@site/../docs/develop-apps/images/neat-overview-workflow-python-dark.svg)

</div>

</LanguageContent>

## SiMa.ai Neat이란 무엇인가

SiMa.ai Neat은 SiMa.ai 플랫폼에서 AI 애플리케이션을 구축하고 실행하기 위한 애플리케이션 개발 프레임워크입니다. 컴파일된 모델 아카이브(`.tar.gz`)를 로드하고 실행하고, Modalix 처리 리소스를 사용하는 엔드-투-엔드 애플리케이션을 구성하고, 런타임 실행을 관리하기 위한 Python 및 C++ API를 제공합니다.

더 넓은 SiMa.ai 소프트웨어 스택 내에서 SiMa.ai Neat은 애플리케이션 계층에 위치합니다. SiMa.ai 런타임 스택을 기반으로 하며, 그 아래에 GStreamer를 사용하므로 개발자는 하위 수준의 런타임 구성 요소를 수동으로 연결하는 대신 애플리케이션 로직에 집중할 수 있습니다.

가장 빠른 추론 경로를 위해 컴파일된 모델 아카이브를 `Model`로 로드하고 직접 실행합니다. 애플리케이션에 여러 입력, 처리 단계, 모델 또는 출력이 필요한 경우 해당 구성 요소를 `Graph`로 구성하고 `Run`으로 빌드합니다. 동일한 공개 API는 기존 및 에이전트 기반 개발을 모두 지원하므로 팀은 두 가지 워크플로 중 하나를 사용하여 애플리케이션을 검토, 확장 및 유지 관리할 수 있습니다.

### 배포 모델 선택

- **Modalix DevKit에서 실행** — 애플리케이션과 Neat 그래프가 장치에서 직접 실행됩니다. `simaai::neat` 또는 `pyneat`와 함께 `Model`, `Graph`, `Node` 및 `Run`을 사용합니다. [실행 / 추론](/develop-apps/development-workflow/overview/)로 시작합니다.
- **코프로세싱을 위해 Modalix PCIe 카드를 사용** — 애플리케이션은 호스트 시스템에서 실행되고 텐서 또는 이미지를 카드로 전송하여 모델을 실행합니다. `simaai::neat::pcie` 또는 `pyneatpcie`를 사용합니다. [PCIe 공동 처리](/develop-apps/development-workflow/pcie-model/)으로 시작합니다.

### C++ 또는 PyNeat

Modalix DevKit에서 직접 실행되는 애플리케이션의 경우, SiMa.ai Neat은 동일한 핵심 워크플로를 두 가지 언어 인터페이스를 통해 제공하므로 애플리케이션에 적합한 것을 선택할 수 있습니다.

- PyNeat — Python 바인딩(`pyneat`). 빠른 반복, 노트북, 데이터 과학 워크플로 및 DevKit에서 Python 애플리케이션을 직접 실행하는 데 가장 적합합니다.
- C++ — 기본 `simaai::neat` API. 더 큰 애플리케이션, 기존 C++ 코드베이스와의 긴밀한 통합 및 교차 컴파일된 호스트-to-DevKit 워크플로에 가장 적합합니다.

두 가지 모두 동일한 컴파일된 모델 아티팩트 및 Modalix 런타임을 사용합니다. 아래의 개념 및 페이지는 둘 다에 적용됩니다. PCIe 코프로세싱은 `simaai::neat::pcie` 및 `pyneatpcie`를 통해 별도의 C++ 및 Python 인터페이스를 제공합니다.

## 애플리케이션을 개발합니다. <span className="neat-heading-highlight">SiMa.ai Neat 사용자에게 지도를 제공합니다.</span>

Modalix는 애플리케이션 코어, 비전 처리, 머신 러닝 가속, 비디오 엔진, 공유 메모리, 고속 I/O를 하나의 SoC에 통합합니다. Python 및 C++ API를 통해 SiMa.ai Neat는 시스템 내의 애플리케이션 관련 처리 리소스 전반에 걸쳐 애플리케이션을 구축하기 위한 단일 프로그래밍 모델을 제공합니다.

카메라 또는 네트워크 스트림에서 시작하여 처리 및 추론을 거쳐 최종 결과에 이르기까지 전체 파이프라인을 구축합니다. SiMa.ai Neat는 런타임 파이프라인을 구성하고, 필요한 경우 가속화된 구현을 선택하며, Modalix 전반에 걸쳐 실행 및 데이터 이동을 조정합니다. 사용자는 애플리케이션에 집중하고, SiMa.ai Neat는 기본 하드웨어 및 런타임 복잡성을 처리합니다.

<div className="overview-workflow-image modalix-application-map-desktop">

![MLSoC Modalix 평면도에 매핑된 SiMa.ai Neat 애플리케이션](@site/../docs/images/neat-modalix-floorplan-animated.svg)

</div>

<div className="overview-workflow-image modalix-application-map-mobile">

![MLSoC Modalix 평면도에 매핑된 SiMa.ai Neat 애플리케이션의 모바일 화면](@site/../docs/images/neat-modalix-floorplan-mobile-animated.svg)

</div>

<p className="overview-figure-caption"><strong>예시 매핑:</strong> 선택하는 경로는 애플리케이션, 모델, 사용 가능한 하드웨어 가속 기능에 따라 달라집니다. 자세한 내용은 다음을 참조하십시오. <a href="/develop-apps/advanced-concepts/processor_backends/">프로세서 백엔드</a> 기술적 매핑을 위해.</p>

## 애플리케이션을 설명해 주세요. <span className="agentic-heading-highlight">Neat 기술을 갖춘 에이전트가 개발합니다.</span>

SiMa.ai Neat 제공되는 기술을 통해 에이전트 기반 애플리케이션 개발을 즉시 지원합니다. Neat 개발 환경(이하 ‘개발 환경’이라고 함) Neat SDK). 이러한 기술은 코딩 에이전트가 공개된 Python 및 C++ API를 사용하고, 확립된 애플리케이션 패턴을 따르며, 관련 작업과 함께 작동할 수 있는 맥락을 제공합니다. Modalix 개발 및 검증
워크플로

권장되는 에이전트 기반 방식은 애플리케이션을 생성하고, 페어링된 환경에서 실행할 수 있습니다.
Modalix DevKit결과 및 진단 정보를 검토하고 구현을 개선합니다. 기존 개발 방식은 동일한 API를 통해 직접 제어할 수 있는 별도의 경로로 유지됩니다. 두 방식 모두 표준화되고 검사 가능한 결과를 생성합니다. SiMa.ai Neat 애플리케이션을 통해 에이전트가 개발한 코드를 검토하거나 수정하고, 애플리케이션이 발전함에 따라 두 워크플로 간에 자유롭게 이동할 수 있습니다. 자세한 내용은 다음을 참조하십시오. [Neat SDK를 설정합니다.](/getting-started/dev-environment/) 에이전트의 능동적인
발전을 가능하게 하기 위해.

<div className="overview-workflow-image agentic-visual-desktop">

![코딩 에이전트가 SiMa.ai Neat 애플리케이션을 생성, 실행, 진단 및 개선하는 과정](@site/../docs/images/agentic-development-loop-animated.svg)

</div>

<div className="overview-workflow-image agentic-visual-mobile">

![SiMa.ai Neat 에이전트 개발 루프의 모바일 화면](@site/../docs/images/agentic-development-loop-mobile-animated.svg)

</div>

## 요구 사항

애플리케이션을 빌드하기 전에 먼저 시작하기 설정을 완료하세요.

- **배포 모델 설치** — Modalix DevKit의 경우, Neat 라이브러리를 Neat SDK에 설치하거나 장치에 직접 설치합니다. PCIe 공동 처리의 경우, 호스트 장치에 `core/pciehost`를 설치하고 Modalix PCIe 카드에 호환되는 Neat Library를 설치합니다.
- **모델 아티팩트** — Model Zoo에서 미리 컴파일된 모델을 사용하거나, 직접 모델을 컴파일하여 Modalix에서 사용할 수 있는 아카이브를 만듭니다.
- **런타임 대상** — DevKit에서 네이티브 애플리케이션을 실행하거나, 호스트 장치에서 직접 공동 처리 애플리케이션을 빌드하고 실행합니다. Neat SDK에서 네이티브 C++ 애플리케이션을 교차 컴파일할 때 DevKit를 페어링하고 동기화합니다.

Hello Neat! 페이지는 첫 번째 추론을 실행하는 데 도움을 주며, 개발 워크플로 페이지는 주요 개념을 더 자세히 설명하고, 튜토리얼은 실제 애플리케이션 패턴에 적용하는 방법을 보여줍니다.

완전한 애플리케이션을 살펴보고, 수정하고, 실행할 수 있도록 [응용 예시](https://developer.sima.ai/examples)를 참조하세요.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>여기서 시작하세요.</h2>
    <p>실제 작업 환경에서 시작하여 핵심 기능을 구축합니다. SiMa.ai Neat 애플리케이션 워크플로.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>안녕하세요 Neat!</strong><span>최소한의 Neat 애플리케이션을 실행하고 개발 주기가 제대로 작동하는지 확인합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/development-workflow/overview/"><strong>개발 프로세스</strong><span>`Model`, `Graph`, 그리고 `Run` 워크플로우에 대해 더 자세히 알아보세요.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>튜토리얼</strong><span>실제 사례를 통해 설명하는 안내 예제를 따르세요. SiMa.ai Neat 애플리케이션 패턴.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-explore">
    <h2>더 많이 구축하세요.</h2>
    <p>더욱 풍부한 기능을 갖춘 애플리케이션을 구축하거나 API를 자세히 검토할 준비가 되었을 때 이 섹션을 활용하세요.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/"><strong>고급 개념</strong><span>그래프, 형식, 메모리, 스레드, 그리고 런타임 동작을 이해합니다.</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>참고</strong><span>C++, Python, Model Compiler, 문제 해결 방법, 관련 자료 등을 살펴보세요.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>기여하다</strong><span>아키텍처, 소스 빌드, 테스트 요구 사항, 그리고 저장소 규칙을 이해합니다.</span></a></li>
    </ul>
  </section>
</div>
