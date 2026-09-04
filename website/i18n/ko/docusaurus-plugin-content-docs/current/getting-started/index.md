---
title: "Palette Neat"
sidebar_position: 1
---

# Palette Neat

Palette Neat는 Modalix에서 AI 애플리케이션을 구축하기 위한 SiMa.ai 소프트웨어 개발
도구 모음입니다. 개발 환경, 런타임 라이브러리, 모델 도구 및 DevKit 검증 워크플로를
포함합니다. 이 구성 요소들은 모델 준비부터 애플리케이션 구축, Modalix 하드웨어에서의
결과 검증까지 전체 개발 과정을 지원합니다.

이 개요에서 Palette Neat의 주요 구성 요소를 살펴보고 설정, 모델 준비 또는
애플리케이션 개발에 적합한 경로를 선택할 수 있습니다.

![Neat SDK, Neat Library, Model Compiler, LLiMa 및 Modalix DevKit으로 구성된 Palette Neat 소프트웨어 스택](@site/../docs/images/neat-software-stack-animated.svg)

<div class="overview-section-label">개발자 여정</div>

:::tip Palette Neat를 처음 사용하는 경우
Palette Neat로 애플리케이션을 개발하는 방법은 두 가지입니다.

- 모델을 컴파일하거나 대규모 C++ 코드를 크로스 컴파일할 계획처럼 더 높은 성능의
  개발 환경이 필요하면 **[Neat SDK 사용](/getting-started/dev-environment/)**을 선택하세요.
- 특히 모델 컴파일 작업을 하지 않으며 개발 환경의 구성 요소를 줄이고 싶다면
  **[DevKit에서 직접 개발](/getting-started/neat-library/)**을 선택하세요.

SDK 경로를 선택한 경우 호스트가 [호스트 요구 사항](/getting-started/dev-environment/#host-requirements)을
충족하는지 확인한 후 SDK를 설치하세요. SDK 설치 과정에서 호환되는 기본값이 적용되므로,
정확한 버전을 고정하거나 구성 요소를 개별적으로 업그레이드하거나 버전 불일치를 해결할
때만 호환성 참조 문서가 필요합니다.

**SDK 전용 빠른 경로(DevKit 없음):** SDK를 설치한 후 바로
[모델 컴파일](/compile-a-model/)로 이동하세요. SDK 구성, DevKit Sync, Neat Library 및
PyNeat 설정은 DevKit을 연결할 때까지 건너뛸 수 있는 선택 단계입니다.
:::

<div class="overview-section-label">명령어 읽는 방법</div>

이 문서의 명령어 블록에는 실행 환경이 레이블과 색상으로 표시되어 있으므로, 어디에 입력해야 할지 추측할 필요가 없습니다.

| 프롬프트 | 실행 위치 |
| --- | --- |
| `host$` | SDK 외부의 사용자 머신에서. |
| `sdk$` | Neat SDK 컨테이너 셸 안에서. |
| `devkit$` | Modalix DevKit에서. |
| `pcie-host$` | Modalix PCIe 카드가 장착된 호스트 머신에서. |

`sdk or devkit$`처럼 두 개 이상의 환경이 표시된 블록은 어느 쪽에서든 동일하게 실행됩니다. 명령어의 경로가 상대 경로인 경우, 블록에 실행할 디렉터리도 함께 표시됩니다.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>시작하기</h2>
    <p>로컬 개발과 하드웨어 검증을 위해 호스트 머신, Neat SDK 및 DevKit을 준비합니다.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/"><strong>Neat SDK</strong><span>고성능 호스트 기반 워크플로, 모델 컴파일, C++ 빌드 및 DevKit 검증에 SDK를 사용합니다.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/"><strong>Neat Library</strong><span>DevKit에서 더 간단하게 애플리케이션을 프로토타이핑하려면 런타임과 PyNeat를 직접 설치합니다.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/compatibility/"><strong>호환성 가이드</strong><span>지원되는 버전 조합을 확인하는 참조 가이드입니다.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>모델 준비</h2>
    <p>학습된 모델을 Modalix 하드웨어에서 실행되는 배포 가능한 아티팩트로 변환합니다.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/compile-a-model/"><strong>모델 컴파일</strong><span>사전 학습된 ONNX 비전 모델 또는 GenAI 모델을 Modalix용으로 컴파일합니다.</span></a></li>
      <li><a class="overview-link-card" href="/tools/model-zoo/"><strong>사전 컴파일된 모델 사용</strong><span>즉시 실행 가능한 모델 아티팩트로 빠르게 시작합니다.</span></a></li>
      <li><a class="overview-link-card" href="/genai-llima/"><strong>LLiMa 기반 GenAI</strong><span>Modalix에서 GenAI 모델을 컴파일하고 테스트하며 벤치마크합니다.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>애플리케이션 구축</h2>
    <p>Neat Library를 사용해 모델을 실행하고 프로덕션 애플리케이션 파이프라인을 구성합니다.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Hello Neat!</strong><span>첫 Neat 애플리케이션을 실행하고 개발 워크플로를 확인합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/"><strong>애플리케이션 개발</strong><span>C++ 또는 PyNeat로 Neat Library 기반 AI 애플리케이션을 구축합니다.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>튜토리얼</strong><span>실제 Neat 애플리케이션 패턴을 안내 예제로 학습합니다.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>도구 및 참조</h2>
    <p>자세한 정보가 필요할 때 지원 도구와 참조 자료를 활용합니다.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/tools/"><strong>도구</strong><span>sima-cli, Model Zoo 및 Insight.</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>참조</strong><span>API, 문제 해결, 환경 변수, 데이터 형식 및 릴리스 노트를 살펴봅니다.</span></a></li>
      <li><a class="overview-link-card" href="/reference/troubleshooting/"><strong>문제 해결</strong><span>설정, 런타임 및 파이프라인 문제의 해결 방법을 찾습니다.</span></a></li>
    </ul>
  </section>
</div>
