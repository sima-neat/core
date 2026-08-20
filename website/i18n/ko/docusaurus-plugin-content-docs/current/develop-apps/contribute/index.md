---
title: "기여하다"
description: "Neat 프레임워크 아키텍처, 빌드, 테스트 및 릴리스 작업에 대한 기여자 안내서"
sidebar_position: 1
slug: /develop-apps/contribute/
---

# 기여하기

이 섹션은 Neat Library 자체를 수정하는 기여자를 위한 것입니다. 여기서는 저장소의 구조, 변경 사항을 빌드하고 테스트하는 방법, 그리고 애플리케이션 개발자를 위해 어떤 인터페이스가 안정적으로 유지되어야 하는지를 설명합니다.

<div class="overview-section-label">기고자 오리엔테이션</div>

**Neat Library**는 이 저장소에 있는 C++/Python 라이브러리 및 런타임입니다.
이 라이브러리는 모델 아카이브를 로드하고, 파이프라인을 구성하고 검증하며, Modalix 하드웨어에서 실행되고, 공개 애플리케이션 API를 제공합니다. Neat SDK 및 DevKit Sync는 관련된 개발 워크플로우입니다.

이 저장소를 변경할 때, 개발자와 에이전트 모두에게 도움이 되는 프레임워크 속성을 고려하여 최적화하세요. 여기에는 명시적인 API, 결정론적 동작, 구조화된 진단, 엄격한 검증, 안정적인 공개 계약 등이 포함됩니다.

<div class="overview-section-label">기여자 원칙</div>

- **결정성이 중요합니다.** — 요소 이름, 생성된 파이프라인, 보고서 및 테스트를 재현 가능하게 유지합니다.
- **디버깅 기능이 최우선입니다.** — 오류 발생 시 구조화된 데이터를 생성해야 하며, 단순히 문자열만 출력해서는 안 됩니다.
- **자동 대체 기능은 사용하지 않습니다.** — 모델 입력 오류 또는 하드웨어/런타임 오류를 숨기지 마십시오.
- **실행 전에 유효성을 검사합니다.** — 런타임 전에 구조, 제한, 형태 및 계약 관련 오류를 감지합니다.
- **공개 API는 안정적으로 유지됩니다.** — `include/*` 아래에 설치된 헤더는 추가적이고 호환 가능한 변경 사항을 요구합니다.
- **동시성은 제한되고 관찰 가능해야 합니다.** — 종료 작업이 중단되지 않아야 하며, 진단 기능은 스레드에 안전해야 합니다.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>여기서 시작하세요.</h2>
    <p>코드를 변경하기 전에 저장소, 명명 규칙, 그리고 기대 사항을 이해하십시오.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>아키텍처</strong><span>리포지토리 구조, 모듈 소유권, 런타임 흐름, 그리고 확장 지점을 학습합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/naming/"><strong>계약 명칭</strong><span>표준 제품, API, 패키지, 유형 이름을 일관되게 사용하십시오.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/coding_standard/"><strong>코딩 표준</strong><span>C++ 스타일, 공개 API, 호환성, 그리고 문서화에 대한 요구 사항을 따르십시오.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>구축 및 테스트</h2>
    <p>프레임워크를 구축하고, 변경 사항을 검증하고, Python 바인딩 작업을 진행합니다.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/build/"><strong>구축하다</strong><span>소스 코드를 사용하여 Neat를 빌드하고 CMake 프로필 또는 빌드 옵션을 선택하세요.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/test_requirements/"><strong>테스트 요구 사항</strong><span>각 유형의 변경 사항에 대해 어떤 테스트와 문서 업데이트가 필요한지 파악하십시오.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/python_bindings/"><strong>Python 바인딩</strong><span>기여자로서 PyNeat 바인딩을 빌드, 테스트하고 패키징합니다.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>계약 및 내부 정보</h2>
    <p>모델 아카이브, 플러그인 계약 또는 내부 패키지를 변경할 때 다음 사항을 활용하세요.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/mpk_contract/"><strong>MPK 계약</strong><span>모델 아카이브 수집, 검증 및 보안 규칙을 이해합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/sima_plugin_json_truth_map/"><strong>플러그인 JSON 진실성 맵</strong><span>고정된 SIMA 플러그인 JSON 계약 맵을 검토하십시오.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/appcomplex_workspace_packaging/"><strong>앱 복합 패키징</strong><span>게이트된 앱 복합 작업 공간 서비스 패키지를 구축하고 설치합니다.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>출시 및 유지 관리</h2>
    <p>이러한 자료를 릴리스 관문, 정리 계획, 그리고 장기간 유효한 디자인 지침으로 활용하십시오.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/release-checklist/"><strong>출시 점검 목록</strong><span>출시 차단 조건, 필수 확인 사항, 재현 가능한 단계를 따르십시오.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/error-taxonomy-rollout/"><strong>오류 분류 체계 적용</strong><span>구조화된 오류 코드 마이그레이션 및 검증 상태를 추적합니다.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/agentic-workflow/"><strong>에이전트 기반 워크플로우</strong><span>API가 인간과 AI의 협업을 통해 개발 효율성을 높일 수 있도록 설계된 이유를 알아보세요.</span></a></li>
    </ul>
  </section>
</div>
