---
title: "오류 분류 체계 적용"
description: "체계적인 오류 코드 마이그레이션 및 검증 점검 목록"
sidebar_position: 2
slug: /develop-apps/contribute/error-taxonomy-rollout
---

# 오류 분류 체계 배포

이 체크리스트는 핵심 및 런타임 플러그인 전반에 걸쳐 표준 오류 의미 체계 배포를 추적합니다.

## 표준 코드

[`include/pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h)는
참조 소스입니다. [오류 코드 목록](/reference/error-codes)는 모든 C++ 상수, 모든 Python `ERROR_*` 이름, 그리고 일반적인 코드에서 구체적인 코드로의 마이그레이션을 문서화해야 합니다.

## 실행 슬라이스

1. 분류 체계 구축
2. 코드 작성/유효성 검사
3. 런타임 풀 코드 작성
4. 그래프 IO 파서/열기 코드 작성
5. 테스트 + 문서

## 호환성 검토

- 기존 오류에서 반환되는 정확한 코드가 변경되는 경우, C++ 또는 Python 시그니처 변경이 없더라도 동작 변경으로 간주합니다.
- 공개 마이그레이션 테이블에 이전 코드와 새 코드 간의 매핑을 문서화합니다.
- 구체적인 분류가 없는 오류에 대해서만 폴백 코드(`build.parse_launch`, `runtime.pull` 및 `runtime.element_failed`)를 유지합니다.
- 프로덕션 빌더에서 버전이 지정된 `simaai-neat-error` 와이어 키를 테스트하고 실제 `GstMessage`를 Core를 통해 파싱합니다.

## 검증 체크리스트

- 터미널 프레임워크 오류 발생 시 `NeatError.report().error_code`가 비어 있지 않아야 합니다.
- 런타임 풀 오류 발생 시 `PullError.code`에 값이 채워져야 합니다.
- 그래프 래퍼 오류에는 코드 + 컨텍스트 + 힌트가 포함되어야 하며, 일반적인 폴백 텍스트는 포함되지 않습니다.
- JSON 파싱 오류에는 `offset=` 및 `near='...'`가 포함되어야 합니다.
- 네거티브 테스트는 분류 체계 클래스별로 코드 + 안정적인 메시지 조각을 확인해야 합니다.
- 진단 문서 및 아키텍처 문서에는 문제 해결 흐름이 포함되어야 합니다. 즉, `error_code`를 읽고, `repro_note`를 검사하고, 버스 진단을 검사한 다음 `repro_gst_launch`를 사용하여 다시 실행합니다.
