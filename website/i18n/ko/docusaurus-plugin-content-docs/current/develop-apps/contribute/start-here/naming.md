---
title: "계약 명칭"
description: "SiMa.ai 및 Neat에 대한 표준 제품, API 및 유형 명명 규칙"
sidebar_position: 2
slug: /develop-apps/contribute/naming
---

# 명명 규칙

본 문서는 SiMa.ai Neat 코드베이스에 대한 표준 명명 규칙을 정의합니다.

## 표준 이름

- 제품 이름: `SiMa.ai Neat`
- CMake 프로젝트: `SimaNeat`
- C++ 네임스페이스: `simaai::neat`
- 핵심 런타임 유형: `Model`, `Graph`, `Run`

## 공개 헤더

`include/` 아래의 헤더 파일을 공개 소스 코드로 사용합니다.

예시:

```cpp
#include "model/Model.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
```

## 기존 별칭

기존 이름은 호환성 참조 및 마이그레이션 가이드로만 지원됩니다.

- `PipelineSession` -> `Graph`
- `PipelineRun` -> `Run`
- `NeatModel` -> `Model`
- `InputAppSrc` -> `Input`
- `OutputAppSink` -> `Output`

기존 기호를 사용하여 새로운 공개 문서/예제를 만들지 마십시오.

## 정책

- 새로운 사용자 대상 문서에는 표준 이름을 사용해야 합니다.
- 기존 용어는 마이그레이션 참고 자료, 호환성 관련 설명 또는 명확하게 표시된 더 이상 사용되지 않는 섹션에서만 허용됩니다.
- CI는 `scripts/ci/check_naming_and_conflicts.sh`를 통해 이를 적용합니다.
