---
title: "출시 점검 목록"
description: "릴리스 게이트 정책 및 재현 가능한 릴리스 단계"
sidebar_position: 1
slug: /develop-apps/contribute/release-checklist
---

# 릴리스 체크리스트

본 문서는 공식적인 릴리스 게이트 정책입니다.

## 릴리스 차단 조건

아래의 모든 조건이 충족되지 않으면 릴리스가 차단됩니다.

1. 추적되는 소스/문서 파일에 병합 충돌 마커가 없습니다.
2. 공개 문서에 대한 명명 규칙 검사가 통과합니다.
3. 구성/빌드 무결성 검사가 통과합니다(`cmake -S . -B ...` 및 `cmake --build ... --target sima_neat`).
4. 문서 링크 검사가 엄격한 모드에서 통과합니다(`DOCS_STRICT_LINKS=1`).
5. 생성 단계 후 작업 트리가 깨끗합니다.
6. 푸시 전과 릴리스 브랜치에서 해결되지 않은 충돌/정확성 오류가 없습니다.
7. 충돌/정확성/스트레스/정적 분석 게이트가 릴리스 브랜치에서 녹색 상태입니다.
8. 모델 아카이브 보안 게이트가 녹색 상태입니다(`model-archive-security-gate`).
9. 설치 스모크 게이트가 녹색 상태입니다(`install-smoke`).
10. 성능 회귀 게이트가 녹색 상태입니다(`perf-regression-gate`).
11. 릴리스 태그에 대한 소크 안정성 테스트가 녹색 상태입니다(`soak-weekly`).
12. 릴리스 후보에 대한 퍼즈 테스트가 녹색 상태입니다(`fuzz-nightly`).
13. 제로 스킵 게이트가 엄격한 테스트 레인에 대해 녹색 상태입니다(`zero-skip-gate`).
14. 필수 거버넌스 파일이 존재하고 유효합니다.
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-checklist.md`
15. 릴리스 메타데이터가 완료되었습니다.
   - `project(SimaNeat VERSION x.y.z)`가 `CMakeLists.txt`에 업데이트되었습니다.
   - `package-version` 및 `platform-version`가 필요한 경우 `deps/manifest.json`에 업데이트되었습니다.
   - `modelzoo-version`이 검증된 Model Zoo 릴리스를 명시적으로 선택하며, `platform-version`과 다를 경우에만 해당됩니다. 생략하면 Model Zoo 해결은 기본적으로 `platform-version`으로 설정됩니다.
   - `abi-version`이 `deps/manifest.json`에 있으며, 공개 C++ 유형 레이아웃 또는 내보낸 바이너리 계약이 호환되지 않게 변경될 때마다 증가합니다. 모든 C++ 애플리케이션 및 Python 바인딩은 해당 ABI를 사용하여 다시 빌드됩니다.
   - `CHANGELOG.md`에 `## [x.y.z]` 항목이 있습니다.
   - 릴리스/태그 본문에 릴리스 노트가 준비되었습니다.

릴리스 흐름에서 "알려진 충돌 목록"은 허용되지 않습니다. 모든 충돌 회귀는 수정될 때까지 릴리스를 차단합니다.

## 필수 상태 검사

다음 검사는 릴리스 PR 및 릴리스 태그에서 필수적으로 수행됩니다.

- `repo-hygiene`
- `configure-build-sanity`
- `docs-link-check`
- `crash-correctness-gate`
- `model-archive-security-gate`
- `install-smoke`
- `perf-regression-gate`
- `zero-skip-gate`
- `soak-weekly` (릴리스 태그에 필요)
- `fuzz-nightly` (릴리스 후보에 필요)
- `stress-gate`
- `asan-ubsan-gate`
- `release-policy-check`

이러한 검사는 다음에서 구현됩니다.

- `.github/workflows/release-gate.yml`
- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/model-archive-security.yml`
- `.github/workflows/install-smoke.yml`
- `.github/workflows/perf-regression.yml`
- `.github/workflows/zero-skip.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`

중복된 게이트 실행을 방지하기 위해 소유권 설정을 적용합니다.

- `main`에 대한 릴리스가 아닌 PR은 자체 워크플로에서 `model-archive-security`, `install-smoke`, `perf-regression` 및 `zero-skip`을 실행합니다.
- 릴리스 PR(`release/*` 헤드 참조) 및 릴리스 참조(`release/**`, `v*`)는 `.github/workflows/release-gate.yml`에서 동일한 단계를 실행합니다.

## GitHub 브랜치 및 태그 보호

GitHub 저장소 설정을 구성합니다.

1. `main` 보호:
   - 병합 전에 풀 리퀘스트를 요구합니다.
   - 최소 한 명 이상의 코드 소유자 승인을 요구합니다(가능한 경우 두 명을 권장합니다).
   - 새 커밋이 있을 때 오래된 승인을 무효화합니다.
   - 모든 필수 상태 검사를 요구합니다.
   - 강제 푸시를 금지합니다.
   - 스쿼시 전용 또는 선형 히스토리를 사용합니다.
2. 릴리스 태그를 생성할 수 있는 사용자를 제한하기 위해 `v*` 태그를 보호합니다.

## 릴리스 흐름

1. `main`의 안정적인 브랜치에서 `release/x.y.z`를 분기합니다.
2. 릴리스가 아닌 PR 병합을 중단합니다.
3. 릴리스 브랜치에서 릴리스 게이트 워크플로를 실행합니다.
4. 후보 검증을 위해 `vX.Y.Z-rcN` 태그를 생성합니다.
5. 최종 `vX.Y.Z` 태그로 승격합니다.
6. 릴리스 브랜치를 `main`으로 빠르게 병합합니다.
7. 릴리스 노트를 게시하고 릴리스 후 후속 작업을 수행합니다.

## 운영 참고 사항

- 더러운 브랜치에서 릴리스하지 않습니다.
- 검토되지 않은 코드에서 릴리스하지 않습니다.
- 필수 검사가 실패하면 릴리스하지 않습니다.
- 로컬 충돌/정확성 게이트가 실패하면 푸시를 허용하지 않습니다.
- 위생 실패에 대한 수동 우회 경로를 제공하지 않습니다.

## 성능 회귀 계약

- 성능 게이트 진입점은 `scripts/ci/run_perf_regression_gate.sh`입니다.
- 기준은 `tests/perf/baselines/v2/modalix_default/`에서 프로필별로 정의됩니다.
  - `profile.json`은 고정된 Modalix 환경 계약을 정의합니다.
  - 시나리오 ID당 하나의 시나리오 파일(`<scenario_id>.json`)이 있습니다.
- 필수 시나리오:
  - `runtime_session_sync_rgb`
  - `runtime_session_async_rgb`
  - `runtime_graph_fanout`
  - `runtime_graph_join_bundle`
  - `runtime_codec_mjpeg_decode`
  - `runtime_codec_h264_decode`
  - `runtime_codec_h265_decode`
  - `runtime_model_archive_load`
- 모든 성능 실행은 `build-perf-gate/perf_results/`에 시나리오별 결과 파일을 게시합니다.
- 각 결과에는 다음이 포함되어야 합니다.
  - `scenario_id`
  - `modalix_profile_id`
  - `status`
  - `failure_class`
  - `reason_code`
  - `metrics`
  - `run_meta`
  - `timestamp`
- `REGRESSION`, `HARNESS_ERROR` 또는 `ENV_BROKEN` 분류는 단계를 차단합니다.
