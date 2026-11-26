---
title: "출시 점검 목록"
description: "릴리스 게이트 정책 및 재현 가능한 릴리스 단계"
sidebar_position: 1
slug: /develop-apps/contribute/release-checklist
---

# 릴리스 체크리스트

본 문서는 공식적인 릴리스 게이트 정책입니다.

## 최신 문서 버전

`deps/manifest.json`을 업데이트한 후 릴리스에 맞게 `website/release-data/current.json`을 한 번 업데이트합니다. `npm --prefix website run sync:release-readmes`를 실행하고 구체적인 버전이 포함된 생성된 README 명령을 커밋합니다. `npm --prefix website run check:releases`와 `./build.sh --doc`를 실행하여 토큰, 과거 버전 예외, 현지화 및 렌더링된 명령을 검증합니다. 호환성 페이지는 구체적인 버전을 사용하여 수동으로 관리합니다. 과거 릴리스 노트의 버전은 고정합니다. 릴리스 데이터만 변경할 때는 번역이나 소스 해시를 업데이트할 필요가 없습니다. 필드별 관리 책임과 토큰 규칙은 `website/release-data/README.md`에 설명되어 있습니다.

## 릴리스 차단 조건

아래의 모든 조건이 충족되지 않으면 릴리스가 차단됩니다.

1. 추적되는 소스/문서 파일에 병합 충돌 마커가 없습니다.
2. 공개 문서에 대한 명명 규칙 검사가 통과합니다.
3. 구성/빌드 무결성 검사가 통과합니다(`cmake -S . -B ...` 및 `cmake --build ... --target sima_neat`).
4. 문서 링크 검사가 엄격한 모드에서 통과합니다(`DOCS_STRICT_LINKS=1`).
5. 생성 단계 후 작업 트리가 깨끗합니다.
6. 푸시 전과 릴리스 브랜치에서 해결되지 않은 충돌/정확성 오류가 없습니다.
7. Vulcan CI(`.github/workflows/vulcan-ci.yml`)가 태그를 지정할 바로 그 커밋에서 통과했습니다.
8. 제로 스킵 게이트(`scripts/ci/run_zero_skip_gate.sh`)가 Vulcan CI 테스트 레인에서 통과합니다.
9. 필수 거버넌스 파일이 존재하고 유효합니다.
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-checklist.md`
10. 릴리스 메타데이터가 완료되었습니다.
   - 태그를 지정할 커밋의 `deps/manifest.json`에 있는 `package-version`이 릴리스 버전과 같습니다. 예를 들어 `v0.7.0`이면 `0.7.0`입니다.
   - `platform-version`가 필요한 경우 `deps/manifest.json`에 업데이트되었습니다.
   - `modelzoo-version`이 검증된 Model Zoo 릴리스를 명시적으로 선택하며, `platform-version`과 다를 경우에만 해당됩니다. 생략하면 Model Zoo 해결은 기본적으로 `platform-version`으로 설정됩니다.
   - `deps/manifest.json`의 `abi-version`은 공개 C++ 유형 레이아웃 또는 내보낸 바이너리 계약에 호환되지 않는 변경이 있고,
     `develop`이 여전히 `main`과 같은 ABI 버전을 사용하는 경우에만 증가시킵니다.
     `develop`에 다음 릴리스를 위한 더 높은 ABI 버전이 이미 있으면, 릴리스 전의 추가적인 ABI 비호환 변경에도 그 버전을 유지합니다.
     호환되지 않는 변경 후에는 일치하는 헤더와 라이브러리를 사용하여 모든 C++ 애플리케이션과 Python 바인딩을 다시 빌드합니다.
   - 릴리스/태그 본문에 릴리스 노트가 준비되었습니다.

릴리스 흐름에서 "알려진 충돌 목록"은 허용되지 않습니다. 모든 충돌 회귀는 수정될 때까지 릴리스를 차단합니다.

## 필수 상태 검사

릴리스 태그를 만들기 전에 태그를 지정할 바로 그 커밋에서 Vulcan CI(`.github/workflows/vulcan-ci.yml`)가
통과해야 합니다. 확인 방법은 릴리스 흐름의 5단계에 있습니다.

## 예약 실행 테스트 레인

이 레인은 릴리스를 차단하지 않습니다. 충돌/정확성, 스트레스, 정적 분석, 퍼즈 테스트는 매일 밤 실행됩니다.
소크 안정성 테스트와 장시간 테스트는 매주 실행됩니다.

- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`

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

`v0.7.0`과 같은 최종 태그로 릴리스합니다. 릴리스 후보 태그는 만들지 않습니다. Release 워크플로
(**Actions → Release**)는 `release-X.Y`를 만들거나 재사용하고, 그 최신 커밋에 태그를 지정한 뒤 GitHub 릴리스 초안을 만듭니다.
아래 단계는 새 라인에서 0.7.0을 릴리스하는 경우입니다. 기존 라인의 패치 릴리스(예: 0.6.3)는 3단계와 6단계가 다릅니다.

1. `develop`에서 `deps/manifest.json`의 `package-version`을 `0.7.0`으로 설정합니다. 릴리스 차단 조건에서 요구하면 `platform-version`, `modelzoo-version`, `abi-version`을 업데이트하고, 최신 문서 버전 절의 설명에 따라 릴리스 데이터를 업데이트합니다.
2. 풀 리퀘스트로 `develop`을 `main`에 병합합니다.
3. 태그를 지정할 커밋을 확인합니다. 먼저 `git fetch origin`을 실행합니다.
   - 새 라인: 워크플로가 `main`의 최신 커밋에서 `release-0.7`을 만듭니다. 커밋은 `git rev-parse origin/main`입니다.
   - 기존 라인: 풀 리퀘스트로 `main`을 `release-0.6`에 병합합니다. `branch_exists_behavior`의 기본값이 `reuse`이므로 워크플로는 `release-0.6`을 재사용하고 그 최신 커밋에 태그를 지정합니다. 커밋은 `git rev-parse origin/release-0.6`입니다.

   태그가 만들어질 때까지 그 브랜치에는 아무것도 병합하지 않습니다.
4. 그 커밋의 버전을 확인합니다. `git show <commit>:deps/manifest.json`에 `"package-version": "0.7.0"`이 표시되어야 합니다. Release 워크플로도 첫 번째 작업에서 같은 검사를 수행합니다. 버전이 다르면 브랜치나 태그를 만들기 전, 그리고 이슈를 릴리스됨으로 표시하기 전에 중단합니다.
5. 바로 그 커밋에서 Vulcan CI가 통과했는지 확인합니다. `gh run list --repo sima-neat/core --workflow vulcan-ci.yml --commit <commit>`으로 실행 목록을 볼 수 있습니다. `main` 또는 릴리스 브랜치에서 성공한 실행도 인정됩니다. 없으면 3단계의 브랜치에서 `gh workflow run vulcan-ci.yml --repo sima-neat/core --ref main`으로 Vulcan CI를 시작합니다. 패치 릴리스에서는 `--ref release-0.6`을 사용합니다. 새 실행의 `headSha`가 그 커밋인지 확인한 뒤 `gh run watch <run-id> --repo sima-neat/core --exit-status`로 완료를 기다립니다. 그 커밋의 실행이 통과할 때까지 태그를 지정하지 않습니다.
6. **Actions → Release → Run workflow**를 엽니다. `release_line`을 `0.7`로, `tag`를 `0.7.0`으로 설정하고, `main`을 `source_branch`로 그대로 두며, `prerelease`는 끄고 `latest`는 켠 상태로 둡니다. 0.6.3의 경우 `release_line`을 `0.6`으로, `tag`를 `0.6.3`으로 설정하고, 더 새로운 라인이 이미 릴리스되었다면 `latest`를 끕니다. 워크플로는 `v0.7.0` 태그를 지정하고 릴리스 초안을 만듭니다. Target Release가 `core-0.7.0`인 모든 이슈가 Done이어야 하며, 그 후 해당 이슈의 Release Status를 `released`로 설정합니다. 해당 Target Release를 가진 이슈가 하나도 없어도 이 검사는 실패합니다. 실행 요약의 **Target SHA**가 3단계의 커밋인지 확인합니다. 다르면 릴리스 초안을 게시하지 않습니다.
7. `gh workflow run vulcan-ci.yml --repo sima-neat/core --ref v0.7.0`으로 태그에 대해 Vulcan CI를 시작합니다. 워크플로가 지정한 태그만으로는 워크플로 실행이 시작되지 않습니다. `https://artifacts.neat.sima.ai/core/v0.7.0/latest.tag`에 태그를 지정한 커밋이 표시될 때까지 기다립니다. 이 실행은 `sima-cli neat install core@v0.7.0`이 설치하는 패키지와 오프라인 번들을 게시합니다.
8. 릴리스 초안에 릴리스 노트를 작성해 게시하고 릴리스 후 후속 이슈를 엽니다.

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
