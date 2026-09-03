---
title: "sccache 사용법 요약"
description: "Neat 컴파일러 캐시를 로컬 환경과 Vulcan 환경에서 사용하고 문제를 해결합니다."
sidebar_position: 2
slug: /develop-apps/contribute/sccache
---

# Neat `sccache` 치트 시트

Neat은 C 및 C++ 컴파일러 런처로 [`sccache`](https://github.com/mozilla/sccache)를 사용합니다. 최종 Neat 패키지, 테스트 결과, 종속성 다운로드 또는 Docker 이미지 대신 컴파일러 출력을 캐시합니다.

대부분의 개발자는 설치하거나 구성할 필요가 없습니다. `build.sh`를 평소대로 사용하고 마지막에 출력되는 캐시 통계를 확인하십시오.

## 간략한 개요

| 빌드 | 캐시 수준 | 쓰기 | `--clean`에서 보존 |
|---|---|---|---|
| 로컬 | 사용자 로컬 디스크 | 로컬 디스크 | 예 |
| Vulcan `develop` 또는 `main` | 런너 로컬 디스크, 그런 다음 보호 브랜치 S3 | 로컬 디스크 및 해당 보호 S3 네임스페이스 | S3에서 보존 |
| Vulcan 기능 브랜치 푸시 | 런너 로컬 디스크, 그런 다음 브랜치 S3 | 로컬 디스크 및 해당 격리된 브랜치 네임스페이스 | 브랜치 삭제 시까지 |
| Vulcan 태그 또는 직접 참조가 아닌 참조 | 런너 로컬 디스크, 그런 다음 가장 가까운 보호 S3 | 로컬 디스크만 | 영구적인 런너 상태 없음 |

지원되는 진입점은 항상 다음과 같습니다.

```bash
./build.sh <options>
```

컴파일러를 대체하거나 런처 옵션을 수동으로 추가하기 위해 `sccache`를 사용하지 마십시오. `build.sh`는 CMake 런처를 모두 제공합니다.

```text
CMAKE_C_COMPILER_LAUNCHER
CMAKE_CXX_COMPILER_LAUNCHER
```

## 로컬 빌드

### 일반적인 사용

`auto` 모드에서 캐싱이 활성화되었습니다.

```bash
./build.sh --dev-only
./build.sh --all --clean
```

기본 캐시 위치와 제한은 다음과 같습니다.

```text
~/.cache/sima-neat/sccache
10 GiB
```

캐시는 `build/` 외부 위치에 있습니다. `build/`를 삭제하거나 `--clean`을 실행해도 캐시된 컴파일러 결과가 삭제되지 않습니다.

`sccache`가 `PATH`에 포함되어 있지 않으면 `build.sh`가 고정된 릴리스를 다음 위치에 다운로드합니다.

```text
${XDG_CACHE_HOME:-$HOME/.cache}/sima-neat/tools/sccache/<version>/
```

아카이브는 `scripts/configure_sccache.sh`에 기록된 SHA-256을 기준으로 검증됩니다. arm64 및 x86-64 아키텍처를 사용하는 Linux 및 macOS가 지원됩니다.

### 일반적인 제어 항목

```bash
# Explicitly require sccache. Fail the build if it cannot be configured.
SIMANEAT_SCCACHE=on ./build.sh --all

# Disable caching for a reproducibility comparison.
SIMANEAT_SCCACHE=off ./build.sh --all --clean

# Put the local cache on a larger or faster volume.
SCCACHE_DIR=/mnt/nvme/sccache ./build.sh --all

# Change the local cache limit.
SCCACHE_CACHE_SIZE=20G ./build.sh --all
```

`SIMANEAT_SCCACHE=auto`가 기본 설정입니다. 이 모드에서는 부트스트랩 실패 시 경고가 표시되고 캐싱 없이 빌드가 계속됩니다. `on`을 사용하면 해당 실패가 치명적인 오류가 됩니다.

### 로컬 캐시 검사 또는 삭제

`build.sh`에서 선택한 것과 동일한 바이너리를 사용하거나, `sccache`가 `PATH`에 설정되어 있을 때 사용합니다.

```bash
sccache --show-stats
sccache --zero-stats
sccache --show-adv-stats
```

공간을 확보하려면 서버를 중지하고 구성된 캐시 디렉터리만 삭제하세요.

```bash
sccache --stop-server
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/sima-neat/sccache"
```

삭제하기 전에 해결된 경로를 확인합니다. 사용자 캐시 디렉터리 전체를 삭제하지 마십시오.

## 벌칸 클라우드 빌드

벌칸은 동일한 로컬 디스크 캐시와 암호화된 S3 계층을 제공합니다.

```text
s3://sima-neat-compiler-cache-production/
  core/
    sccache-v1/
      <architecture>/
        <sdk-cache>/
          <build-mode>/
            develop/
              branches/<encoded-feature-branch>/
            main/
              branches/<encoded-feature-branch>/
```

예를 들어:

```text
core/sccache-v1/arm64/sdk-develop/standard/develop/
```

네임스페이스는 의도적으로 다음을 포함합니다.

- `sccache-v1`: 캐시 스키마로, 의도적인 전역 재설정을 허용합니다.
- 아키텍처: arm64 및 x86-64 컴파일러 출력이 혼합되지 않도록 방지합니다.
- SDK 캐시 식별자: 호환되지 않는 SDK/툴체인 출력이 혼합되지 않도록 방지합니다.
- 빌드 모드: 표준 및 퍼징 계측을 분리합니다.
- 보호된 기본 브랜치: `develop` 및 `main`이 동일한 네임스페이스에 쓰는 것을 방지합니다.

S3 버킷은 비공개이며 자체 KMS 키로 암호화되어 있으며 아티팩트 버킷과 분리되어 있습니다. 컴파일러 캐시 객체가 비공개이고 일시적이므로 CloudFront 배포가 없습니다. 객체는 45일 후에 자동으로 만료됩니다.

### 브랜치 접근

| Git 참조 | OIDC 역할 | S3 모드 |
|---|---|---|
| 정확히 `refs/heads/develop` | 보호된 작성자 | `develop/`의 `READ_WRITE` |
| 정확히 `refs/heads/main` | 보호된 작성자 | `main/`의 `READ_WRITE` |
| 직접 기능 브랜치 푸시 | 브랜치 작성자 | `<base>/branches/<branch>/` 아래의 `READ_WRITE` |
| 태그 또는 비직접 참조 | 읽기 전용 | 선택한 보호된 기준에서 `READ_ONLY` |

기능 브랜치의 첫 번째 빌드에서 가장 가까운 보호된 Git 조상(`develop` 또는 `main`)에서 캐시를 복사하여 자체 네임스페이스에 저장합니다. 그런 다음 빌드는 해당 브랜치 네임스페이스만 읽고 씁니다. 후속 빌드는 GitHub의 브랜치 삭제 이벤트가 해당 브랜치의 모든 아키텍처, SDK 및 빌드 모드 네임스페이스를 제거할 때까지 이를 재사용합니다. 기능 브랜치는 보호된 캐시에 쓸 수 없습니다.

자동 조상 감지는 `develop` 및 `main`에 대한 병합 기반 거리를 비교합니다. 재사용 가능하거나 수동으로 실행되는 워크플로는 예외적인 브랜치에 명시적인 기준이 필요한 경우 `cache_base_branch=develop|main`을 설정할 수 있습니다. AWS 자격 증명은 단기 GitHub OIDC 자격 증명이며, 장기 AWS 키는 GitHub 또는 SDK 컨테이너에 저장되지 않습니다.

첫 번째 쓰기 가능한 보호된 브랜치 빌드 전에 빈 보호된 네임스페이스가 예상됩니다. 기능 브랜치는 여전히 자체 네임스페이스를 채울 수 있지만, 선택한 보호된 기준이 비어 있으면 초기 캐시 적중이 없습니다.

Vulcan은 CMake을 구성하기 전에 명시적으로 `sccache` 시작을 확인합니다. S3, KMS, 네트워크 또는 임시 자격 증명이 캐시 서버가 시작되는 것을 방지하는 경우 워크플로는 경고를 출력하고 `sccache` 없이 컴파일합니다. 따라서 원격 캐시 가용성은 최적화이며 컴파일을 차단할 수 없습니다. Vulcan 러너는 일시적이므로 러너 로컬 캐시는 대체 수단으로 사용되지 않습니다. 로컬 개발 빌드는 일반적인 영구 디스크 캐시를 유지합니다.

## 빌드 통계 읽기

캐시된 모든 빌드는 다음과 유사한 출력으로 끝납니다.

```text
Compile requests                    623
Cache hits                          619
Cache misses                          4
Cache hits rate                   99.36 %
Cache timeouts                        0
Cache read errors                     0
Cache write errors                    0
Compilations                          4
```

다음과 같이 중요한 필드를 해석합니다.

| 필드 | 의미 |
|---|---|
| 컴파일 요청 | `sccache`에서 감지된 컴파일러 호출 |
| 캐시 적중 | 컴파일러를 실행하지 않고 복원된 요청 |
| 캐시 미적중 | 컴파일이 필요한 요청 |
| 컴파일 | 실제로 실행된 컴파일러 프로세스 |
| 캐시할 수 없는 호출 | `sccache`가 의도적으로 우회한 호출 |
| 읽기/쓰기 오류 | 캐시 백엔드 오류; 로컬 또는 쓰기 가능한 빌드에서 0이 아닌 경우 조사 |
| 캐시 위치 | 활성 백엔드(예: 로컬 디스크 또는 다단계) |

새 툴체인, SDK 캐시, 아키텍처, 빌드 모드 또는 크게 변경된 소스 트리의 첫 번째 빌드에서는 낮은 적중률이 정상입니다. 동일한 커밋 및 구성으로 두 번째 빌드를 수행하여 캐시를 평가합니다.

어떤 수준이든 읽기 전용인 경우, `sccache` v0.16은 읽기 전용 빌드가 성공하더라도 시도된 쓰기를 쓰기 오류로 보고할 수 있습니다. 이는 태그 및 기타 간접적인 컨텍스트에 적용됩니다. 직접적인 기능 분기 푸시는 `READ_WRITE`를 보고해야 하며, 해당 빌드에서 쓰기 오류를 조사해야 합니다.

## 빠른 검증

### 로컬 재사용 확인

동일한 클린 빌드를 두 번 실행합니다.

```bash
SIMANEAT_SCCACHE=on SIMANEAT_SCCACHE_ZERO_STATS=ON \
  ./build.sh --dev-only --clean --no-dist

SIMANEAT_SCCACHE=on SIMANEAT_SCCACHE_ZERO_STATS=ON \
  ./build.sh --dev-only --clean --no-dist
```

두 번째 실행 시에는 훨씬 더 높은 성공률을 보일 것입니다. 정확한 수치는 소스 변경, 컴파일러 테스트, 생성된 파일 및 선택한 빌드에 따라 달라집니다.

### Vulcan 구성 확인

GitHub Actions 빌드 로그에서 다음 항목을 확인합니다.

```text
sccache enabled: sccache <version>
sccache local cache: <path> (<limit>)
sccache remote cache: s3://<bucket>/<prefix> (READ_ONLY|READ_WRITE)
```

그런 다음 최종 통계에 예상치 못한 읽기, 쓰기, 시간 초과 또는 캐시 오류가 없는지 확인합니다.

## 문제 해결

### `sccache`가 활성화되지 않음

- 빌드에 `build.sh`가 사용되는지 확인합니다.
- `SIMANEAT_SCCACHE`가 `off`로 설정되어 있지 않은지 확인합니다.
- `SIMANEAT_SCCACHE=on`으로 다시 실행하여 부트스트랩 오류가 치명적인 오류가 되도록 합니다.
- `curl`, `tar` 및 `sha256sum` 또는 `shasum`이 사용 가능한지 확인합니다.

### 두 번째 로컬 빌드에서도 여전히 누락됨

- 두 빌드 모두 동일한 컴파일러, SDK, 빌드 모드 및 플래그를 사용하는지 확인합니다.
- `SCCACHE_DIR`이 동일한 영구 디렉터리를 가리키는지 확인합니다.
- 타임스탬프 또는 변경되는 절대 경로가 포함된 생성된 입력을 찾습니다.
- `Non-cacheable calls` 및 `Unsupported compiler calls`를 확인합니다.
- `SCCACHE_CACHE_SIZE`로 인해 캐시가 삭제되지 않았는지 확인합니다.

### Vulcan에서 원격 적중 횟수가 0으로 표시됨

- 선택한 `develop` 또는 `main` 기준선이 채워졌는지 확인합니다.
- 기능 브랜치의 경우 로그에 예상되는 기본 브랜치와 해당 브랜치별 인코딩된 접두사가 표시되는지 확인합니다.
- 아키텍처, SDK 캐시 식별자 및 빌드 모드를 비교합니다.
- 로그에 예상되는 버킷과 접두사가 표시되는지 확인합니다.
- 콜드 네임스페이스를 정상으로 처리하고 두 개의 동일한 빌드를 비교합니다.

### S3 시작 시 `AccessDenied` 오류 발생

캐시 역할에는 `.sccache_check` 프로브에 대한 접두사 범위의 `s3:ListBucket`와 함께 객체 권한이 필요합니다.

- 읽기 권한: `GetObject`
- 쓰기 권한: `GetObject` 및 `PutObject`

두 역할 모두 해당 KMS 권한도 필요합니다. 임시 해결 방법으로 EC2 러너 역할에 S3 또는 KMS 권한을 추가하지 말고 Vulcan의 GitHub OIDC 캐시 역할을 수정합니다.

### 컴파일러 오류 진단 시 캐시를 우회합니다.

```bash
SIMANEAT_SCCACHE=off ./build.sh --all --clean
```

오류가 계속 발생하면 캐시된 컴파일러 출력으로 인해 발생한 것이 아닙니다.

## 소유권 및 진실의 근원

| 문제점 | 출처 |
|---|---|
| 로컬 부트스트랩, 버전, 체크섬, 캐시 기본값 | `scripts/configure_sccache.sh` |
| CMake 런처 통합 및 통계 | `build.sh` |
| 브랜치 역할 선택 및 캐시 네임스페이스 | `.github/workflows/vulcan-ci.yml` |
| 재사용 가능한 Vulcan 워크플로 입력 및 OIDC 설정 | `sima-neat/.github` |
| S3, KMS, 수명 주기 및 캐시 IAM 역할 | `sima-neat/vulcan` |

캐시 스키마, 툴체인 호환성 범위 또는 액세스 모델을 변경할 때 영향을 받는 모든 저장소와 이 페이지를 함께 업데이트하십시오.
