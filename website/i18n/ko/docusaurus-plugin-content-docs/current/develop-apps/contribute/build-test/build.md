---
title: "구축하다"
description: "build.sh를 사용하여 소스 코드에서 SiMa.ai Neat을 빌드합니다."
sidebar_position: 1
slug: /develop-apps/contribute/build
---

# Neat을 구축하세요.

이 안내서는 Neat의 소스 코드 빌드 방법을 설명합니다.
미리 빌드된 패키지 설치에 대해서는 [Neat Library](/getting-started/neat-library/)를 참조하십시오.

`build.sh`는 지원되는 빌드 진입점입니다. 이 스크립트는 종속성 확인, 선택적 종속성 동기화, CMake 구성/빌드, 선택적 문서 생성, 설치 무결성 검사 및 패키징을 처리합니다.

## 개발 환경 구축

`build.sh`는 활성 환경을 자동으로 감지합니다.

- Modalix DevKit의 기본 환경
- Neat SDK 환경(크로스 컴파일)

두 환경 모두에서 동일한 `build.sh` 명령을 실행할 수 있습니다.

### 크로스 컴파일 필수 조건

일반적으로 크로스 컴파일은 DevKit에서 직접 빌드하는 것보다 빠르지만, 그 후에는 빌드된 아티팩트를 DevKit로 전송해야 합니다. 크로스 컴파일을 위해서는 Neat SDK가 필요합니다.

먼저 호스트 머신에 `sima-cli`를 설치한 다음 SDK를 설치하십시오.

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
sima-cli install sdk
```

`sima-cli`에서 메시지가 표시되면 SDK 옵션을 선택합니다.

그런 다음 SDK를 시작합니다.

```bash
sima-cli sdk elxr
```

그런 다음 SDK 내에 `sima-cli`를 설치하고, SDK 패치를 설치합니다.

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
source ~/.bash_profile
sima-cli install tools/sdk-patch
```

- SDK 설치는 Windows 및 Ubuntu에서 지원됩니다.
- Modalix DevKit에서 기본적으로 개발하는 경우 SDK 설치/패치 단계는 필요하지 않습니다.

## 빌드 옵션

지원되는 `build.sh` 옵션:

- `--dev-only`: 핵심 라이브러리와 헤더 파일만 빌드합니다(기본 설정).
- `--all`: 라이브러리, 테스트, 튜토리얼, Python 패키지 빌드; 문서 및 종속성 관리 기능 활성화.
- `--python`: 선택한 대상 외에 Python 바인딩(`pyneat`)을 구축합니다.
- `--install-neat-internals`, `--install-deps`: 빌드하기 전에 필요한 아티팩트를 다운로드하여 설치합니다.
- `--doc`: 문서만 빌드합니다.
- `--install`: 빌드/패키징 후, 생성된 아티팩트를 현재 환경에 설치합니다. 페어링된 Neat SDK 모드에서는, 이와 함께 페어링된 DevKit에 해당 아티팩트를 배포하고 설치합니다.
- `--no-dist`: 배포 패키지 생성을 건너
- `--clean`: 설정을 진행하기 전에 `build/`를 제거하세요.
- `--no-doc`: 문서 빌드 건너뛰기(`--all` 옵션이 활성화된 경우에도).
- `--no-node`: Node.js 설치를 건너
- `--install-deps-only`: 시스템 종속성과 종속성 헤더를 설치한 후 종료합니다.

## 컴파일러 캐시

`build.sh`는 `sccache`를 자동으로 활성화하며, `--clean` 후에도 캐시가 유지됩니다. 로컬 빌드는 사용자 로컬 디스크 캐시를 사용합니다. Vulcan은 `develop` 및 `main`에 대해 별도의 보호된 캐시를 제공합니다. 기능 브랜치는 가장 가까운 보호된 기본 브랜치에서 격리된 쓰기 가능 캐시를 생성하고, 해당 브랜치가 삭제될 때까지 유지합니다.

로컬 제어, 클라우드 액세스 규칙, 캐시 네임스페이스, 통계, 검증 및 문제 해결에 대한 자세한 내용은 [Neat sccache 사용법 요약](/develop-apps/contribute/sccache)를 참조하십시오.

## 일반적인 빌드 구성

핵심 라이브러리만 (기본 설정):

```bash
./build.sh
```

전체 빌드(라이브러리, 테스트, 튜토리얼, 문서, 휠, 패키징):

```bash
./build.sh --all
```

핵심 라이브러리 + Python 바인딩:

```bash
./build.sh --dev-only --python
```

문서만 해당:

이 명령어는 macOS에서도 작동합니다.

```bash
./build.sh --doc
```

문서 빌드 프로세스는 `build/autodoc/insight/neat_insight/openapi.json`에서 다운로드한 OpenAPI 사양을 기반으로 Insight API 참조 자료를 생성합니다. 로컬 개발 환경에서는 `INSIGHT_OPENAPI_SPEC`를 사용하여 기본 설정을 재정의할 수 있습니다.

```bash
INSIGHT_OPENAPI_SPEC=../insight/neat_insight/openapi.json ./build.sh --doc
```

상대 경로 오버라이드는 Core 저장소의 루트에서 확인되고, Docusaurus 생성기가 실행되기 전에 절대 경로로 변환됩니다. 선택한 파일이 존재하지 않으면 Insight API 생성 단계가 건너뛰고 해당 경로가 보고됩니다.

전체 빌드 정리:

```bash
./build.sh --all --clean
```

코어 부분을 빌드하지 않고 필요한 패키지를 설치합니다.

```bash
./build.sh --install-deps-only
```

## 결과

- 트리 구축: `build/`
- Docusaurus 사이트 출력물(문서 빌드 실행 시): `website/build/`
- 정상성 검사 접두사 설치: 빌드 중에 출력되는 고유한 임시 디렉터리(`${TMPDIR:-/tmp}/sima-neat-install-test.XXXXXX`); 성공 시에는 삭제하고, 실패 시에는 검사를 위해 보관합니다.
- Neat 패키지 아티팩트(`*.deb`)는 `--no-dist`를 사용하지 않는 한 Linux 전체 빌드에서 생성됩니다.
- 추가 패키지(`*extras.tar.gz`)는 `--no-dist`를 사용하지 않는 한 Linux 전체 빌드에서 생성됩니다.
- Python 빌드가 활성화되면 Python 휠 파일(`dist/*.whl`)이 생성됩니다.

Python 휠 패키지는 주 CMake 빌드에서 생성된 `_pyneat_core` 확장 모듈을 포함합니다. 휠을 생성하는 과정에서 두 번째 CMake 트리를 구성하거나 컴파일하지 않으므로, 라이브러리, DEB 패키지, 추가 아카이브, 휠은 모두 하나의 컴파일 결과물을 공유합니다.

## 프로필 및 CMake 옵션 구성

프레임워크의 최상위 `CMakeLists.txt` 파일은 빌드되는 내용과 빌드 방식을 제어하는 몇 가지 옵션을 제공합니다. 아래에 나열된 옵션들이 핵심적인 역할을 합니다.

### 프로필을 생성합니다.

이 프레임워크는 세 가지 명명된 프로필을 지원합니다.

| 프로필 | 사용 사례 | 무엇이 컴파일되었나요? |
|---------|----------|-----------------|
| **제작** | 고객에게 직접 제공되는 빌드 | 모든 공개 노드, 모델 아카이브 로딩, Modalix 백엔드, 최적화 |
| **개발자** | 프레임워크 엔지니어 | 프로덕션 환경 설정 + 디버그 노드 + 확장된 진단 기능 + 테스트 |
| **샌드박스** | 다중 테넌트 배포 | 프로덕션 환경 설정 + 강화된 모델 아카이브 보안 기본 설정 |

구성 시 `-DSIMA_NEAT_PROFILE=Production|Developer|Sandbox`를 선택하거나, `CMakeLists.txt`에서 기본값을 사용합니다.

### 일반적인 CMake 옵션

| 옵션 | 기본값 | 효과 |
|--------|---------|--------|
| `SIMA_NEAT_BUILD_TESTS` | `ON` (개발자) | gtest 테스트 스위트를 빌드합니다. 프로덕션 빌드 시 CI 속도를 높이기 위해 비활성화합니다. |
| `SIMA_NEAT_BUILD_TUTORIALS` | `OFF` | 튜토리얼 실행 파일을 빌드합니다. |
| `SIMA_NEAT_BUILD_PYTHON` | `ON` | `pyneat` 나노바인드 모듈을 빌드합니다. |
| `SIMA_NEAT_BUILD_INTERNALS` | `OFF` (공개) | 내부 연결 계층을 구축합니다(`core/src/pipeline/internal/sima/`). |
| `SIMA_NEAT_ENABLE_TVM_FALLBACK` | `ON` | MLA에서 처리할 수 없는 연산에 대해 TVM을 기반으로 하는 대체 커널을 컴파일합니다. |
| `SIMA_NEAT_ENABLE_RTSP` | `ON` | RTSP 소스/싱크 노드를 구축합니다. |
| `SIMA_NEAT_DEBUG_PLUGINS` | `OFF` | GStreamer 플러그인의 디버그 정보를 표준 출력으로 전달합니다. |
| `SIMA_NEAT_USE_SYSTEM_GSTREAMER` | `ON` (호스트) / `OFF` (교차) | 패키지에 포함하는 대신 시스템의 GStreamer와 연결하세요. |
| `SIMA_NEAT_WARN_AS_ERROR` | `OFF` | 컴파일 경고를 오류로 승격합니다. CI 환경에서 사용하는 것이 좋습니다. |

### 도구 모음의 조정 가능한 매개변수

Modalix를 대상으로 하는 크로스 컴파일의 경우:

```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/modalix.cmake \
  -DSIMA_NEAT_PROFILE=Production
```

호스트 측 개발을 위한 사항:

```bash
cmake -B build -DSIMA_NEAT_PROFILE=Developer
```

트리가 노출하는 내용을 나열하면 다음과 같습니다.

```bash
cmake -L -B build       # list all cache variables
cmake -LA -B build      # include advanced
```

최상위 수준의 `CMakeLists.txt` 파일은 옵션 이름에 대한 정확한 정보를 담고 있습니다.
