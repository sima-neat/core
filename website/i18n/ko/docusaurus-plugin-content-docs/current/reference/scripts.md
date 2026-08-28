---
title: "사용하는 도구 및 스크립트 목록"
description: "`core/scripts/`와 `core/tools/`에는 무엇이 들어 있으며, 각각 언제 사용하는가?"
sidebar_position: 90
---

# 사용하는 도구 및 스크립트 목록

이 프레임워크에는 두 개의 헬퍼 디렉터리가 포함되어 있습니다. 이 페이지는 해당 디렉터리의 목록입니다.

## `core/tools/` — 문서 및 빌드 지원 도구


| 스크립트 | 목적 |
|---|---|
| `generate_api_docs.sh` | Doxygen XML 파일을 기반으로 doxygen2docusaurus를 실행하여 C++ API 레퍼런스 사이트에 대한 Markdown 파일을 생성합니다. 공개 헤더 파일을 편집한 후에 실행합니다. |

| `generate_python_api_docs.py` | `pyneat` 모듈의 독스트링에서 Python API 참조 Markdown을 생성합니다. |
| `generate_tutorial_docs.py` | (튜토리얼은 더 이상 지원되지 않으며, 이 스크립트는 곧 사용 중단될 예정입니다.) |
| `postprocess_d2d_links.py` | 생성 후 doxygen2docusaurus 링크 슬러그를 수정합니다. `generate_api_docs.sh`에 의해 자동으로 호출됩니다. |
| `strip_empty_programlisting.py` | doxygen2docusaurus를 혼동시키는 빈 `<programlisting>` 요소에 대한 임시 해결 방법입니다. |

| `compute_version.sh` | 프레임워크의 패키지 버전 문자열을 `deps/manifest.json`의 `package-version`에서 계산하고, 브랜치 빌드에 대한 Git 메타데이터를 추가합니다. CI 및 패키징에 사용됩니다. |
| `expand_code_tabs.py` | 튜토리얼 자료에서 다국어 탭을 확장합니다. |

| `run_clean_env.sh` | 깨끗한 셸 환경 내에서 명령을 실행합니다(상속된 `LD_*` / `PATH`로 인한 문제점을 방지). |
| `tutorial_quality_lint.py` / `tutorial_scorecard.py` | 튜토리얼 Markdown 파일을 검사하고 점수를 매깁니다. (튜토리얼과 함께 사용 중단 예정입니다.) |

공개 헤더를 편집할 때 일반적인 절차는 다음과 같습니다.

```bash
cd core
doxygen docs/doxygen/Doxyfile      # regenerate XML
bash tools/generate_api_docs.sh    # regenerate Markdown
cd website && yarn start           # preview the site
```

## `core/scripts/` — 저장소 수준 검사 및 개발 지원 도구


| 스크립트 | 목적 |
|---|---|
| `check_format.sh` | C++ 코드에 clang-format을 적용하고, 변경 사항이 있을 경우 빌드를 실패시킵니다. |

| `check_cmake_format.sh` / `check_cmake_style.py` | 파일에서 cmake-format 또는 lint를 실행하여 `CMakeLists.txt` 파일을 검사합니다. |
| `check_duplicate_includes.{sh,py}` | 헤더 파일에서 중복된 `#include` 구문을 감지합니다. |

| `check_internal_headers.sh` | `core/src/pipeline/internal/sima/` 파이프라인의 내부 계층이 공개/내부 경계를 준수하는지 확인합니다. |
| `run_cpp_tidy.sh` | 전체 코드에 대해 clang-tidy를 실행합니다. |
| `route_refactor_validation.sh` | 특정 경로 계획 기능에 대한 회귀 테스트를 수행합니다 (CI에서 호출). |
| `install_neat_plugins.sh` | 프레임워크의 GStreamer 플러그인을 시스템 플러그인 디렉터리에 설치합니다. |
| `install_codex_skill.sh` | Codex CLI의 NEAT 스킬을 설치합니다(개발 편의를 위한 기능). |
| `fix_devkit_runtime.sh` | 새로 설치된 개발 키트의 런타임 라이브러리/경로를 수정하고 코프로세서를 재시작합니다. `simaai-appcomplex.service`가 실행 중일 때만 M4를 부팅합니다. |
| `sync_neatdecoder.sh` / `use_neatdecoder.sh` | 번들된 디코더 빌드와 외부 디코더 빌드 간에 전환합니다. |

### `core/scripts/ci/`, `core/scripts/dev/`, `core/scripts/release/`

이 하위 디렉터리에는 각 워크플로에 속한 스크립트가 저장되어 있습니다. CI는 `ci/` 스크립트 세트를 실행하고, 개발자는 `dev/` 스크립트를 임시로 실행하며, 릴리스 엔지니어는 `release/` 스크립트를 실행합니다. 애플리케이션 코드에서 이 스크립트에 의존하지 마십시오.

## 새로 체크아웃한 소스 코드에서 문서 생성기를 실행합니다.

```bash
sudo apt-get install -y doxygen   # if not installed
cd core
doxygen docs/doxygen/Doxyfile      # generates docs/doxygen/out/xml/
bash tools/generate_api_docs.sh    # populates docs/reference/cppapi/
python3 tools/generate_python_api_docs.py   # populates docs/reference/pythonapi/
cd website && yarn install && yarn start    # serve at http://localhost:3000/
```

## 추가 정보

- “도구 및 스크립트” — 디자인 심층 분석의 제55조.
- `core/AGENTS.md` 저장소에는 pre-commit 실행해야 하는 도구에 대한 기여자 계약이 있습니다.
