---
title: "Python 바인딩(pyneat)"
description: "기여자로서 pyneat 바인딩을 빌드, 테스트하고 패키징합니다."
sidebar_position: 3
slug: /develop-apps/contribute/python_bindings
---

# Python 바인딩 (`pyneat`)

이 페이지는 `pyneat`에 기여하거나 유지 관리하는 사람들을 위한 것입니다.

`pyneat`는 SiMa.ai의 Neat에 대한 Python 바인딩 레이어이며, `nanobind`를 사용하여 구축되고 `scikit-build-core`로 패키징되었습니다.

생성된 API 문서는 [Python API 레퍼런스](/reference/pythonapi/modules/pyneat)를 참조하십시오.

## 필수 조건

`pyneat`는 C++ 라이브러리와 동일한 네이티브 종속성을 사용하며, 여기에는 다음이 포함됩니다.

- GStreamer 개발/런타임 패키지
- OpenCV 개발/런타임 패키지
- C++ 툴체인(`cmake`, 컴파일러, `pkg-config`)

호스트 설정 지침은 [구축하다](/develop-apps/contribute/build)를 참조하십시오.

## 소스 코드에서 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

개발을 위한 수정 가능한 설치:

```bash
python -m pip install -e .[dev]
```

## 테스트 실행

```bash
pytest -q python/tests
```

## 패키징

저장소 루트에 있는 `pyproject.toml` 파일은 `pyneat`의 휠/소스 배포 패키지 빌드 구성을 정의합니다.
