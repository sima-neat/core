---
title: "MPK 계약"
description: "모델 아카이브 수집 계약, 검증 및 보안 규칙"
sidebar_position: 1
slug: /develop-apps/contribute/mpk_contract
---

# MPK 계약

본 문서는 모델 아카이브 로드 및 MPK 계약 파싱에 대한 권한 있는 계약을 정의합니다.

## 범위

본 계약은 다음 항목에 적용됩니다.

- `src/model/ModelPack.cpp`
- `src/model/ModelArchiveLoader.cpp`
- `src/model/internal/ModelArchiveLoader.h`
- `src/pipeline/internal/sima/MpkContract.cpp`

## 허용되는 아카이브 형식

허용되는 패키지 확장자: 정확히 소문자 `.tar.gz`만 허용됩니다. `.mpk`, `.tgz`, `.tar` 및 확장자 없는 `.gz`는 tar 검사 전에 거부됩니다.

아카이브 요구 사항:

- 아카이브는 tar 스트림으로 읽을 수 있어야 합니다.
- 아카이브 크기는 로더에서 구성된 `max_archive_bytes` 이하이어야 합니다.
- 항목 수는 로더에서 구성된 `max_entries` 이하이어야 합니다.
- 항목 페이로드 크기는 로더에서 구성된 `max_entry_bytes` 이하이어야 합니다.

## 허용되는 레이아웃

추출을 위해 허용되는 파일은 일반 파일뿐입니다. 디렉터리 항목은 허용되지만 무시됩니다.

허용되는 추출 파일 클래스:

- JSON 구성 파일(`*.json`) -> `etc/` 아래에 추출됨
- 공유 객체(`*.so`) -> `lib/` 아래에 추출됨
- ELF 바이너리(`*.elf`) -> `share/` 아래에 추출됨

필수 패키지 콘텐츠:

- MPK 추론 계약(`mpk.json` 또는 `*_mpk.json`)
- 런타임에서 필요한 로더 측 스테이지/구성 JSON
- 최소 하나의 모델 바이너리 아티팩트(`*.elf` 또는 `*.so`)

## 추출 안전 규칙

추출은 실패 시 안전하게 종료되어야 합니다.

거부되는 경로 형식:

- 절대 경로(예: `/etc/passwd`)
- 트래버스 세그먼트(`..`)
- Windows 드라이브 접두사(`C:`)
- 혼합된 구분 기호 트래버스 형식(`..\\`, `..//`)
- 유효하지 않은 UTF-8 경로 바이트
- 유니코드 슬래시/백슬래시 혼동 가능 문자(예: `U+FF0F`, `U+2215`, `U+FF3C`)
- 트래버스 유사 경로에서 사용되는 유니코드 점 혼동 가능 문자(예: `U+FF0E`, `U+2024`, `U+FE52`)

거부되는 항목 유형:

- 심볼릭 링크 항목
- 하드 링크 항목
- 장치 항목
- FIFO 항목

추출 동작:

- 아카이브 경로를 파일 시스템 출력 경로에 직접 쓰지 않음
- 먼저 아카이브 항목 경로를 정규화하고 검증
- 승인된 항목을 콘텐츠 스트림으로 제어된 임시 루트에 추출
- 추출 루트 외부에는 쓰기 금지
- 정규화된 tar 경로가 중복되면 `invalid_archive`로 거부
- tar 헤더/체크섬 손상은 `invalid_archive`로 거부

## JSON 및 시퀀스 검증

`pipeline_sequence.json`은 다음을 만족해야 합니다.

- 비어 있지 않은 `pipelines` 배열이 있는 JSON 객체
- 첫 번째 파이프라인 객체에는 비어 있지 않은 `sequence` 배열이 포함되어야 합니다.
- 각 스테이지 항목에는 다음이 포함되어야 합니다.
  - `sequence_id` (정수)
  - `name` (비어 있지 않은 문자열)
  - `pluginId` (비어 있지 않은 문자열)
  - `configPath` (비어 있지 않은 문자열)
  - `processor` (비어 있지 않은 문자열)
  - `kernel` (비어 있지 않은 문자열)
- 중복된 스테이지 이름은 거부됩니다.
- 중복된 JSON 키는 거부됩니다.
- 로더 `max_json_depth`보다 깊은 JSON 중첩은 거부됩니다.
- 지원되지 않는 `kernel` 값은 거부됩니다.
- `input`의 스테이지 종속성은 다음만 참조해야 합니다.
  - `decoder` 또는
  - 안정적인 순서로 정렬된 이전 스테이지 이름

## 오류 분류

모든 모델 아카이브 또는 MPK 계약 데이터 수집 실패는 다음 클래스 중 하나에 매핑되어야 합니다.

- `invalid_archive`
- `path_traversal`
- `schema_error`
- `unsupported_version`
- `size_limit_exceeded`

공개 오류 메시지에는 분류의 결정성을 확인할 수 있도록 분류 체계 키가 포함되어야 합니다.

## 결정성 요구 사항

- 반복 실행 시 시퀀스 순서가 결정적이어야 합니다.
- `tests/assets/mpk` 아래의 픽스처 아카이브는 비트 단위로 재현 가능해야 합니다.
- `test-assets/model-archive/fixtures_manifest.json` 아래에서 생성된 픽스처 매니페스트 체크섬은 빌드 트리 보안 픽스처의 근거가 됩니다.

## 테스트 매핑 요구 사항

모든 부정적인 모델 아카이브/MPK 계약 테스트는 위의 분류 체계 키 중 하나를 확인해야 합니다.
