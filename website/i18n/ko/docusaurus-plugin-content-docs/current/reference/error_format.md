---
title: "플러그인 오류 형식"
description: "플러그인 오류로 인해 발생하는 구조화된 오류 필드"
sidebar_position: 8
---

# 플러그인 오류 형식

플러그인이 치명적인 오류를 만나면 GStreamer 버스에 `GST_MESSAGE_ERROR` 메시지를 게시합니다. Neat은 이 오류를 `NeatError`로 격상시키고, 분류 및 렌더링을 위해 지원되는 구조화된 세부 정보를 보존합니다.

## 오류 영역 및 코드

다음은 플러그인에서 사용되는 권장 도메인/코드 목록입니다.

- 구성 파일 분석/검증: `GST_RESOURCE_ERROR_SETTINGS`
- 파일을 찾을 수 없습니다: `GST_RESOURCE_ERROR_NOT_FOUND`
- 디스패처를 사용할 수 없습니다: `GST_RESOURCE_ERROR_BUSY`; 사용하세요.
  `GST_RESOURCE_ERROR_NOT_FOUND` 오류는 디스패처별 진단 ID 또는 구조화된 디스패처 필드와 함께만 발생합니다.
- 할당 실패: `GST_RESOURCE_ERROR_NO_SPACE_LEFT`
- 캡스/협상 오류: `GST_STREAM_ERROR_FORMAT`
- 런타임 처리 오류: `GST_STREAM_ERROR_FAILED`

## 버전 관리가 되는 구조화된 상세 정보

새로운 Neat 플러그인 오류는 `simaai-neat-error`라는 이름의 `GstStructure`를 첨부합니다. 버전 1에는 부호 없는 정수 필드 `neat-schema-version=1`이 포함됩니다. 코어는 버전 1에서 구조화된 필드를 읽고, 알 수 없거나 누락된 버전의 경우 일반적인 GStreamer 도메인, 코드, 메시지 및 디버그 문자열로 되돌립니다. 이를 통해 향후 스키마가 이전의 가정으로 해석되는 것을 방지합니다.

일반 필드:
- `neat-schema-version`
- `neat-diagnostic-id`
- `neat-reason`
- `plugin`
- `node`
- `stage`
- `graph-id`
- `frame-id`
- `stream-id`
- `input-caps`
- `output-caps`
- `allocator`
- `dispatcher-error`

입력 용량 오류는 `actual-width`, `actual-height`, `actual-stride`,
`maximum-width`, `maximum-height`, `maximum-stride`, `resize-width`, `resize-height`,
`required-bytes`, `allocated-bytes` 및 `input-format`를 제공합니다.

입력 계약 오류는 `input-name`, `segment-name`, `required-bytes`, `actual-bytes`,
`expected-shape`, `expected-layout`, `expected-dtype`, `received-shape`, `received-layout` 및
`received-dtype`를 제공합니다. 레이아웃 필드는 `[3, 224, 224]` (`CHW`) 및
`[224, 224, 3]` (`HWC`)와 같은 모양을 구별합니다.

이전 플러그인은 디버그 문자열에 공백으로 구분된 `key='value'` 목록을 배치할 수 있습니다. Core는 호환성을 위해 해당 필드를 계속 사용합니다.

## 예시

```text
simaai-neat-error, neat-schema-version=(uint)1,
neat-diagnostic-id=(string)neatprocesscvu.input_contract_mismatch,
plugin=(string)neatprocesscvu, node=(string)model_0,
expected-shape=(string)"[3, 224, 224]", expected-layout=(string)CHW,
expected-dtype=(string)Float32, received-shape=(string)"[224, 224, 3]",
received-layout=(string)HWC, received-dtype=(string)UInt8;
```

## 참고 사항

- 기본적으로 `NeatError::what()`에는 정규화된 오류 코드와 사용자에게 표시되는 메시지가 포함됩니다.
  컨텍스트, 시정 조치, 진단 ID를 포함합니다. 원시 GStreamer 메시지와 디버그 문자열은 생략합니다.
- `SIMA_NEAT_VERBOSE_LEVEL=2` 및 `SIMA_NEAT_VERBOSE_TOPICS=gstreamer`를 설정하여 짧은 시간 동안 테스트합니다.
  진단 실행입니다. 이 작업은 편집된 기술 세부 정보를 다음 항목에 추가합니다. `NeatError::what()` 및 `GraphReport.repro_note`. `NeatError::report()`는 진단을 위한 구조화된 인터페이스로 유지됩니다.
- `NEAT_LOG_LEVEL=debug`는 Neat Library 설정이 아닙니다.
- URI 사용자 정보 및 인증된 자격 증명 필드(`auth`, `playback-token`, `hdnts` 포함)
  `stream-key` 및 `tkn`은 보고서에 표시되는 파이프라인 문자열, 재현 명령어, 구조화된 세부 정보 및 JSON에서 삭제됩니다. 배포 관련 경로 및 미디어 주소를 확인하려면 지원 번들을 검토한 후 공유하십시오.
