---
title: "Neat Library 릴리스 노트"
sidebar_position: 3
---

# Neat Library 릴리스 노트

SiMa.ai Neat Library의 릴리스 노트입니다.

## 미출시

### 호환성이 깨지는 변경 사항

- Neat Library C++ ABI는 이제 버전 4이며, 공유 라이브러리의 SONAME은 `libsima_neat.so.4`입니다. 텐서는 이제 특징 추출기의 의미 메타데이터를 포함하고, 공개 GenAI 요청/결과 유형은 ASR 작업, 언어 및 프로브 메타데이터를 포함하며, `GraphLinkOptions`에는 실시간 허용 한도가 포함됩니다. C++ 애플리케이션과 플러그인을 다시 빌드하고 해당 Core 런타임 및 개발 패키지를 설치합니다.
- 실시간 그래프 구성은 이제 `GraphLinkOptions`, `Graph::connect()` 및 `Graph::build()`를 사용합니다. 미리보기 API인 `RealtimeGraphLinkOptions`, `connect_realtime()`, `build_fused_realtime_sources()` / `build_fused_realtime_source()` 및 `RealtimeEveryFrameByStream`은 제거되었습니다. `realtime_every_frame_by_stream`을 포함하는 저장된 그래프는 지원되는 정책으로 다시 생성해야 합니다. 자세한 내용은 [라이브 프래그먼트 연결](/develop-apps/development-workflow/graph/#connect-live-fragments)을 참조하십시오.

### 런타임 변경 사항

- C++ 및 Python에서 `SimaDecode` 및 `RtspDecodedInput`을 통해 기본 H.265/HEVC 디코딩을 사용할 수 있습니다. `RtspEncodedInput`는 디코딩하지 않고 파싱된 H.265 액세스 유닛을 제공합니다. H.265 입력은 HEVC Main 프로필, 8비트, 4:2:0을 사용해야 합니다. 코덱 선택기는 `H265` 및 `HEVC`를 모두 허용합니다. H.264 선택기는 `AVC`도 허용합니다. `FormatTag` / `pyneat.Format`은 인코딩된 그래프 경계에서 동일한 별칭을 허용하며, 여전히 `H264` 및 `H265`로 직렬화됩니다.
- `VideoSender`은 `VideoSenderOptions::Passthrough(codec)` / `pyneat.VideoSenderOptions.passthrough(codec)`를 통해 재인코딩 없이 UDP를 통해 RTP로 인코딩된 H.264 또는 H.265를 전달합니다. H.265는 기본적으로 RTP 페이로드 유형 98을 사용하고, H.264는 96을 유지합니다. `H264RtpUdpFromEncoded()`은 `Passthrough(RtspCodec::H264)`로 대체되어 더 이상 사용되지 않습니다.
- 원시 `VideoSender` 입력은 이제 시스템 또는 SiMaAI 메모리에서 NV12임이 확인되고 설치된 인코더가 `input-layout-aware=true`를 알릴 때 형식 변환을 자동으로 생략합니다. 다른 원시 형식, 알 수 없는 메모리/레이아웃 및 신뢰할 수 있는 형식 계약이 없는 입력은 기존 NV12 변환을 유지합니다. `H264RtpUdpFromRaw(...)` C++ 및 Python API는 변경되지 않았습니다.
- RTSP 입력은 `RtspEncodedInputOptions`와 `RtspDecodedInputOptions`의 코덱 중립적인 단일 `payload_type` 필드로 RTP 페이로드 유형을 선택합니다. `-1`은 코덱 기본값(H.264/H.265의 경우 96, MJPEG의 경우 26)을 선택하고, `0`은 페이로드 필터링을 비활성화하며, 양수 값은 지정한 페이로드를 선택합니다. `RtspEncodedInputOptions::h264_payload_type` 및 `mjpeg_payload_type`는 더 이상 사용되지 않으며, 해석된 페이로드를 변경하는 경우 런타임에서 한 번 경고합니다.
- 일반적인 `build()`는 이제 조건을 충족하는 라이브 팬인에 대해 융합 lowering을 자동으로 선택합니다. 직접 인코딩된 H.264 또는 H.265 `VideoSender` 분기는 디코딩 전에 융합되며, 디코딩된 프레임을 CPU로 복사하지 않습니다. 소스, 디코더 및 송신자의 코덱이 일치해야 하며, 일치하지 않는 쌍은 별도의 파이프라인 세그먼트에 남습니다. 라이브 미리 보기를 위해 해당 에지를 `RealtimeLatestByStream`으로 설정하면 느린 비디오 수신기가 디코더 분기에 역압력을 가하는 대신 오래된 액세스 유닛을 새것으로 교체합니다.

- MIPI/libcamera 소스 소유 그래프에 대한 C++ 및 Python `CameraInput` 문서와 튜토리얼을 추가했습니다. 여기에는 CVU/MLA 모델 경로 앞의 적응형 SiMaAI 메모리 핸드오프가 포함됩니다.
- `MetadataSender`은 이제 더 큰 JSON 메시지를 분할하여 UDP 페이로드를 1200바이트 이내로 유지합니다. 메타데이터 청크 재조합 기능을 포함한 버전으로 Insight를 업데이트하고, 이 Neat Library 버전과 함께 또는 그 전에 업데이트하십시오. 이전 버전의 Insight는 변경되지 않은 JSON 페이로드를 최대 1200바이트까지 계속 지원합니다.

### 그래프 구성 및 검증

- 그래프 구성은 이제 하나의 노드 객체를 하나의 논리적 정점으로 처리합니다. 중복 삽입 및 겹치는 조각 가져오기는 원자적으로 실패하며, 반복적인 `connect()` 호출은 기존 노드를 재사용하여 분기 처리를 수행합니다.
- 이제 모든 구체화된 파이프라인 세그먼트는 `build()` 중에 최종 GStreamer 이름을 검증합니다. 명시적인 `validate()` 호출은 필요하지 않습니다. 중복되거나 누락된 이름은 `misconfig.pipeline_shape`와 함께 실패하며, 잘린 파이프라인을 생성하지 않습니다.
- 사용자 지정 프래그먼트는 이제 모든 명시적 이름을 보고하고, 선언과 함께 이름이 지정된 패드 참조도 변환합니다. 이름 충돌은 자동으로 이름을 바꾸는 대신 거부됩니다.

| 릴리스 | 호환되는 Neat SDK | 참고 사항 |
| --- | --- | --- |
| 0.4.0 | 2.1.3.0 | [Neat Library 0.4.0](https://github.com/sima-neat/core/releases/tag/v0.4.0) |
| 0.3.0 | 2.1.2.3 | [Neat Library 0.3.0](https://github.com/sima-neat/core/releases/tag/v0.3.0) |
| 0.2.2 | 2.1.2.2 | [Neat Library 0.2.2](https://github.com/sima-neat/core/releases/tag/v0.2.2) |
| 0.2.1 | 2.1.2.1 | [Neat Library 0.2.1](https://github.com/sima-neat/core/releases/tag/v0.2.1) |
| 0.2.0 | 2.1.2 | [Neat Library 0.2.0](https://github.com/sima-neat/core/releases/tag/v0.2.0) |
| 0.1.0 | 2.0.0 | [Neat Library 0.1.0](https://github.com/sima-neat/core/releases/tag/v0.1.0) |
