---
title: 비디오 보내기
description: VideoSender 원시 인코딩 및 H.264, H.265, MJPEG RTP/UDP 출력
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# 비디오 전송

외부 수신 장치로 비디오를 전송해야 할 때 `VideoSender`를 사용합니다. `VideoSender`는 재사용 가능한 `Graph` 조각을 반환하므로, `Graph::add(...)`를 사용하여 추가합니다.

`VideoSender`는 H.264, H.265 또는 MJPEG를 RTP/UDP로 전송합니다. `FromRaw`는 `SimaEncodeOptions.type`으로 선택한 코덱으로 원시 프레임을 인코딩하며 기본값은 H.264입니다. `Passthrough`는 이미 인코딩된 프레임을 재인코딩 없이 전송합니다. 기본 UDP 포트는 `video_port_base + channel`이며 `video_port_base = 9000`입니다.

| 코덱 | 원시 입력 인코더 유형 | 인코딩 입력 패스스루 유형 | 기본 RTP 페이로드 유형 |
| --- | --- | --- | ---: |
| H.264 | `SimaEncodeType::H264` | `RtspCodec::H264` | 96 |
| H.265 | `SimaEncodeType::H265` | `RtspCodec::H265` | 98 |
| MJPEG | `SimaEncodeType::MJPEG` | `RtspCodec::MJPEG` | 26 |

Insight의 RTP/WebRTC 뷰어는 H.264와 H.265를 지원합니다. MJPEG 출력에는 호환되는 RTP/JPEG 수신기가 필요하며 이 뷰어에서는 지원하지 않습니다.

수신기가 컨테이너 포트 매핑 뒤에서 실행되면 앱에서 매핑된 호스트와 그에 맞는 `video_port_base`를 전달합니다.

## 원본 프레임

`VideoSender`에 대한 파이프라인 입력이 원본 비디오 프레임인 경우 원본 경로를 사용합니다. Neat는 안전한 인코더 입력을 자동으로 선택합니다.

```text
NV12 in a compatible DMA-BUF:
SimaEncode -> codec parser -> RTP payloader -> UdpOutput

CPU input or raw frames requiring conversion:
Convert/upload into encoder DMA-BUF -> SimaEncode -> codec parser -> RTP payloader -> UdpOutput
```

코덱은 인코더, 파서, RTP 페이로드 생성기를 함께 선택합니다. 사용 중단 예정인 `H264RtpUdpFromRaw(...)` 팩토리도 계속 사용할 수 있으며 H.264 기본값을 유지합니다. 호환되는 NV12 DMA-BUF 입력은 기존 할당을 유지합니다. CPU 메모리의 NV12는 업로드가 필요하며 RGB, BGR, 회색조, I420은 변환이 필요합니다. Neat는 인코더 입력 경계에서 최종 DMA 메모리에 직접 기록하므로 인코더 내부에서 두 번째 복사를 준비하지 않습니다. 앱에서 메모리 백엔드를 선택할 필요가 없습니다.

### 원시 프레임의 기하학적 구조 및 레이아웃

`width` 및 `height`는 보이는 이미지의 가로 및 세로 크기입니다. 이 값들은 8, 16 또는 32의 배수일 필요가 없습니다. NV12 및 I420 4:2:0 형식의 경우, 가로 및 세로 크기는 모두 양수이고 짝수여야 합니다. 활성 코덱, 프로필, 레벨 및 하드웨어는 나머지 최소 및 최대 제한을 정의합니다. 예를 들어, `680x382`, `672x384`, `642x480`은 설치된 인코더가 지원하는 경우 유효한 크기입니다.

RTP/JPEG는 크기 제한이 더 엄격합니다. MJPEG 원시 전송의 각 축 크기는 8의 배수이며 8..2040 범위여야 합니다. 인코딩된 JPEG 패스스루에는 표준 허프만 테이블을 사용하는 베이스라인 JPEG와 RTP/JPEG로 표현할 수 있는 크기가 필요합니다. 독립 인코딩과 RTP 전송에는 서로 다른 크기 제한이 적용됩니다.

하드웨어 저장 정렬은 표시 크기와 별개입니다. Neat는 요청한 크기를 caps에 유지하고 하드웨어에 필요한 피치와 저장 높이로 인코더 메모리를 생성합니다. 사용자 지정 물리 레이아웃을 가진 원시 버퍼에는 기준이 되는 평면 오프셋과 스트라이드를 포함한 `GstVideoMeta`가 있어야 합니다. 이 메타데이터가 없으면 협상된 GStreamer 레이아웃을 사용합니다. 속성 기반 파일 입력의 각 버퍼에는 조밀하게 패킹된 프레임이 정확히 하나 있어야 합니다. 잘못되거나 잘리거나 지원되지 않는 레이아웃은 일부만 복사되지 않고 동기적으로 실패합니다.

```cpp
simaai::neat::Graph graph;
const int channel = 0;

simaai::neat::SimaEncodeOptions encode;
encode.type = simaai::neat::SimaEncodeType::H265;
encode.fps = 30;
encode.bitrate_kbps = 2500;
encode.gop_length = 30;
auto opt = simaai::neat::nodes::groups::VideoSenderOptions::FromRaw(encode);
opt.host = "127.0.0.1";
opt.channel = channel;
opt.video_port_base = 9000;

graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

파이썬:

```python
channel = 0

encode = pyneat.SimaEncodeOptions()
encode.type = pyneat.SimaEncodeType.H265
encode.fps = 30
encode.bitrate_kbps = 2500
encode.gop_length = 30
opt = pyneat.VideoSenderOptions.from_raw(encode)
opt.host = "127.0.0.1"
opt.channel = channel
opt.video_port_base = 9000

graph = pyneat.Graph()
graph.add(pyneat.groups.video_sender(opt))
```

원시 전송은 입력 프레임에서 해상도를 감지하고 크기 조정 없이 유지합니다. 다른 해상도로 전송하려면 새 실행을 시작합니다. 레거시 `H264RtpUdpFromRaw(width, height, fps)` 팩토리는 고정 입력 크기를 유지합니다.

MJPEG를 사용하려면 C++에서는 `SimaEncodeType::MJPEG`, Python에서는 `pyneat.SimaEncodeType.MJPEG`를 선택하고 `quality`를 1~100으로 설정합니다. 비트레이트, 속도 제어, 프로파일, 레벨, GOP, IDR 설정은 지정하지 않습니다. `FromRaw`는 전달한 옵션을 복사하므로 팩토리를 호출하기 전에 설정해야 합니다.

기존 `opt.encoder`의 비트레이트/프로파일/레벨 재정의는 원시 H.264/H.265 전송에 계속 적용됩니다. `sync=true`는 타임스탬프에 맞춰 전송하며 기본값 `false`는 해당 클록 대기 없이 전송합니다. `async=true`는 UDP 싱크 시작 시 첫 버퍼를 기다릴 수 있게 하며 기본값은 `false`입니다. Python에서는 이 옵션을 `async_`로 제공합니다.

## 인코딩된 프레임

인코딩된 입력의 경우, 스트림 코덱을 패스스루 팩토리에 전달합니다. Neat는 스트림을 재인코딩하지 않고 파싱, 패킷화하여 전송합니다.

| 코덱 | C++ 팩토리 | Python 팩토리 | 기본 RTP 페이로드 유형 |
| --- | --- | --- | ---: |
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |
| MJPEG | `Passthrough(RtspCodec::MJPEG)` | `passthrough(pyneat.RtspCodec.MJPEG)` | 26 |

패스스루는 인코더를 생성하지 않습니다. 호출자가 선택한 코덱은 인코딩된 입력과 일치해야 하며 인코더 설정은 적용되지 않습니다.

RTSP 소스가 이 송신기와 `SimaDecode`에 모두 공급되는 경우, 기본값 `async=false`에서는 Core가 송신기를 소스 파이프라인 안의 GStreamer `tee` 뒤에 배치하므로 소스, 송신기, 디코더가 함께 시작, 중지, 실패합니다. 별도의 파이프라인으로 유지하려면 `async=true`(Python에서는 `async_`)를 설정합니다.

H.265 예시:

```cpp
auto opt = simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(
    simaai::neat::nodes::groups::RtspCodec::H265);
opt.host = "127.0.0.1";
opt.channel = 0;
graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

```python
opt = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.H265)
opt.host = "127.0.0.1"
opt.channel = 0
graph.add(pyneat.groups.video_sender(opt))
```

### 인코딩된 RTSP를 추론 및 미리보기로 분산 전송

하나의 인코딩된 RTSP 소스가 디코딩/추론 및 `VideoSender` 모두에 연결되는 경우, 소스를 직접 전송 장치에 연결합니다. Insight와 같은 실시간 미리 보기를 위해 인코딩된 전송 장치를 `RealtimeLatestByStream`으로 설정합니다.

```cpp
simaai::neat::GraphLinkOptions video_link;
video_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;

graph.connect(encoded_source, decoder);
graph.connect(decoder, detector, detector_link);
graph.connect(encoded_source, video_sender, video_link);
```

```python
video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

graph.connect(encoded_source, decoder)
graph.connect(decoder, detector, detector_link)
graph.connect(encoded_source, video_sender, video_link)
```

송신 브랜치는 `SimaDecode` 이전 상태를 유지하므로 비디오를 다시 인코딩하거나 디코딩된 프레임을 CPU로 복사하지 않습니다. `RealtimeLatestByStream`을 사용하면 융합된 송신 브랜치는 최대 하나의 대기 중인 인코딩된 액세스 유닛을 유지하고 UDP 전송 속도가 느려지면 오래된 데이터를 대체합니다. 기본 엣지 정책은 무손실 상태를 유지하며, 디코더 브랜치를 포함한 공유된 인코딩 소스에 역압력을 가할 수 있습니다. 모든 액세스 유닛을 보존하는 것이 실시간 추론을 최신 상태로 유지하는 것보다 더 중요할 때만 기본 설정을 사용하십시오.
