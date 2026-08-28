---
title: "비디오 보내기"
description: "VideoSender의 H.264 및 H.265 RTP/UDP 전송 형식"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# 비디오 전송

외부 수신 장치로 비디오를 전송해야 할 때 `VideoSender`를 사용합니다. `VideoSender`는 재사용 가능한 `Graph` 조각을 반환하므로, `Graph::add(...)`를 사용하여 추가합니다.

`VideoSender`는 RTP/UDP를 통해 H.264 또는 H.265를 전송합니다. 원본 입력은 H.264로 인코딩됩니다. 인코딩된 H.264 및 H.265 입력은 재인코딩 없이 전달됩니다. H.264는 기본적으로 RTP 페이로드 유형 96을 사용하고, H.265는 98을 사용합니다. 기본 UDP 포트 규칙은 `video_port_base + channel`이며, `video_port_base = 9000`입니다. 수신 장치가 컨테이너 포트 매핑 뒤에서 실행되는 경우, 매핑된 호스트와 일치하는 `video_port_base`를 앱에서 전달합니다.

## 원본 프레임

`VideoSender`에 대한 파이프라인 입력이 원본 비디오 프레임인 경우 원본 경로를 사용합니다. Neat는 안전한 인코더 입력을 자동으로 선택합니다.

```text
NV12 with a proven compatible boundary:
H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput

Other or unknown raw formats:
VideoConvert -> H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput
```

자동 선택 기능은 애플리케이션 옵션을 추가하거나 `H264RtpUdpFromRaw(...)` API를 변경하지 않습니다. 시스템 또는 SiMaAI 메모리에 저장된 검증된 NV12 데이터를 사용하여 설치된 인코더가 `input-layout-aware=true`를 지원하는 경우 H.264 인코더에 직접 데이터를 전송할 수 있습니다. RGB, BGR, 그레이스케일, I420, 알 수 없는 메모리/레이아웃, 그리고 신뢰할 수 있는 형식 계약이 없는 입력은 NV12로 한 번 변환됩니다.

### 원시 프레임의 기하학적 구조 및 레이아웃

`width` 및 `height`는 보이는 이미지의 가로 및 세로 크기입니다. 이 값들은 8, 16 또는 32의 배수일 필요가 없습니다. NV12 및 I420 4:2:0 형식의 경우, 가로 및 세로 크기는 모두 양수이고 짝수여야 합니다. 활성 코덱, 프로필, 레벨 및 하드웨어는 나머지 최소 및 최대 제한을 정의합니다. 예를 들어, `680x382`, `672x384`, `642x480`은 설치된 인코더가 지원하는 경우 유효한 크기입니다.

하드웨어 저장 정렬은 보이는 기하학적 구조와 별개입니다. Neat는 요청된 크기를 캡슐화하고 하드웨어에서 요구하는 피치 및 저장 높이를 사용하여 인코더 표면에 할당하거나 스테이징합니다. 사용자 지정 물리적 레이아웃을 가진 원시 버퍼는 권한 있는 플레인 오프셋 및 스트라이드를 포함하는 `GstVideoMeta`를 포함해야 합니다. 해당 메타데이터가 없으면 협상된 GStreamer 레이아웃이 사용됩니다. 속성 기반 파일 입력은 버퍼당 정확히 하나의 밀집된 프레임을 포함해야 합니다. 유효하지 않거나 잘리거나 지원되지 않는 레이아웃은 부분적으로 복사되는 대신 동기적으로 실패합니다.

```cpp
simaai::neat::Graph graph;
const int channel = 0;

auto opt = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
    width, height, fps);
opt.host = "127.0.0.1";
opt.channel = channel;
opt.video_port_base = 9000;
opt.encoder.bitrate_kbps = 2500;

graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

파이썬:

```python
channel = 0

opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
    width=1920,
    height=1080,
    fps=30,
)
opt.host = "127.0.0.1"
opt.channel = channel
opt.video_port_base = 9000
opt.encoder.bitrate_kbps = 2500

graph = pyneat.Graph()
graph.add(pyneat.groups.video_sender(opt))
```

## 인코딩된 H.264 또는 H.265

인코딩된 입력의 경우, 스트림 코덱을 패스스루 팩토리에 전달합니다. Neat는 스트림을 재인코딩하지 않고 파싱, 패킷화하여 전송합니다.

| 코덱 | C++ 팩토리 | Python 팩토리 | 기본 RTP 페이로드 유형 |
|---|---|---|---|
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |

MJPEG 패스스루는 거부됩니다. 송신자는 RTP/JPEG 패킷화 장치가 없습니다.

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
