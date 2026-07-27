---
title: Send Video
description: VideoSender H.264 and H.265 RTP/UDP wire formats
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# Send Video

Use `VideoSender` when a Graph should send video to an external receiver. `VideoSender` returns a reusable `Graph` fragment, so add it with `Graph::add(...)`.

`VideoSender` sends H.264 or H.265 over RTP/UDP. Raw input is encoded as H.264;
encoded H.264 and H.265 input is forwarded without re-encoding. H.264 uses RTP
payload type 96 by default, while H.265 uses 98. The default UDP port rule is
`video_port_base + channel`, with `video_port_base = 9000`.
If the receiver runs behind container port remapping, pass the mapped host and a matching `video_port_base` from the app.

## Raw Frames

Use the raw path when the pipeline input to `VideoSender` is raw video frames. Neat converts, encodes, packetizes, and sends:

```text
VideoConvert -> H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput
```

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

Python:

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

## Encoded H.264 or H.265

For encoded input, pass the stream codec to the passthrough factory. Neat
parses, packetizes, and sends the stream without re-encoding.

| Codec | C++ factory | Python factory | Default RTP payload type |
| --- | --- | --- | ---: |
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |

MJPEG passthrough is rejected: the sender has no RTP/JPEG packetizer.

H.265 example:

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

### Fan out encoded RTSP to inference and preview

When one encoded RTSP source feeds both decoding/inference and `VideoSender`, connect the source directly to the sender. For a live preview such as Insight, set the encoded sender edge to `RealtimeLatestByStream`:

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

The sender branch stays before `SimaDecode`, so it does not re-encode video or copy decoded frames to CPU. With `RealtimeLatestByStream`, the fused sender branch keeps at most one pending encoded access unit and replaces stale data if UDP egress slows. The default edge policy remains lossless and can backpressure the shared encoded source, including its decoder branch. Use the default only when preserving every access unit is more important than keeping live inference fresh.
