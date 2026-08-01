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

Use the raw path when the pipeline input to `VideoSender` is raw video frames.
Neat selects the safe encoder ingress automatically:

```text
NV12 with a proven compatible boundary:
H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput

Other or unknown raw formats:
VideoConvert -> H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput
```

The automatic selection does not add an application option or change the
`H264RtpUdpFromRaw(...)` API. Proven NV12 in system or SiMaAI memory can feed
the H.264 encoder directly when the installed encoder advertises
`input-layout-aware=true`. RGB, BGR, grayscale, I420, unknown memory/layouts,
and inputs without a reliable format contract retain one conversion to NV12.

### Raw frame geometry and layout

`width` and `height` are the visible image dimensions. They do not need to be
multiples of 8, 16, or 32. For the NV12 and I420 4:2:0 formats, both dimensions
must be positive and even; the active codec, profile, level, and hardware define
the remaining minimum and maximum limits. For example, `680x382`, `672x384`,
and `642x480` are valid shapes when the installed encoder accepts them.

Hardware storage alignment is separate from visible geometry. Neat preserves
the requested dimensions in caps and allocates or stages into encoder surfaces
with the pitch and storage height required by the hardware. A raw buffer with a
custom physical layout must carry `GstVideoMeta` with authoritative plane
offsets and strides. Without that metadata, the negotiated GStreamer layout is
used; property-driven file input must contain exactly one tightly packed frame
per buffer. Invalid, truncated, or unsupported layouts fail synchronously
instead of being partially copied.

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
