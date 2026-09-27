---
title: Send Video
description: VideoSender raw encoding and H.264, H.265 and MJPEG RTP/UDP output
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# Send Video

Use `VideoSender` when a Graph should send video to an external receiver. `VideoSender` returns a reusable `Graph` fragment, so add it with `Graph::add(...)`.

`VideoSender` sends H.264, H.265 or MJPEG over RTP/UDP. `FromRaw` encodes raw frames using `SimaEncodeOptions.type`, which defaults to H.264. `Passthrough` sends already encoded frames without re-encoding. The default UDP port is `video_port_base + channel`, with `video_port_base = 9000`.

| Codec | Raw encoder type | Encoded passthrough type | Default RTP payload type |
| --- | --- | --- | ---: |
| H.264 | `SimaEncodeType::H264` | `RtspCodec::H264` | 96 |
| H.265 | `SimaEncodeType::H265` | `RtspCodec::H265` | 98 |
| MJPEG | `SimaEncodeType::MJPEG` | `RtspCodec::MJPEG` | 26 |

Insight's RTP/WebRTC viewer supports H.264 and H.265. MJPEG output requires a compatible RTP/JPEG receiver; it is not supported by that viewer.

If the receiver runs behind container port remapping, pass the mapped host and a matching `video_port_base` from the app.

## Raw Frames

Use the raw path when the pipeline input to `VideoSender` is raw video frames.
Neat selects the safe encoder ingress automatically:

```text
NV12 in a compatible DMA-BUF:
SimaEncode -> codec parser -> RTP payloader -> UdpOutput

CPU input or raw frames requiring conversion:
Convert/upload into encoder DMA-BUF -> SimaEncode -> codec parser -> RTP payloader -> UdpOutput
```

The codec selects the encoder, parser and RTP payloader together. The deprecated `H264RtpUdpFromRaw(...)` factory remains available and preserves its H.264 defaults. Compatible NV12 DMA-BUF input retains its backing
allocation. CPU-backed NV12 requires an upload; RGB, BGR, grayscale and I420
require conversion. Neat performs that work at the encoder-input boundary,
writing into the final DMA surface rather than staging a second copy inside
the encoder. Applications do not need to select a memory backend.

### Raw frame geometry and layout

`width` and `height` are the visible image dimensions. They do not need to be
multiples of 8, 16, or 32. For the NV12 and I420 4:2:0 formats, both dimensions
must be positive and even; the active codec, profile, level, and hardware define
the remaining minimum and maximum limits. For example, `680x382`, `672x384`,
and `642x480` are valid shapes when the installed encoder accepts them.

RTP/JPEG has stricter limits: MJPEG raw sending requires dimensions divisible by eight and within 8..2040 per axis. Encoded JPEG passthrough requires baseline JPEG with standard Huffman tables and dimensions representable by RTP/JPEG. Standalone encoding and RTP transport have separate size limits.

Hardware storage alignment is separate from visible geometry. Neat preserves
the requested dimensions in caps and produces encoder surfaces
with the pitch and storage height required by the hardware. A raw buffer with a
custom physical layout must carry `GstVideoMeta` with authoritative plane
offsets and strides. Without that metadata, the negotiated GStreamer layout is
used; property-driven file input must contain exactly one tightly packed frame
per buffer. Invalid, truncated, or unsupported layouts fail synchronously
instead of being partially copied.

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

Python:

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

Raw sending detects resolution from the input frames and preserves it without resizing.
Start a new run to send raw frames at a different resolution.
The legacy `H264RtpUdpFromRaw(width, height, fps)` factory retains its fixed input dimensions.

For MJPEG, select `SimaEncodeType::MJPEG` in C++ or `pyneat.SimaEncodeType.MJPEG` in Python and set `quality` from 1 to 100. Leave bitrate, rate control, profile, level, GOP and IDR settings unset. `FromRaw` copies the supplied options, so configure them before calling the factory.

The existing `opt.encoder` bitrate/profile/level overrides remain effective for raw H.264/H.265 senders. `sync=true` schedules sends against timestamps; the default `false` sends without that clock wait. `async=true` allows UDP sink startup to wait for its first buffer; the default is `false`. Python exposes this option as `async_`.

## Encoded frames

For encoded input, pass the stream codec to the passthrough factory. Neat
parses, packetizes, and sends the stream without re-encoding.

| Codec | C++ factory | Python factory | Default RTP payload type |
| --- | --- | --- | ---: |
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |
| MJPEG | `Passthrough(RtspCodec::MJPEG)` | `passthrough(pyneat.RtspCodec.MJPEG)` | 26 |

Passthrough creates no encoder. The caller's codec selection must match the encoded input; encoder settings do not apply.

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

### Encoder level

Leave `SimaEncodeOptions::level` unset for normal use. Both `SimaEncode` and `VideoSenderOptions::FromRaw` then leave level selection to the backend. The current backend starts at 4.0 and raises it when the stream settings require it; it does not select the lowest possible level. This does not resize frames or change their configured cadence.

An explicit level is optional and is forwarded to the backend, which may raise it if necessary. It is not a strict receiver-compatibility limit. For a sender created with `FromRaw`, an empty `options.encoder.level` retains backend selection. The deprecated fixed-resolution H.264 factory keeps its existing 4.0 default.
