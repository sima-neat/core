---
title: Send Video
description: VideoSender H.264 RTP/UDP wire format
sidebar_position: 5
---

# Send Video

Use `VideoSender` when a Graph should send video to an external receiver. `VideoSender` returns a reusable `Graph` fragment, so add it with `Graph::add(...)`.

The current wire format is H.264 over RTP/UDP. The default UDP port rule is `video_port_base + channel`, with `video_port_base = 9000`.
If the receiver runs behind container port remapping, pass the mapped host and a matching `video_port_base` from the app.

## Raw Frames

Use the raw path when the pipeline input to `VideoSender` is raw video frames. NEAT will convert, encode, packetize, and send:

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

## Encoded H.264

Use the encoded path when the upstream pipeline already produces H.264. NEAT will parse, packetize, and send without re-encoding:

```text
H264Parse -> H264Packetize -> UdpOutput
```

```cpp
simaai::neat::Graph graph;
const int channel = 0;

auto opt = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
opt.host = "127.0.0.1";
opt.channel = channel;
opt.video_port_base = 9000;

graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

Python:

```python
channel = 0

opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()
opt.host = "127.0.0.1"
opt.channel = channel
opt.video_port_base = 9000

graph = pyneat.Graph()
graph.add(pyneat.groups.video_sender(opt))
```
