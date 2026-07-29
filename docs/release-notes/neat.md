---
title: Neat Library Release Notes
sidebar_position: 3
---

# Neat Library Release Notes

Release notes for the SiMa.ai Neat Library.

## Unreleased

### Breaking changes

- The Neat Library C++ ABI is now 4 and the shared-library SONAME is `libsima_neat.so.4`. Public GenAI request/result types now carry ASR task, language, and probe metadata, and `GraphLinkOptions` contains realtime admission limits. Rebuild C++ applications and plugins and install matching Core runtime and development packages.
- Realtime graph composition now uses `GraphLinkOptions`, `Graph::connect()`, and `Graph::build()`. The preview `RealtimeGraphLinkOptions`, `connect_realtime()`, `build_fused_realtime_sources()` / `build_fused_realtime_source()`, and `RealtimeEveryFrameByStream` APIs were removed. Saved graphs containing `realtime_every_frame_by_stream` must be recreated with a supported policy; see [Connect live fragments](/develop-apps/development-workflow/graph/#connect-live-fragments).

### Runtime changes

- Native H.265/HEVC decode is available through `SimaDecode` and
  `RtspDecodedInput` in C++ and Python. `RtspEncodedInput` provides parsed H.265
  access units without decoding them. H.265 inputs must use HEVC Main profile,
  8-bit, 4:2:0. The codec selectors accept both `H265` and `HEVC`; H.264
  selectors also accept `AVC`. `FormatTag` / `pyneat.Format` accept the same
  aliases at encoded graph boundaries, and still serialize as `H264` and `H265`.
- `VideoSender` forwards encoded H.264 or H.265 as RTP over UDP without
  re-encoding through `VideoSenderOptions::Passthrough(codec)` /
  `pyneat.VideoSenderOptions.passthrough(codec)`. H.265 uses RTP payload type 98
  by default; H.264 keeps 96. `H264RtpUdpFromEncoded()` is deprecated in favor
  of `Passthrough(RtspCodec::H264)`.
- RTSP inputs select the RTP payload type with a single codec-neutral
  `payload_type` field on `RtspEncodedInputOptions` and
  `RtspDecodedInputOptions`: `-1` selects the codec default (96 for H.264/H.265,
  26 for MJPEG), `0` disables payload filtering, and a positive value selects an
  exact payload. `RtspEncodedInputOptions::h264_payload_type` and
  `mjpeg_payload_type` are deprecated and warn once at runtime when they change
  the resolved payload.
- Ordinary `build()` now selects fused lowering automatically for eligible live fan-in. A direct encoded H.264 or H.265 `VideoSender` branch is fused before decode without a decoded-frame CPU copy. The source, decoder, and sender must agree on codec; a mismatched pair stays in separate pipeline segments. Set that edge to `RealtimeLatestByStream` for live preview so a slow video receiver replaces stale access units instead of backpressuring the decoder branch.

- Added C++ and Python `CameraInput` documentation and tutorial coverage for MIPI/libcamera source-owned graphs, including adaptive SiMaAI memory handoff before CVU/MLA model routes.
- `MetadataSender` now keeps UDP payloads within 1200 bytes by chunking larger
  JSON messages. Update Insight to a version with metadata chunk reassembly
  before or together with this Neat Library version; older Insight versions
  continue to support unchanged JSON payloads up to 1200 bytes.

| Release | Compatible Neat SDK | Notes |
| --- | --- | --- |
| 0.3.0 | 2.1.2.3 | [Neat Library 0.3.0](https://github.com/sima-neat/core/releases/tag/v0.3.0) |
| 0.2.2 | 2.1.2.2 | [Neat Library 0.2.2](https://github.com/sima-neat/core/releases/tag/v0.2.2) |
| 0.2.1 | 2.1.2.1 | [Neat Library 0.2.1](https://github.com/sima-neat/core/releases/tag/v0.2.1) |
| 0.2.0 | 2.1.2 | [Neat Library 0.2.0](https://github.com/sima-neat/core/releases/tag/v0.2.0) |
| 0.1.0 | 2.0.0 | [Neat Library 0.1.0](https://github.com/sima-neat/core/releases/tag/v0.1.0) |
