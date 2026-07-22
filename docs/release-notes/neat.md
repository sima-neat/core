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

- Ordinary `build()` now selects fused lowering automatically for eligible live fan-in. A direct encoded H.264 `VideoSender` branch is fused before decode without a decoded-frame CPU copy. Set that edge to `RealtimeLatestByStream` for live preview so a slow video receiver replaces stale access units instead of backpressuring the decoder branch.

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
