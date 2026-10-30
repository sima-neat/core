---
title: Neat Library Release Notes
sidebar_position: 3
---

# Neat Library Release Notes

Release notes for the SiMa.ai Neat Library.

## 0.6.0

### Breaking changes

- Model outputs now expose the terminal logical shape and layout authored in the
  MPK, rather than an intermediate device shape. For example, the ResNet-50
  batch-flatten output has shape `[1, 1000]` and unknown logical layout, not a
  four-dimensional HWC shape. Its Float32 tensor format may be `EVXX_FLOAT32`
  instead of `FP32`. Use the tensor's dtype and shape to interpret values; do not
  infer logical axes from a backend format tag.
- The Neat Library C++ ABI is now 6 and the shared-library SONAME is `libsima_neat.so.6`. Tensors now carry feature-extractor semantic metadata, public GenAI request/result types carry ASR task, language, and probe metadata, `GraphLinkOptions` contains realtime admission limits, and `BenchmarkOptions` and `BenchmarkReport` gained fields. Rebuild C++ applications and plugins and install matching Core runtime and development packages. `SimaArgMaxOptions` gained `axis` and `keepdims`, and the `SimaArgMax` class layout also changed.
- Realtime graph composition now uses `GraphLinkOptions`, `Graph::connect()`, and `Graph::build()`. The preview `RealtimeGraphLinkOptions`, `connect_realtime()`, `build_fused_realtime_sources()` / `build_fused_realtime_source()`, and `RealtimeEveryFrameByStream` APIs were removed. Saved graphs containing `realtime_every_frame_by_stream` must be recreated with a supported policy; see [Connect live fragments](/develop-apps/development-workflow/graph/#connect-live-fragments).

- The package version is **0.6.0**. `Model::Options` and
  `BoxDecodeOptions` now carry `yolox_seg_pose.pose_classes`; these public layout
  additions require ABI 6 consumers. Do not use ABI compatibility symlinks.
  `GraphOptions` now includes `kernel_runner_a65_workers`. Rebuild applications,
  plugins and Python extensions against the matching runtime and development
  packages when updating to 0.6.0. Do not mix preview binaries with release
  packages merely because both use ABI 6.

### Runtime changes

- `Model::benchmark()` reads board telemetry from [Sentinel](/tools/sentinel/)
  instead of sampling power rails itself. The latency and throughput passes now
  each run until they have measured `num_samples` inputs and at least
  `BenchmarkOptions::min_duration_ms` has elapsed, 10 s by default, so power and
  temperature readings span several Sentinel samples. Set `min_duration_ms` to 0
  for the previous count-based timing. `BenchmarkReport` adds `latency_p50_ms`, `latency_p95_ms`,
  `latency_p99_ms`, `peak_power_watts`, `throughput_samples`,
  `throughput_duration_ms`, and `telemetry` with rail power, die and board
  temperatures, CPU utilization overall and per core, and MLA, EV74, and
  board-wide Linux memory.
  `avg_power_watts` and `energy_joules` keep their meaning; `energy_joules` is
  the energy for `num_samples` inputs. When Sentinel is unavailable, latency and
  FPS are still reported and `telemetry.status` names the reason. The DevKit
  installer installs Sentinel with `sima-cli neat install sentinel`, or upgrades
  it when the running version is not the latest production build. See
  [Measure model execution](/develop-apps/development-workflow/model#power-and-board-telemetry).
- Run power telemetry, enabled with `enable_board_power()`, also reads from
  Sentinel. Core no longer opens the I2C power monitors, so its readings no
  longer interfere with Sentinel's reads of the same rails, and the installer no
  longer adds `sima` to the `i2c` group. Sentinel samples every 2 s, so a
  measurement window shorter than about 2 s can have no power samples. Rails are
  named after Sentinel's metrics, such as `mla`, and report power only.
  `sample_interval_ms`, `PowerMonitorProfile`, the built-in rail tables, the
  `PowerRailConfig` I2C fields, `enable_modalix_som_power()`,
  `enable_modalix_dvt_power()`, and `modalix_som_power_monitor_options()` /
  `modalix_dvt_power_monitor_options()` are deprecated and ignored.
- `TelemetryMonitor` records board power, temperatures, CPU usage, and memory
  from Sentinel between `start()` and `stop()`, in C++ and Python. `read()`
  returns the summary so far, and a `Telemetry` prints as YAML and saves as
  YAML or JSON. `Model::benchmark()` uses it for `BenchmarkReport::telemetry`.
  `PowerMonitor`, `read_power_snapshot()`, `RunOptions::enable_board_power()`,
  `RunOptions::power_monitor`, and `MeasureOptions::include_power` are
  deprecated in favor of `TelemetryMonitor` and keep working. See
  [Record board telemetry](/develop-apps/development-workflow/model#record-board-telemetry).
- `BenchmarkOptions::inputs` runs `Model::benchmark()` on your own tensors
  instead of synthetic ones, and `BenchmarkReport::input` records which were
  used. See
  [Measure with your own inputs](/develop-apps/development-workflow/model#measure-with-your-own-inputs).
- Plugin latency is deprecated and will be removed. This covers
  `MeasureOptions::include_plugin_latency` and `plugin_latency_source`,
  `BenchmarkOptions::include_plugin_latency`, `start_measurement(bool)`, and
  `Model::benchmark(int, bool)` / `Model::benchmark(bool)`. Use
  `start_measurement()` and `benchmark(num_samples)` instead.
- Hardware decode uses direct codec sessions and DMA-BUF output. Core no longer
  uses decoder-daemon admission leases or `SIMA_DECODER_ADMISSION_REQUIRE`.
  Decoder pool and tuning options remain available.
- RTSP inputs support opt-in reconnect, including initially offline cameras,
  through `reconnect_timeout_ms` and `reconnect_attempts` in C++ and Python.
  Python `Run.pull_status()` exposes native status, sample and diagnostics.
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
- Raw `VideoSender` input now omits its format conversion automatically for
  compatible DMA-backed NV12 when the installed encoder advertises
  `input-layout-aware=true`. The existing encoder-input boundary prepares other
  raw inputs in final encoder storage. The `H264RtpUdpFromRaw(...)` C++ and Python
  APIs are unchanged.
- `InputMemoryPolicy::Auto` distinguishes preferred allocation from required
  input storage. Device-preferred routes retain caller-owned `Tensor` and
  `Sample` storage instead of converting it through a CPU image. Compatible
  DMA-BUF inputs stay zero-copy; existing CPU inputs can still require
  device-ingress materialization. Explicit EV74/DMS0 input requirements remain
  enforced.
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

### Graph construction and validation

- New `SimaVoxelize`, `SimaPillarScatter`, `SimaBEVPoolPlan`, and `SimaBEVPool` nodes run the
  kernels-library LiDAR and camera-to-BEV operations of PointPillars and BEVFusion
  on A65 or EV74, with INT8 or BF16 tensors. Each runs on the target you select,
  with no fallback. See
  [LiDAR and BEV kernel nodes](/reference/lidar_bev_kernel_nodes).
- A default public `Input` directly connected to one model image ingress now
  inherits its declared image format. Explicit input options remain authoritative;
  ambiguous connections require explicit options. This changes sample
  interpretation without adding a payload copy.
- Graph composition now treats one Node object as one logical vertex. Duplicate insertion and
  overlapping fragment imports fail atomically, while repeated `connect()` calls reuse an existing
  Node for fan-out.
- Every materialized pipeline segment now validates final GStreamer names during `build()`; an
  explicit `validate()` call is not required. Duplicate or dropped names fail with
  `misconfig.pipeline_shape` instead of producing a truncated pipeline.
- Custom fragments now report all explicit names and transform named-pad references together with
  declarations. Name collisions are rejected rather than auto-renamed.

| Release | Compatible Palette SDK | Notes |
| --- | --- | --- |
| 0.6.0 | 3.0.0.0 | [Neat Library 0.6.0](https://github.com/sima-neat/core/releases/tag/v0.6.0) |
| 0.5.0 | 2.1.3.1 | [Neat Library 0.5.0](https://github.com/sima-neat/core/releases/tag/v0.5.0) |
| 0.4.0 | 2.1.3.0 | [Neat Library 0.4.0](https://github.com/sima-neat/core/releases/tag/v0.4.0) |
| 0.3.0 | 2.1.2.3 | [Neat Library 0.3.0](https://github.com/sima-neat/core/releases/tag/v0.3.0) |
| 0.2.2 | 2.1.2.2 | [Neat Library 0.2.2](https://github.com/sima-neat/core/releases/tag/v0.2.2) |
| 0.2.1 | 2.1.2.1 | [Neat Library 0.2.1](https://github.com/sima-neat/core/releases/tag/v0.2.1) |
| 0.2.0 | 2.1.2 | [Neat Library 0.2.0](https://github.com/sima-neat/core/releases/tag/v0.2.0) |
| 0.1.0 | 2.0.0 | [Neat Library 0.1.0](https://github.com/sima-neat/core/releases/tag/v0.1.0) |
