# Changelog

All notable changes to SiMa NEAT are documented in this file.

The format is based on Keep a Changelog and follows semantic versioning.

## [Unreleased]

### Added
- Release hygiene framework for Phase 1 (governance docs, CI policy scripts, release-gate workflow).
- `TelemetryMonitor` records board power, temperatures, CPU usage, and memory from Sentinel over
  a window the caller chooses, in C++ and Python. `Telemetry::save()` writes YAML or JSON, and a
  `Telemetry` prints as YAML. `Model::benchmark()` uses it for `BenchmarkReport::telemetry`.
  Memory is reported as MLA memory and EV74 memory, which add up, and board-wide Linux memory.
  EV74 memory needs a Sentinel that reports `ev74_mem_allocated_mb`; older daemons leave it at
  `samples == 0`.
- `SimaVoxelize`, `SimaPillarScatter`, `SimaBEVPoolPlan` and `SimaBEVPool` run the kernels-library
  LiDAR and camera-to-BEV operations of PointPillars and BEVFusion on A65 or EV74, with INT8 or
  BF16 tensors, in C++ and Python (`pyneat.nodes.sima_voxelize` and so on). Each runs on the
  selected target with no fallback. See `docs/reference/lidar_bev_kernel_nodes.md`.

### Changed
- Canonical naming contract docs and migration guidance.
- Build/config naming consistency updates.
- BREAKING: The unreleased 0.6.0 line uses public C++ ABI 6 (`libsima_neat.so.6`).
  In addition to the tensor, GenAI, graph-link and benchmark layout changes,
  `SimaArgMaxOptions` gained `axis` and `keepdims`, and the `SimaArgMax` class layout
  changed. Rebuild downstream C++ consumers and Python extensions against matching
  headers and libraries, including when updating between ABI 6 previews. Do not
  redirect released ABI 5 binaries to ABI 6 with compatibility symlinks.
- `Model::benchmark()` reads power, temperature, CPU, and memory from the Sentinel daemon and
  reports them in `BenchmarkReport::telemetry`, including per-core CPU usage, with
  `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`, `peak_power_watts`, `throughput_samples`,
  and `throughput_duration_ms`. Its latency and throughput passes now each last at least
  `BenchmarkOptions::min_duration_ms` (10 s by default); set it to 0 for the previous
  count-based timing. `energy_joules` stays the energy for `num_samples` inputs. The board
  installer now installs Sentinel, or upgrades it when it is not the latest production build.
- Run power telemetry reads from Sentinel instead of the I2C power monitors, and the board
  installer no longer adds `sima` to the `i2c` group.
- BREAKING: Model archives must now use the exact lowercase `.tar.gz` suffix; `.mpk`, `.tgz`,
  `.tar`, and bare `.gz` inputs are rejected before tar inspection.
- BREAKING: The archive loader is now an internal `Model` implementation detail; public
  `mpk/*` headers and `pyneat.mpk` inspection/extraction bindings were removed.

### Fixed
- Public C++ options for Cast, CastTess, Quant, QuantTess, Tess and Preproc now
  match Core’s internal layouts, fixing standalone-node crashes. Rebuild C++
  consumers against matching headers; the unreleased 0.6.0 line remains ABI 6.

### Deprecated
- `PowerMonitor`, `read_power_snapshot()`, `RunOptions::enable_board_power()`,
  `RunOptions::power_monitor`, and `MeasureOptions::include_power`; use `TelemetryMonitor`. They
  keep working.
- `PowerMonitorOptions::sample_interval_ms`, `profile`, and `rails`, the `PowerRailConfig` I2C
  fields, the built-in power rail tables and profile helpers, `enable_modalix_som_power()`,
  `enable_modalix_dvt_power()`, and `modalix_som/dvt_power_monitor_options()`; they are ignored.
- Plugin latency: `MeasureOptions::include_plugin_latency` and `plugin_latency_source`,
  `BenchmarkOptions::include_plugin_latency`, `start_measurement(bool)`, and
  `Model::benchmark(int, bool)` / `Model::benchmark(bool)`. It will be removed.

## [0.3.0]

## [0.1.0]

### Added
- Initial public baseline of SiMa NEAT framework.
