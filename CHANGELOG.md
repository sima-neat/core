# Changelog

All notable changes to SiMa NEAT are documented in this file.

The format is based on Keep a Changelog and follows semantic versioning.

## [Unreleased]

### Added
- Release hygiene framework for Phase 1 (governance docs, CI policy scripts, release-gate workflow).

### Changed
- Canonical naming contract docs and migration guidance.
- Build/config naming consistency updates.
- BREAKING: The public C++ ABI is now 6 (`libsima_neat.so.6`) because tensors now carry
  feature-extractor semantic metadata, the existing GenAI and graph-link public structures
  changed, and `BenchmarkOptions` and `BenchmarkReport` gained fields. Rebuild downstream C++
  consumers.
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

### Deprecated
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
