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
  reports them in `BenchmarkReport::telemetry`, with `peak_power_watts`, `throughput_samples`, and
  `throughput_duration_ms`. Its throughput pass now lasts at least
  `BenchmarkOptions::min_duration_ms` (30 s by default); set it to 0 for the previous
  count-based timing. `energy_joules` stays the energy for `num_samples` inputs. The board
  installer now installs Sentinel when it is not running.
- BREAKING: Model archives must now use the exact lowercase `.tar.gz` suffix; `.mpk`, `.tgz`,
  `.tar`, and bare `.gz` inputs are rejected before tar inspection.
- BREAKING: The archive loader is now an internal `Model` implementation detail; public
  `mpk/*` headers and `pyneat.mpk` inspection/extraction bindings were removed.

## [0.3.0]

## [0.1.0]

### Added
- Initial public baseline of SiMa NEAT framework.
