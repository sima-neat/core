# Changelog

All notable changes to SiMa NEAT are documented in this file.

The format is based on Keep a Changelog and follows semantic versioning.

## [Unreleased]

### Added
- Release hygiene framework for Phase 1 (governance docs, CI policy scripts, release-gate workflow).

### Changed
- Canonical naming contract docs and migration guidance.
- Build/config naming consistency updates.
- BREAKING: Model archives must now use the exact lowercase `.tar.gz` suffix; `.mpk`, `.tgz`,
  `.tar`, and bare `.gz` inputs are rejected before tar inspection.
- BREAKING: The archive loader is now an internal `Model` implementation detail; public
  `mpk/*` headers and `pyneat.mpk` inspection/extraction bindings were removed.

## [0.3.0]

## [0.1.0]

### Added
- Initial public baseline of SiMa NEAT framework.
