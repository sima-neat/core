---
title: Neat Library Release Notes
sidebar_position: 3
---

# Neat Library Release Notes

Release notes for the SiMa.ai Neat Library.

## Unreleased

- Added C++ and Python `CameraInput` documentation and tutorial coverage for MIPI/libcamera source-owned graphs, including adaptive SiMaAI memory handoff before CVU/MLA model routes.
- `MetadataSender` now keeps UDP payloads within 1200 bytes by chunking larger
  JSON messages. Update Insight to a version with metadata chunk reassembly
  before or together with this Neat Library version; older Insight versions
  continue to support unchanged JSON payloads up to 1200 bytes.

| Release | Compatible Neat SDK | Notes |
| --- | --- | --- |
| 0.2.2 | 2.1.2.2 | [Neat Library 0.2.2](https://github.com/sima-neat/core/releases/tag/v0.2.2) |
| 0.2.1 | 2.1.2.1 | [Neat Library 0.2.1](https://github.com/sima-neat/core/releases/tag/v0.2.1) |
| 0.2.0 | 2.1.2 | [Neat Library 0.2.0](https://github.com/sima-neat/core/releases/tag/v0.2.0) |
| 0.1.0 | 2.0.0 | [Neat Library 0.1.0](https://github.com/sima-neat/core/releases/tag/v0.1.0) |
