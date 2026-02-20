---
title: MPK Contract
description: Model package ingestion contract, validation, and security rules
sidebar_position: 2
---

# MPK Contract

This document defines the authoritative contract for MPK/ModelPack ingestion.

## Scope

The contract applies to:

- `src/model/ModelPack.cpp`
- `src/mpk/MpKLoader.cpp`
- `src/mpk/MpKPipelineAdapter.cpp`
- `src/mpk/PipelineSequence.cpp`

## Accepted Archive Formats

Accepted package extensions:

- `.tar.gz`
- `.tgz`
- `.mpk`
- `.tar`

Archive requirements:

- Archive must be readable as a tar stream.
- Archive size must be <= loader configured `max_archive_bytes`.
- Number of entries must be <= loader configured `max_entries`.
- Entry payload size must be <= loader configured `max_entry_bytes`.

## Allowed Layout

Only regular files are accepted for extraction. Directory entries are allowed but ignored.

Allowed extracted file classes:

- JSON config files (`*.json`) -> extracted under `etc/`
- Shared objects (`*.so`) -> extracted under `lib/`
- ELF binaries (`*.elf`) -> extracted under `share/`

Required package content:

- `pipeline_sequence.json`
- At least one additional `*.json` stage config file
- At least one model binary artifact (`*.elf` or `*.so`)

## Extraction Safety Rules

Extraction must be fail-closed.

Rejected path forms:

- Absolute paths (for example `/etc/passwd`)
- Traversal segments (`..`)
- Windows drive prefixes (`C:`)
- Mixed separator traversal forms (`..\\`, `..//`)
- Invalid UTF-8 path bytes
- Unicode slash/backslash confusables (for example `U+FF0F`, `U+2215`, `U+FF3C`)
- Unicode dot confusables used in traversal-like paths (for example `U+FF0E`, `U+2024`, `U+FE52`)

Rejected entry types:

- Symlink entries
- Hardlink entries
- Device entries
- FIFO entries

Extraction behavior:

- Never write archive paths directly to filesystem output paths.
- Normalize and validate archive entry path first.
- Extract approved entries by content stream into a controlled temp root.
- Never allow writes outside extraction root.
- Duplicate normalized tar paths are rejected as `invalid_archive`.
- Tar header/checksum corruption is rejected as `invalid_archive`.

## JSON and Sequence Validation

`pipeline_sequence.json` must satisfy:

- JSON object with non-empty `pipelines` array.
- First pipeline object must contain non-empty `sequence` array.
- Each stage entry must include:
  - `sequence_id` (integer)
  - `name` (non-empty string)
  - `pluginId` (non-empty string)
  - `configPath` (non-empty string)
  - `processor` (non-empty string)
  - `kernel` (non-empty string)
- Duplicate stage names are rejected.
- Duplicate JSON keys are rejected.
- JSON nesting deeper than loader `max_json_depth` is rejected.
- Unsupported `kernel` values are rejected.
- Stage dependencies in `input` must only reference:
  - `decoder`, or
  - earlier stage names after stable ordering.

## Error Taxonomy

All MPK ingestion failures must map to one of the following classes:

- `invalid_archive`
- `path_traversal`
- `schema_error`
- `unsupported_version`
- `size_limit_exceeded`

Public error messages must include the taxonomy key so tests can assert deterministic classification.

## Determinism Requirements

- Sequence ordering must be deterministic across repeated runs.
- Fixture archives under `tests/assets/mpk` must be reproducible bit-for-bit.
- Fixture manifest checksums in `tests/assets/mpk/fixtures_manifest.json` are the source of truth.

## Test Mapping Requirement

Every negative MPK test must assert one of the taxonomy keys above.
