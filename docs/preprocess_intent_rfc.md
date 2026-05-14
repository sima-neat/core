---
title: Preprocess Intent RFC
description: Intent-driven preprocess planning, strict metadata contract, and no-fallback runtime policy
sidebar_position: 30
---

# Preprocess Intent RFC

## Status

- Accepted
- Scope: `Model` / `ModelSession` managed preprocess path (C++ and Python)
- Non-goals: changing standalone `nodes::Preproc(PreprocOptions)` semantics

## Product Direction

This RFC defines an intent-driven preprocess API and planner that:

- hides kernel names from end users
- lowers intent to one frontend graph family (`Preproc`, `Quant`, `Tess`, `QuantTess`)
- validates against model-pack capabilities before runtime
- enforces strict no-fallback behavior
- propagates per-buffer geometry/context through `GstSimaMeta` for downstream stages

## Locked Decisions

1. No host fallback is allowed.
2. Preset `COCO_YOLO` is supported.
3. If both `preprocess.transforms` and simple `preprocess.*` fields are set:
   - emit a warning
   - use `transforms` as authoritative
4. If `preprocess.enable=Off` and `transforms` is non-empty:
   - emit a warning
   - apply `transforms`
5. `resolved_preprocess_plan()` exposes full input-to-MLA contract in typed form.
6. Product API is typed only:
   - C++ debug: `to_debug_string()`
   - Python debug: `to_dict()`
   - no JSON dump in core API

## Public API Contract

`Model::Options` uses `options.preprocess` as the intent namespace:

- `kind`: `Auto | Image | Tensor`
- `enable`: `Auto | On | Off`
- `resize`, `color_convert`, `layout_convert`, `normalize`
- `quantize`, `tessellate`
- `transforms` (advanced ordered chain)
- `preset`: `None | ImageNet | COCO_YOLO`

Standalone node compatibility remains:

- `nodes::Preproc(PreprocOptions)` still accepts explicit input/output dimensions.

## Planner Semantics

1. Build capabilities from MPK sequence + CVU graph templates.
2. Canonicalize user request:
   - apply preset expansion
   - compile transforms if present
3. Resolve `kind`, op enables, and graph family.
4. Validate strictly:
   - contradictory options
   - invalid normalize stats
   - unavailable graph family
   - requested op not supported by selected family
5. Lowering family rules:
   - no quant/tess: `Preproc`
   - quant only: `Quant`
   - tess only: `Tess`
   - quant+tess: `QuantTess`

## No-Fallback Policy

Runtime does **not** execute hidden host-side preprocessing or auto-degrade to another graph family.
Unsupported or invalid requests fail with actionable errors.

## Metadata ABI (GstSimaMeta)

Per-buffer preprocess context is transported via `GstSimaMeta` fields:

- geometry:
  - `preproc_original_width`, `preproc_original_height`
  - `preproc_resized_width`, `preproc_resized_height`
  - `preproc_scaled_width`, `preproc_scaled_height`
  - `preproc_pad_left`, `preproc_pad_right`, `preproc_pad_top`, `preproc_pad_bottom`
  - `preproc_resize_mode`
- conversion/context:
  - `preproc_color_in`, `preproc_color_out`
  - `preproc_layout_in`, `preproc_layout_out`
  - `preproc_normalize`, `preproc_quantize`, `preproc_tessellate`
- geometry remap:
  - `preproc_affine_m00..m12`
  - `preproc_affine_scale_x`, `preproc_affine_scale_y`
  - `preproc_affine_offset_x`, `preproc_affine_offset_y`

Downstream stages requiring preprocess context (for example box decode) must validate required fields and fail fast if fields are missing/corrupt.

## Preset Definitions

### `ImageNet`

- normalize enabled
- default mean/std:
  - mean: `[0.485, 0.456, 0.406]`
  - std: `[0.229, 0.224, 0.225]`

### `COCO_YOLO`

- resize enabled
- resize mode: `Letterbox`
- pad value: `114`
- normalize enabled
- default mean/std:
  - mean: `[0.0, 0.0, 0.0]`
  - std: `[255.0, 255.0, 255.0]` (equivalent scale `1/255`)

## Error/Warning Semantics

### Required-op disable policy

When users explicitly disable preprocess operations, the planner/runtime preserves
flexibility and only hard-fails when incompatibility is provable from framework-owned
contract data (MPK + planner + MLA contract, then runtime observed caps/meta).

1. Hard error:
   - proven contract incompatibility (`MLA contract` / `runtime observed mismatch`)
2. Warning:
   - model default / preset indicates operation is expected for accuracy,
     but compatibility is not definitively broken

Diagnostics use a normalized message payload:

`preprocess requirement violation: code=<...> op=<...> severity=<error|warning> source=<...> reason=<...> fix=<...>`

### Warnings

- transforms precedence warning
- `enable=Off` + transforms warning
- explicit disable of ops that model defaults/presets strongly expect
  (for example `normalize=Off` with `ImageNet`/`COCO_YOLO`)

### Errors

- resize enabled without valid target dimensions
- normalize enabled with non-positive stddev
- selected graph family unavailable in model pack
- selected family cannot satisfy requested ops
- required preprocess metadata missing/invalid at runtime
- required preprocess operation mismatch observed at runtime
  (for example downstream requires `preproc_quantize=true` but metadata reports `false`)

## Test Requirements

Minimum test matrix:

1. planner precedence and preset expansion
2. capability-gated family/op validation
3. strict no-fallback failures
4. metadata ABI required-field validation
5. dynamic geometry propagation and affine remap correctness in downstream consumers
6. C++/Python resolved-plan parity
