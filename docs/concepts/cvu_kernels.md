---
title: CVU kernels and graphs catalog
description: What the framework's CVU-side kernels do, and how preprocess / postprocess graphs are composed from them.
sidebar_position: 8
---

# CVU kernels and graphs catalog

The framework ships a small catalog of CVU (EV74) kernels that combine into the preprocess and postprocess graphs the planner picks for each model. This page describes the kernels and the graph families that compose them.

## Kernel families

### Preprocess kernels

Run on the EV74 between input and the MLA. The standard family:

- **Resize** — bilinear / nearest scaling, with optional letterbox or center-crop.
- **Color convert** — RGB ↔ BGR, NV12 → RGB, I420 → RGB, GRAY8 ↔ packed.
- **Layout convert** — HWC ↔ CHW, axis permutation.
- **Normalize** — per-channel mean/stddev (FP32 in, FP32 out).

### Boundary kernels

Bridge FP32 / BF16 / INT8 across the MLA boundary:

- **Quant** — FP32 → INT8 with scale + zero-point.
- **Dequant** — INT8 → FP32 with scale + zero-point.
- **Cast** — FP32 ↔ BF16 (no scale / zero-point).
- **Tess** / **Detess** — layout shuffle into / out of MLA tile geometry. Same bytes, different order.

### Fused kernels

Combinations the planner picks when a model's contract demands the boundary kernels but doesn't include them in the MLA stage:

- **QuantTess** — fuse Quant + Tess.
- **DetessDequant** — fuse Detess + Dequant.
- **CastTess** / **DetessCast** — fuse Cast with Tess on the BF16 path.

### Generic Preproc

When the application supplies arbitrary user-defined transforms, the planner upgrades the preprocess graph to a generic variant that fuses those transforms into a single CVU kernel. The contract at the MLA boundary doesn't change.

### BoxDecode

A postprocess kernel that fuses NMS / decode for detection models. Produces `DetectionMeta` on the output sample. See [`BoxDecodeType.h`](/reference/cppapi/files/include-pipeline-boxdecodetype-h).

## How graphs compose

The four `PreprocessGraphFamily` values map to four kernel chains:

| Graph family | Chain (input → MLA) |
|--------------|---------------------|
| `Preproc` | Resize → ColorConvert → Normalize → MLA (which tessellates internally) |
| `Quant` | Resize → ColorConvert → Normalize → Quant → MLA (which tessellates internally) |
| `Tess` | Resize → ColorConvert → Normalize → Tess → MLA |
| `QuantTess` | Resize → ColorConvert → Normalize → QuantTess → MLA |

The dual on the output side — `Postproc` / `Detess` / `DetessDequant` / pass-through — depends on whether the MLA's compiled output kernel includes detess/dequant.

See [the dtype contract](dtype_contract) for why these four families exist.

## Kernel naming convention

Inside the framework, kernels are referenced by stable string names that show up in `RoutePlanner` decisions and `RunDiagSnapshot` reports:

- `cvu/preproc/<variant>` — preprocess kernels.
- `cvu/quant/<dtype>` — quant variants.
- `cvu/tess/<geometry>` — tess variants.
- `cvu/postproc/box_decode_<type>` — BoxDecode variants.

The exact catalog is in `core/src/pipeline/internal/sima/` (the framework's reach-through layer).

## Further reading

- "CVU kernels and graphs catalog" — §86, §87 of the design deep dive.
- "Tessellation, quant, cast" — §17 of the design deep dive.
- [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h) — the four-corner enum.
- [`BoxDecodeType.h`](/reference/cppapi/files/include-pipeline-boxdecodetype-h) — postprocess box decode.
