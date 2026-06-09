---
title: Glossary
description: Acronyms and terms used in the Neat framework documentation.
sidebar_position: 8
---

# Glossary

Terms and acronyms that appear across the framework's docs and code.

## Framework concepts

| Term | Definition |
|---|---|
| **Neat (the framework)** | The library this documentation describes — a C++/Python framework for building, validating, and running GStreamer-based AI inference pipelines on Modalix. |
| **Node** | The smallest building block of a pipeline. A typed wrapper over a GStreamer element. See `include/builder/Node.h`. |
| **Reusable Graph fragment** | A `Graph` that is used as a reusable composition unit, such as a model preprocess/inference/postprocess fragment or an input/output helper. |
| **Graph** | The assembly stage that turns Nodes into a runnable Run. |
| **Run** | A live, running pipeline. Push samples in, pull samples out. |
| **Model** | The simplified entry point for loading and running a compiled `.tar.gz` model archive. |
| **MPK** | The model inference contract JSON (`mpk.json` or `*_mpk.json`) embedded in a `.tar.gz` model archive. |
| **Sample** | The framework's unit of pipeline data — wraps a tensor or encoded media plus metadata. |
| **Tensor** | The framework's typed view of a buffer of pixels / audio / inference results. |
| **TensorBuffer** | The underlying memory backing a `Tensor` — carries the `(buffer_id, paddr, vaddr)` triple. |
| **Segment** | A named memory region with allocator + access metadata. Buffers come from segments. |

## Hardware

| Term | Definition |
|---|---|
| **Modalix** | SiMa's edge AI SoC platform. The framework targets it. |
| **MLA** | Machine Learning Accelerator — the SoC's main inference engine. |
| **MLASHM** | MLA Shared Memory — a low-latency memory region the MLA reads from. |
| **EV74 / CVU** | Compute Vision Unit — a SIMD-friendly DSP for preprocess / postprocess kernels. |
| **A65** | The application ARM cores running Linux. |
| **APU** | Audio Processing Unit. |
| **M4** | A Cortex-M4 used for low-level coordination (RPMsg, hardware sequencing). |
| **TVM** | Apache TVM — the framework's CPU-side fallback compiler for ops the MLA can't handle. |
| **VCCM** | A SoC-internal coherent memory region used by some accelerators. |
| **OCM** | On-chip memory used by the MLA. |
| **DMS** | Direct Memory Server — the SoC's memory controller / allocator service. |
| **RPMsg** | Remote-processor messaging — the IPC channel between A65 and M4. |
| **IOMMU** | I/O memory-management unit — maps physical to virtual for hardware. |

## Data formats

| Term | Definition |
|---|---|
| **BF16** | Brain Float 16 — IEEE 754 binary16 with full FP32 exponent range. The MLA's preferred float dtype. |
| **NV12** | YUV 4:2:0 with Y plane + interleaved UV plane. The framework's default decoded-video format. |
| **I420** | YUV 4:2:0 with separate Y, U, V planes. |
| **HWC / CHW** | Tensor layouts (Height-Width-Channels vs. Channels-Height-Width). |
| **Tessellation** | The tile-shuffle that arranges a tensor into the geometry the MLA's input scratchpad expects. Pure layout — same bytes, different order. |
| **Quantization** | Mapping FP32 values to INT8 with a scale and zero-point. |

## Operational

| Term | Definition |
|---|---|
| **Generic Preproc** | A preprocess upgrade that fuses arbitrary user-supplied transforms into the standard preprocess graph. |
| **BoxDecode** | A postprocess fusion that runs NMS / decode for detection models on the EV74. |
| **DetectionMeta** | The metadata struct attached to detection-model output samples by BoxDecode. |
| **GstSimaMeta** | The framework's GStreamer metadata struct, attached to every framework-managed buffer. |
| **Route plan** | The framework's compile-time decision about which processor runs each stage and which segments hold each buffer. |
| **Repro launch string** | The deterministic `gst-launch` text reproducer emitted by `Graph::describe()`. |

## See also

- [`Graph::describe()`](/reference/cppapi/classes/simaai-neat-graph) — emits the repro launch string.
- [The dtype contract](/develop-apps/advanced-concepts/dtype_contract).
- [Memory model](/develop-apps/advanced-concepts/memory_model).
- [Processor backends](/develop-apps/advanced-concepts/processor_backends).
