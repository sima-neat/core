---
title: Memory model
description: Zero-copy buffers, segments, coherent MLA ingress, and ownership transitions.
sidebar_position: 3
slug: /develop-apps/advanced-concepts/memory_model
---

# Memory model

The Neat framework's runtime moves a lot of bytes — encoded video frames, decoded YUV planes, FP32 input tensors, INT8 quantized tiles, MLA scratchpad images. Doing that without an explicit memory model would mean a copy at every stage boundary. This page explains how the framework avoids those copies.

## Internal buffer identity

Framework-managed buffers carry an internal identity that may include:

- **`buffer_id`** — a stable integer the runtime uses to track the buffer's lifecycle (refcount, segment ownership).
- **`paddr`** — a platform address used by older accelerator integrations.
- **`vaddr`** — virtual address, the application's view. CPU code dereferences this.

The direct MLA path does not accept a physical address supplied by an
application. ProcessMLA exports or receives a dma-buf file descriptor, and the
kernel validates the imported allocation and every bound byte range. Physical
address fields remain internal compatibility data for components that still
need them.

Stage-to-stage handoffs pass buffer ownership and metadata, not the tensor
bytes. A shared allocation can therefore be visible to both CPU and device
without making its address part of the public MLA contract.

## Segments

Buffers come from named **segments**. A segment is a contiguous region of memory backed by a specific allocator (DMA-BUF, CMA, ION, plain heap) and tagged with metadata about who can access it: CPU only, MLA only, both, etc. The runtime picks the right segment for each buffer based on which stages will touch it.

Examples:

- A `nv12_decode` segment holds decoded YUV from H.264 — CPU-readable for diagnostic taps, IOMMU-readable for the resize node.
- A direct MLA input segment holds a coherent, dma-buf-exportable tensor handed
  to ProcessMLA. CPU access uses an explicit map and must finish before device
  ownership begins.
- A `model_output` segment holds FP32 tensors after detessellation — CPU-readable so the application can pull them out.

A `Tensor` carries its segment and placement metadata, so the framework knows
whether CPU mapping and a device handoff are valid.

## Cache coherence

The MLA, EV74, and CPU have different access and cache contracts. Neat chooses
the allocation policy and performs the ownership transition when a buffer
crosses a stage boundary. A coherent mapping removes contradictory
cached/coherent aliases; it does not remove execution ordering. ProcessMLA
does not submit a buffer while application or framework CPU code owns it.

Application code expresses CPU ownership by **mapping** a `TensorBuffer` for
read or write through `Mapping`. Unmapping ends that access before the tensor
is handed to another stage. See
[`MapMode`](/reference/cppapi/namespaces/simaai-neat) and
[`TensorBuffer::map()`](/reference/cppapi/structs/simaai-neat-tensorbuffer).

## Coherent direct MLA ingress

The route determines where a public input must be allocated:

- If EV74 preprocessing is the first effective consumer, Neat uses the
  EV74-compatible input policy and hands the transformed result to MLA.
- If MLA is the first effective consumer, Neat allocates coherent,
  exportable DMS0 storage at the source. ProcessMLA imports that allocation
  directly.

Model runners and synthetic benchmark inputs resolve this policy from the
model route. They do not allocate a cached DMS0 input and repair it with a
hidden per-frame staging copy.

For explicit tensor placement, `tensor.mla(true)` creates a new coherent MLA
destination and copies the source tensor into it. The copy is the placement
operation requested by the caller; ProcessMLA does not add another copy.
Calling `tensor.mla()` without `force=true` may return the tensor unchanged
when it already satisfies the target placement.

Do not create cached and coherent aliases of the same DMS0 allocation. A
legacy cached allocation that cannot be exported safely is rejected instead
of being reinterpreted behind the application's back.

## Zero-copy in practice

A typical inference pipeline:

```
file → demux → H.264 decode → resize → preproc → MLA → postproc → app
```

Without a shared memory contract, each arrow could require a copy. With
zero-copy, compatible stages retain and hand off the same allocation rather
than copying its bytes.

The framework's planner is responsible for picking segments such that consecutive stages can share. When two adjacent stages have incompatible segment requirements, the planner inserts a `Transfer` `ConversionKind` and records it in any active `ConversionTraceCollector`. Watch for these — they're the only places real bytes move at runtime.

## Camera sources and adaptive memory

Live camera frames enter through the platform camera stack, so their memory type depends on the installed kernel, driver, and `libcamerasrc` path. `CameraInput` asks for device/SiMaAI zero-copy memory first. When the camera stack already provides it, Neat passes the buffer through and normalizes the metadata used by downstream CVU/MLA stages.

When the camera stack only provides OS/libcamera buffers and `allow_cpu_fallback` is enabled, Neat inserts a private camera memory bridge. The bridge copies each frame into a pooled SiMaAI buffer, stamps the expected metadata, and hands that buffer to model-managed CVU preprocessing. That copy is the compatibility bridge; resize, color conversion, normalization, quantization, and tessellation should still run on CVU/EV74.

Do not add a public `OsToSima`, `videoconvert`, or `videoscale` stage just to make MIPI camera input work. Use [`CameraInput`](/reference/nodes/camera-input) and let the source path own the memory adaptation.

## Related types

- [`TensorBuffer`](/reference/cppapi/structs/simaai-neat-tensorbuffer) — buffer identity, mapping, and storage.
- [`Segment`](/reference/cppapi/structs/simaai-neat-segment) — segment handle.
- [`Mapping`](/reference/cppapi/structs/simaai-neat-mapping) — RAII map handle for direct CPU access.
- [`MemoryContract`](/reference/cppapi/files/include-contracts-contracttypes-h) — how a Node prefers to allocate.
- [`ConversionKind::Transfer`](/reference/cppapi/files/include-pipeline-tensorconversion-h) — the only conversion kind that copies across segments.

## Further reading

- [Processor backends](/develop-apps/advanced-concepts/processor_backends) —
  direct ProcessMLA and workload-priority behavior.
