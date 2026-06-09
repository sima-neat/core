---
title: Memory model
description: Zero-copy buffers, segments, the (buffer_id, paddr, vaddr) triple, and cache coherence.
sidebar_position: 4
---

# Memory model

The Neat framework's runtime moves a lot of bytes — encoded video frames, decoded YUV planes, FP32 input tensors, INT8 quantized tiles, MLA scratchpad images. Doing that without an explicit memory model would mean a copy at every stage boundary. This page explains how the framework avoids those copies.

## The buffer triple: `(buffer_id, paddr, vaddr)`

Every buffer the framework moves is identified by three things:

- **`buffer_id`** — a stable integer the runtime uses to track the buffer's lifecycle (refcount, segment ownership).
- **`paddr`** — physical address, the IOMMU's view of the buffer. MLA / EV74 / DMA hardware sees this.
- **`vaddr`** — virtual address, the application's view. CPU code dereferences this.

The triple lets a buffer be addressed by either side (CPU or accelerator) without copying. A single allocation appears in *both* the kernel page tables (so software can read it) and the IOMMU page tables (so hardware can DMA into it).

Stage-to-stage hand-offs pass the triple, not the bytes.

## Segments

Buffers come from named **segments**. A segment is a contiguous region of memory backed by a specific allocator (DMA-BUF, CMA, ION, plain heap) and tagged with metadata about who can access it: CPU only, MLA only, both, etc. The runtime picks the right segment for each buffer based on which stages will touch it.

Examples:

- A `nv12_decode` segment holds decoded YUV from H.264 — CPU-readable for diagnostic taps, IOMMU-readable for the resize node.
- A `mla_input` segment holds the tessellated tensor handed to the MLA — only the MLA hardware reads it; CPU access requires an explicit map.
- A `model_output` segment holds FP32 tensors after detessellation — CPU-readable so the application can pull them out.

A `Tensor` carries its segment alongside the triple, so the framework knows whether a peek/poke from CPU code is valid.

## Cache coherence

The MLA, EV74, and CPU all have their own caches. When a buffer is written by one and read by another, the framework inserts cache flush / invalidate calls at the boundary. Application code never has to think about this — it's handled at the segment level when buffers cross stages.

The one place application code does have to think about it: when **mapping** a `TensorBuffer` for direct CPU read or write via `Mapping`. The framework inserts the right invalidate (read map) or flush (write map) at unmap time. See [`MapMode`](/reference/cppapi/namespaces/simaai-neat) and [`TensorBuffer::map()`](/reference/cppapi/structs/simaai-neat-tensorbuffer).

## Zero-copy in practice

A typical inference pipeline:

```
file → demux → H.264 decode → resize → preproc → MLA → postproc → app
```

Without zero-copy, that's seven copies. With the buffer triple and segments, it's zero — every stage hands off `(buffer_id, paddr, vaddr)` and the next stage operates in place.

The framework's planner is responsible for picking segments such that consecutive stages can share. When two adjacent stages have incompatible segment requirements, the planner inserts a `Transfer` `ConversionKind` and records it in any active `ConversionTraceCollector`. Watch for these — they're the only places real bytes move at runtime.

## Related types

- [`TensorBuffer`](/reference/cppapi/structs/simaai-neat-tensorbuffer) — the buffer-triple container.
- [`Segment`](/reference/cppapi/structs/simaai-neat-segment) — segment handle.
- [`Mapping`](/reference/cppapi/structs/simaai-neat-mapping) — RAII map handle for direct CPU access.
- [`MemoryContract`](/reference/cppapi/files/include-contracts-contracttypes-h) — how a Node prefers to allocate.
- [`ConversionKind::Transfer`](/reference/cppapi/files/include-pipeline-tensorconversion-h) — the only conversion kind that copies across segments.

## Further reading

- "Tensors and buffers" — §0.10, §18, §19, §20 of the design deep dive.
- "TensorBuffer ABI" — §20 of the design deep dive.
