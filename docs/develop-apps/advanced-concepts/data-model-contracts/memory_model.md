---
title: Memory model
description: Zero-copy buffers, segments, the (buffer_id, paddr, vaddr) triple, and cache coherence.
sidebar_position: 3
slug: /develop-apps/advanced-concepts/memory_model
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

Scheduler prepares CPU/device access through Runner's shared memory authority,
which owns allocation identity, DMA-BUF retention, CPU/device exclusion, checked
ranges and generation-based visibility. MLA and EV publish completed device
writes without an eager CPU READ epoch. Required READ cache maintenance runs
when a CPU mapping or the A65 worker acquires checked access, **before reading**,
not at unmap or in backend completion callbacks. Device-only successors do no
CPU visibility work; independent frames can still overlap.

Use `Mapping` with the appropriate [`MapMode`](/reference/cppapi/namespaces/simaai-neat)
when calling [`TensorBuffer::map()`](/reference/cppapi/structs/simaai-neat-tensorbuffer).
A Tensor mapping retains its map guard and current storage/producer loan until
CPU access ends. Required END synchronization, including write cleanup, completes
before those owners release. Partial writes preserve neighboring cache-line bytes.
Failed synchronization quarantines the affected allocation instead of returning
it to normal reuse.

Each public DMA mapping retains its own Scheduler lease. Compatible reads share
Runner's access authority, including overlapping views; writes remain excluded.
Virtual addresses from separate mappings need not be identical. Releasing one
reader cannot invalidate another, and only the final owner performs the required
END transition.

Owned cached CMA requires cold platform and exporter admission; failure is
actionable, not permission to fall back to whole-buffer synchronization. Imported DMA-BUFs and other heaps keep
their exporter contract through the same ownership engine.

Private all-A65 recipes may retain an explicit parent batch; children require the
live parent capability, exact permissions and range coverage.
Final parent release cleans writes after every child has retired. Mixed device
phases do not receive implicit batch membership. No hidden staging copy is added.
Raw-FD writers and custom callbacks must still honor the ownership contract;
removing a graph allowlist does not prove that untracked access is safe.

Transport boundaries preserve backing allocation identity, checked offsets and
lengths, and original pool-buffer ownership. Metadata-only envelopes share memory
and retain the pool parent. EV/CVU and MLA imports retain the producer until
completion. Unknown completion must not recycle potentially active storage.

## Zero-copy in practice

A typical inference pipeline:

```
file → demux → H.264 decode → resize → preproc → MLA → postproc → app
```

Without zero-copy, that's seven copies. With the buffer triple and segments, it's zero — every stage hands off `(buffer_id, paddr, vaddr)` and the next stage operates in place.

The framework's planner is responsible for picking segments such that consecutive stages can share. When two adjacent stages have incompatible segment requirements, the planner inserts a `Transfer` `ConversionKind` and records it in any active `ConversionTraceCollector`. Watch for these — they're the only places real bytes move at runtime.

## Camera sources and adaptive memory

Live camera frames enter through the platform camera stack, so their memory type depends on the installed kernel, driver, and `libcamerasrc` path. `CameraInput` asks for device/SiMaAI zero-copy memory first. Its private bridge proposes a standard pool through `GST_QUERY_ALLOCATION`; the pool allocates the validated planes from one packed SiMaAI allocation and exports them as DMA-BUFs. `libcamerasrc` imports those DMA-BUFs into the ISP, and the bridge unwraps the same packed allocation for downstream CVU/MLA stages.

When the camera stack only provides OS/libcamera buffers and `allow_cpu_fallback` is enabled, Neat inserts a private camera memory bridge. The bridge copies each frame into a pooled SiMaAI buffer, stamps the expected metadata, and hands that buffer to model-managed CVU preprocessing. That copy is the compatibility bridge; resize, color conversion, normalization, quantization, and tessellation should still run on CVU/EV74.

Do not add a public `OsToSima`, `videoconvert`, or `videoscale` stage just to make MIPI camera input work. Use [`CameraInput`](/reference/nodes/camera-input) and let the source path own the memory adaptation.

## Related types

- [`TensorBuffer`](/reference/cppapi/structs/simaai-neat-tensorbuffer) — the buffer-triple container.
- [`Segment`](/reference/cppapi/structs/simaai-neat-segment) — segment handle.
- [`Mapping`](/reference/cppapi/structs/simaai-neat-mapping) — RAII map handle for direct CPU access.
- [`MemoryContract`](/reference/cppapi/files/include-contracts-contracttypes-h) — how a Node prefers to allocate.
- [`ConversionKind::Transfer`](/reference/cppapi/files/include-pipeline-tensorconversion-h) — the only conversion kind that copies across segments.

## Further reading

- "Tensors and buffers" — §0.10, §18, §19, §20 of the design deep dive.
- "TensorBuffer ABI" — §20 of the design deep dive.
