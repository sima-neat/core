---
title: Memory model
description: Allocation identity, FrameLayout placement, zero-copy ownership and cache visibility.
sidebar_position: 3
slug: /develop-apps/advanced-concepts/memory_model
---

# Memory model

Core plans storage; Scheduler prepares its pools through Runner, which owns allocation identity, DMA-BUF mappings and CPU/device access.

## Allocations and views

Each stage has one immutable `FrameLayout`: a list of `FrameAllocation` objects, allocation-local `FrameSpan` values, input routes and output ownership. A span identifies an allocation, offset and reserved capacity. Logical tensor shapes, strides and published bytes remain separate; MLA hardware padding must fit the reserved capacity.

Views share a root carrier. Allocation indices belong to a buffer group: offset zero in two different groups does not mean the buffers overlap. Imported backing keeps its exporter, checked offsets, plane geometry and producer loans.

With bound storage, a dense numeric `Tensor` with shape `{}` is a rank-zero
scalar: its dense size is one `dtype` element. It differs from a zero-extent shape
such as `{0,256}`, whose dense size is zero. Tight byte copies and `clone()`
preserve the scalar value while checking the view offset and storage bounds. A
default, unbound Tensor reports zero from `dense_bytes_tight()`; tight copying
and cloning reject it. Explicit encoded, byte-stream, tessellated and text
carriers retain their carrier-byte semantics. This does not change the byte
extent returned by public `map()`.

## Placement

Core selects routes and resolves executors, compiles checked recipes, resolves carriers and ownership, then packs from the access and lifetime evidence. It finalizes stage layouts and publication once.

- Any EV74 access requires CMA.
- Owned MLA/A65-only carriers use DMS by default.
- Qualified fresh A65 public terminals may use HostCpu storage.
- Imported storage stays pinned. Fan-out includes every consumer; views retain their carrier's policy.

`AllocationPolicy::PreferDmsThenCma` explicitly permits eligible owned DMS allocations to fall back to CMA on capacity failure. It does not permit a fallback to ordinary Linux memory. Queue boundaries, separate model invocations and retained outputs keep their ownership and lifetime constraints.

For EV74 Preproc → MLA → A65 BoxDecode, the inference input uses CMA and the MLA output uses a separate DMS allocation. Actual placement follows every consumer's resolved executor.

## Ownership and CPU mapping

A zero-copy output retains its backing and producer loan while the application holds it. Scheduler and Runner keep device inputs alive through definitive completion; timeout or cancellation alone does not allow reuse.

Use [`TensorBuffer::map()`](/reference/cppapi/structs/simaai-neat-tensorbuffer) with the appropriate [`MapMode`](/reference/cppapi/namespaces/simaai-neat). A `Mapping` retains checked access and storage until it closes. Compatible reads may share access; writes remain excluded. Required cache maintenance happens before CPU reads and after writes, including preservation of neighboring cache-line bytes. Failed synchronization quarantines affected storage.

Owned cached CMA and DMS use the admitted SimaHeap range API. Imported buffers retain their exporter contract. Device-only successors do not open a CPU read epoch; CPU access is not inferred from a node name or graph position. Private CPU batches retain their parent until all child accesses retire.

Metadata envelopes may share memory without copying payloads, but dtype conversion, explicit transfers and Owned output can copy bytes. Zero-copy does not mean every stage operates in place.

## Camera buffers

[`CameraInput`](/reference/nodes/camera-input) validates captured DMA-BUF layout and retains the camera parent while downstream consumers use it. Compatible buffers keep their backing and plane geometry. When the layout needs adaptation and `allow_cpu_fallback` is enabled, the bridge makes an explicit CMA copy. Source adaptation remains separate from model preprocessing.

## Related types

- [`TensorBuffer`](/reference/cppapi/structs/simaai-neat-tensorbuffer): backing storage and views.
- [`Segment`](/reference/cppapi/structs/simaai-neat-segment): retained allocation range.
- [`Mapping`](/reference/cppapi/structs/simaai-neat-mapping): scoped CPU access.
- [`MemoryContract`](/reference/cppapi/files/include-contracts-contracttypes-h): a node's declared memory requirements.
