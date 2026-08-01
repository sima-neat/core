# Arbitrary Video Shape Encoding: Revisited Implementation Plan

Status: correctness implementation complete; matched-platform device verification pending

Date: 2026-08-01

Scope: SiMa `neatencoder`, encoder IPC/daemon, codec integration, and Core `VideoSender` verification

Issue: [sima-neat/core#526](https://github.com/sima-neat/core/issues/526)

## Implementation status

Implemented in the companion Core/Internals worktrees on 2026-08-01:

- removed the daemon's artificial `%32`/`%8` admission gate and retained
  codec-library validation as the authority;
- restored automatic encoder-core selection;
- introduced one checked, dependency-free NV12/I420 frame-layout library shared
  by the plugin and daemon;
- replaced the daemon bulk copy with bounded, plane-aware staging into the
  hardware pitch/storage layout, including deterministic padding;
- made the plugin consume `GstVideoMeta`/`GstVideoInfo`, request video-layout
  metadata during allocation, and repack only when direct tight NV12 is not
  proven safe;
- hardened the v1 transport/error/teardown paths and checked real CMA/output
  allocation capacities;
- parameterized the Core hardware E2E and registered exact-shape tests for
  `680x382`, `672x384`, and `642x480`;
- documented the visible-geometry versus physical-layout contract.

The versioned transactional IPC described in phases 3–5 remains a follow-up;
the correctness path intentionally works with the deployed v1 ABI first. Direct
hardware-surface import is likewise an optimization behind this validated copy
contract, not a prerequisite for arbitrary codec-valid shapes.

## Current verification boundary

The implementation and package were exercised on a Modalix DevKit running
platform 2.1.2, while the built Core/Internals artifacts target platform 2.1.3.
The system package manager correctly rejected that dependency mismatch, so the
device tests used an isolated 0.4 userland over the board's older system
services. In that configuration, `680x382`, `672x384`, and `642x480` all fail at
the encoder stage, but the aligned/default raw-encoder scenarios fail there as
well. Encoded H.264/H.265 passthrough succeeds, and the two ARM frame-layout
unit binaries pass.

Those results prove that issue #526 still reproduces in the tested environment,
but they do not isolate arbitrary geometry from the broader codec/runtime
version skew. Acceptance therefore still requires a DevKit whose platform,
system services, Internals packages, and Core artifact all match the 2.1.3
dependency contract. The pull request must not claim the hardware regression is
fixed until that matched run produces RTP, exact decoded geometry, and the
quality assertions below.

## Executive decision

Support every **codec-valid visible shape**, not every integer width and height and not only shapes already aligned for hardware storage.

Keep four independent concepts:

1. visible geometry exposed in caps and decoded output;
2. the codec-coded canvas and its conformance crop;
3. the incoming buffer's plane layout;
4. the hardware surface's plane layout.

For 680x382 NV12:

| Concept | AVC/H.264 | HEVC/H.265 |
|---|---:|---:|
| Visible output | 680x382 | 680x382 |
| Codec-coded canvas | 688x384 | 680x384 |
| Hardware raster pitch | 704 bytes | 704 bytes |
| Hardware storage height | 384 | 384 |
| Tight NV12 input size | 389,640 bytes | 389,640 bytes |
| Hardware surface size | 405,504 bytes | 405,504 bytes |

The codec already expresses the coded canvas and conformance crop. Align **storage**; do not change visible caps, resize the image, or expose padded dimensions to the application.

Recommended architecture:

```text
Core VideoSender
  preserves visible caps; contains no hardware-alignment rules
        |
        v
neatencoder
  reads GstVideoMeta/GstVideoInfo
  advertises the preferred hardware layout in allocation negotiation
  imports compatible buffers or performs one conversion/repack
        |
        v
versioned codec IPC
  atomically creates a session and returns accepted layout/capacity
  registers buffers, correlates requests, tracks ownership by token
        |
        v
encoder daemon
  uses the codec library as capability authority
  directly imports compatible CMA or performs checked plane staging
        |
        v
codec/hardware
  owns codec block alignment, conformance cropping, and admission
```

This is the smallest correct model and is reusable for other raw formats, codecs, decoders, and hardware blocks.

## Baseline reviewed

The review used clean, unmodified worktrees at:

- current Core `develop`: `aa8c2ded2cbfbb7a30efb5dd13b1f4a057c76b51`;
- merged companion Core work: `ae47ddbd6ad45093ec2feb661b24c1efd60dcbff`;
- companion Internals work: `8b2c140bcec4a3f839cdf0b0fa87ce0f6afe924f`;
- local Internals `origin/develop` codec/daemon baseline: `1c1cbfc`.

Core PR [#658](https://github.com/sima-neat/core/pull/658) is merged into `develop`. It correctly makes raw `VideoSender` ingress layout-aware and preserves a conservative fallback for older plugins. It does not alter daemon shape admission, hardware staging, or fire-and-forget IPC. Issue #526 therefore still reproduces at the latest reviewed Core `develop` revision.

Observed device results with the layout-aware plugin:

| Input | Result | Interpretation |
|---|---|---|
| 640x360 | 3 decoded frames, about 43.92 dB PSNR | Aligned control works |
| 680x382 | 0 RTP frames | Rejected by daemon precheck |
| 680x384 | 0 RTP frames | Width rejected by daemon precheck |
| 688x384 | 0 RTP frames | Width rejected by daemon precheck |
| 672x384 | Frames, about 6.10 dB PSNR | Admitted, but copy/layout is corrupt |
| 704x384 | 3 decoded frames, about 44.08 dB PSNR | Fully aligned control works |

This is not one isolated `% 32` validation bug. It is a contract failure spanning capability validation, raw-plane layout, IPC, error propagation, and buffer ownership.

## What should be retained

- Core `VideoSender` specializes ingress only when the raw contract, memory domain, and plugin capability are known, and retains the legacy converter fallback.
- `neatencoder` derives input offsets and strides from `GstVideoMeta` or `GstVideoInfo`.
- `raw420_layout.{h,cpp}` uses checked arithmetic, handles padded NV12, converts I420 without source mutation, and rejects truncated or overflowing layouts.
- Layout staging is pooled rather than allocated for each frame.
- Core has a broad media matrix for formats, memory domains, graph shapes, save/load, passthrough, and native decode-to-encode.

This solves the upstream GStreamer-layout problem. It does not solve the independent hardware-surface-layout problem.

## Root cause and gaps

### 1. Artificial admission policy

`sima-ai-swsoc-video-codec/daemon/sima_enc_daemon/SimaEncoderWrapper.cpp:2021` rejects widths not divisible by 32 and heights not divisible by 8. That is daemon policy, not the AVC/HEVC 8-bit 4:2:0 shape rule.

`mlsoc/modalix/lib_common_enc/ParamConstraints.c:12` accepts even width and height for those paths. Complete acceptance also depends on codec, profile, level, frame rate, bitrate, slices, minimums/maximums, and live resources. Hard-coded divisors and static resolution lists are therefore wrong abstractions.

The daemon also overrides automatic core selection and requests both encoder cores at `SimaEncoderWrapper.cpp:1398`. The scheduler supports `NUMCORE_AUTO`; forcing two cores can reject otherwise valid smaller sessions.

### 2. Unsafe hardware staging

The daemon creates correctly pitched AL surfaces with `AL_EncGetMinPitch()` and 8-row storage-height alignment around `SimaEncoderWrapper.cpp:1523`, then defeats that metadata at `SimaEncoderWrapper.cpp:1163`: it copies `AL_Buffer_GetSize(destination)` bytes from the source as one block and ignores `inFrame.buffSize`.

For 680x382 the source supplies 389,640 tight bytes and the destination requires 405,504 accessible bytes. Relaxing the guard alone would over-read 15,864 bytes. Passing shapes can still be corrupt when pitch exceeds visible row width, as the 672x384 PSNR result demonstrates.

### 3. Initialization is not a transaction

The client sends `INIT` from its constructor and proceeds without an `INIT_RESP` or Ready state. Profile/level and other immutable configuration arrive later, so the daemon cannot validate the complete session atomically.

On failed INIT, the daemon writes a status but does not set the response command at `SimaEncoderWrapper.cpp:2146`. Zero maps to `ENCODE_DONE_RESP`; the client treats rejection as a frame completion and later reports connection reset/broken pipe.

`START` and `PUSH_BUFF` are also fire-and-forget. The plugin ignores their results, marks itself started, starts output threads, returns `GST_FLOW_OK`, and logs success after rejection.

### 4. The wire ABI cannot describe a safe layout

The protocol serializes native structs containing `void *`, `size_t`, native enums/bools, compiler padding, and a large union. It has no magic, version, encoded length, request ID, or session ID.

`frameBuffDesc_t` contains only a client pointer, physical ID, and size. It cannot describe visible geometry, format, plane offsets, strides, row bytes, rows, or storage extent. Uninitialized inactive union bytes are also sent; the union includes a 4 KiB SEI payload.

### 5. Ownership and teardown are implicit

- Encoder STOP is a no-op.
- Client RX stops before DEINIT, so it cannot reliably await release.
- GOODBYE precedes resource destruction and is not an ownership barrier.
- Pool acquisition/copy run on the socket reader and can block control processing.
- There are no negotiated credits or bounded per-session submission queues.
- A global response mutex lets a non-reading client delay other sessions.
- Completed client threads accumulate until daemon exit.
- The keepalive resends advanced configuration in a state that rejects it.

### 6. Output capacity is guessed

The plugin estimates output capacity from input pixels, while the daemon already obtains the authoritative maximum NAL size. The daemon copy path does not consistently validate client output capacity. A complete shape contract must negotiate output size as well as input layout.

### 7. Tests stop at the wrong boundary

`raw420_layout_test.cpp` tests plugin packing, including padded 642-wide input, but never creates a hardware encoder. IPC tests are primarily source-text assertions. Core's relevant encoder E2E uses fixed 640x360 geometry. Nothing crosses GStreamer layout, IPC, daemon staging, hardware encode, codec cropping, RTP, decode, and exact 680x382 output.

## Goal and invariants

Goal: any static input shape valid for its format, complete codec configuration, available hardware, and live resource state encodes without application-side resize/padding. Unsupported configurations fail synchronously and actionably.

Non-goals:

- promise odd NV12 dimensions (current 4:2:0 paths require even visible geometry);
- bypass codec/profile/level/resource limits;
- silently crop, stretch, or change image semantics;
- expose pitch or codec block alignment as caps dimensions;
- include live resolution changes in the first correctness release;
- change Core's public API.

Required invariants:

1. Decoded visible geometry equals accepted requested geometry.
2. Codec canvas alignment and conformance crop remain codec-owned.
3. Every plane access is proven within allocation bounds using checked arithmetic.
4. Bulk copy occurs only when descriptors prove compatible contiguous layouts.
5. The daemon is authoritative for complete configuration and admission.
6. Every accepted control request gets exactly one correlated response.
7. Every accepted frame returns input and output ownership exactly once.
8. Destroy responds only after callbacks quiesce, buffers return, and resources free.
9. Direct import is an optimization behind the same validated staging contract.
10. Mixed old/new daemon/plugin combinations have explicit tested behavior.

## Target design

### A. Shared dependency-free frame-layout library

Create a small internal `core/video-frame-layout` library shared by the plugin, daemon, and tests. Keep GStreamer and Allegro/Modalix dependencies in thin adapters.

```cpp
struct FrameGeometry {
  uint32_t visible_width;
  uint32_t visible_height;
  uint32_t fourcc;
};

struct PlaneLayout {
  uint64_t offset;
  int64_t stride;
  uint32_t row_bytes;
  uint32_t rows;
};

struct FrameLayout {
  FrameGeometry geometry;
  uint64_t allocation_size;
  uint32_t plane_count;
  std::array<PlaneLayout, kMaxPlanes> planes;
};
```

Pure operations:

- construct canonical NV12/I420 layouts;
- validate format-specific geometry;
- compute every plane end with checked addition/multiplication;
- validate stride, rows, offsets, allocation size, and forbidden overlap;
- classify direct, per-plane bulk, row, and conversion-plus-copy paths;
- perform a bounded copy into caller-owned storage;
- convert I420 directly into destination-pitched NV12;
- explicitly adapt legacy v1 tight NV12;
- translate `GstVideoMeta`/`GstVideoInfo` and `AL_TPixMapMetaData` in adapters.

Host inspection may represent signed strides. Direct DMA registration requires positive compatible strides; negative-stride GStreamer input uses staging. Consolidate the useful `raw420_layout.*` logic into this library rather than building a daemon copy.

### B. Codec remains the capability authority

Remove the daemon `%32/%8` gate. Validate in three stages:

1. **Structural:** fixed-width positive bounded values, supported format, chroma divisibility, safe arithmetic.
2. **Configuration:** build complete settings and call the existing `AL_Settings_CheckValidity()` and `AL_Settings_CheckCoherency()` path.
3. **Admission:** let `AL_Encoder_Create()` decide live device/core capacity.

Return distinct invalid-structure, unsupported-configuration, adjusted-configuration, and resource-unavailable statuses. Do not encode resource-dependent decisions in static caps. Retain `NUMCORE_AUTO` unless a measured explicit policy requires otherwise.

Build hardware layout from actual AL pixmap metadata and use `AL_PixMapBuffer_GetPlaneAddress()`/`AL_PixMapBuffer_GetPlanePitch()`. Preserve source/encoded visible dimensions. Do not activate source crop/resize for storage padding.

### C. Universal safe staging

For each AL destination:

1. validate source layout/allocation;
2. obtain Y/UV destination addresses and pitch;
3. copy `width` bytes for `height` Y rows;
4. copy `width` bytes for `height / 2` UV rows;
5. initialize destination padding when the pool is created;
6. submit only after validation succeeds.

Use vendor reference neutral padding (luma 0, chroma 0x80) unless hardware requirements say otherwise. Initialize once per fixed-layout pool, not across the surface every frame. Never calculate NV12 sizes with floating-point `1.5`.

### D. Shared codec IPC v2

Build a transport-independent `codec-ipc-v2` library usable by encoder and decoder. Serve v2 on a separate service-owned endpoint such as `/run/sima/codec/encoder-v2.sock`; serve v1 concurrently during migration. A new client falls back only on `ENOENT` or explicit unsupported-version response, never timeout or socket-file presence.

Fixed-width envelope:

```text
magic
protocol_major / protocol_minor
header_size
message_type / flags
payload_size
request_id
session_id
```

Rules:

- no pointer, `size_t`, native enum/bool, or implicit padding;
- exactly one correlated response per request;
- typed frame events with `frame_id` rather than generic response slots;
- append-only extension through explicit structure sizes;
- zero and validate reserved fields;
- strict packet length, truncation, enum, range, and overflow validation;
- structured status: domain, code, failing field, retryability, concise diagnostics;
- service-owned permissions and `SO_PEERCRED`.

A local ordered packet transport needs no CRC, but does need exact truncation/length handling.

### E. Atomic session creation

Replace INIT plus later advanced configuration/watermarks with one `CREATE_SESSION` carrying all immutable parameters. Return success only after validation, channel admission, encoder creation, pool creation, and layout calculation.

Response:

```text
accepted visible geometry and format
selected/adjusted codec, profile, level, and core count
preferred input plane layout
minimum pitch/alignment and storage-height alignment
exact required input bytes
maximum encoded-output size
recommended pool counts and max in-flight frames
supported staged/registered-CMA/direct-import paths
```

The plugin may reject universally impossible raw geometry and arithmetic overflow, but the daemon response remains authoritative.

### F. Registered buffers and ownership

Keep `GstBuffer *` client-local in a `buffer_token -> retained GstBuffer` registry. V2 registration describes token, memory handle/type, allocation size, coherency, format/visible geometry, and each plane's offset/stride/row bytes/rows.

Registration attaches/maps and validates once. A frame submit carries `frame_id`, input token, output token, and necessary flags/timestamps. Every accepted frame produces exactly:

- `INPUT_RELEASED(frame_id, input_token)`; and
- `OUTPUT_READY(...)` or `FRAME_FAILED(...)` for the output token.

No path may lose, double-return, or reuse a token early. Output registration uses the negotiated maximum encoded size.

### G. State, concurrency, and teardown

```text
NEW
  -> CONFIGURED             CREATE_SESSION succeeded
  -> STARTING -> RUNNING    START after hardware/pools are ready
  -> DRAINING -> CONFIGURED output/EOS and ownership returned
  -> STOPPING -> CONFIGURED cancellation/drain completed
  -> DESTROYING -> CLOSED   resources freed, callbacks quiesced

Any state -> FAILED -> DESTROYING
```

If a context cannot restart, STOP destroys and START recreates it. STOP is never a no-op.

- bounded submission queue and explicit credits per session;
- session worker so pool pressure cannot block DESTROY;
- bounded response writer per session, not a global blocking send mutex;
- bounded client request table by `request_id`;
- RX failure atomically fails the session, wakes waiters, and posts one fatal error;
- RX stays alive until `DESTROY_RESP` or defined daemon-death cleanup;
- replace the advanced-config keepalive with PING/PONG or local-socket EOF.

Controls are transactional; frames remain asynchronous and credit-controlled.

### H. GStreamer negotiation and path selection

GStreamer separates visible media geometry from memory layout. `GstVideoMeta` carries offsets/strides; allocation negotiation advertises padding, alignment, meta, and pools. Its hardware-encoder guidance recommends a compatible zero-copy pool with copy fallback: [exact layout negotiation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html#negotiating-the-exact-layout-of-video-buffers), [`GstVideoMeta`](https://gstreamer.freedesktop.org/documentation/video/gstvideometa.html), and [buffer pools](https://gstreamer.freedesktop.org/documentation/additional/design/bufferpool.html).

`neatencoder` is a custom `GstElement`, not a `GstVideoEncoder` subclass. Extend its existing sink `GST_QUERY_ALLOCATION` handler instead of combining a base-class rewrite with this fix:

- request `GST_VIDEO_META_API_TYPE`;
- advertise negotiated plane alignment/padding;
- offer a reusable hardware-shaped SiMa pool/allocator;
- default-handle query fields not owned by the plugin;
- validate each actual buffer rather than assuming negotiation was obeyed.

Use `gst_video_info_align_full()` where the platform version supports it, with a compatibility implementation for the deployed GStreamer.

Select at registration:

1. **Direct import:** compatible CMA, exact accepted layout/coherency; zero input copies.
2. **Combined staging:** system/padded/I420 input copied or converted directly into reusable hardware-shaped CMA; one input copy.
3. **Legacy staging:** explicit tight NV12 v1 adapter followed by one safe daemon row copy.

Reuse/extract the decoder daemon's external-CMA wrapper/loan pattern. The encoder retains the loan until the hardware source-release callback.

Linux V4L2 uses the same model: visible dimensions are separate from `bytesperline`/`sizeimage`, and all hardware-addressed padding must be accessible: [single-planar layout](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/pixfmt-v4l2.html), [multi-planar layout](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/pixfmt-v4l2-mplane.html), and [encoder initialization](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/dev-encoder.html#initialization).

### I. Core responsibilities

- preserve generic positive geometry checks and public APIs;
- retain capability-gated adaptive ingress and legacy fallback;
- add no `%2/%8/%32`, pitch, core-count, or codec-profile policy;
- surface structured plugin/daemon errors;
- add exact arbitrary-shape E2E coverage and documentation.

Core must not predict a resource-dependent device decision.

## Phased implementation

### Phase 0 — Freeze evidence, contracts, and budgets

- Characterize AVC/HEVC even shapes around 8/16/32/64 boundaries, profiles/levels, and representative FPS.
- Separate configuration-invalid from resource-unavailable results.
- Freeze a v1 ABI manifest.
- Specify `FrameLayout`, `SurfaceRequirements`, v2 messages, states, and ownership invariants.
- Baseline copies/frame, bytes copied, attach/map calls, CPU, CMA, latency, and throughput.

Exit: design reviewed by codec/GStreamer/runtime owners; 680x382 and 672x384 retained as red regressions; performance budgets agreed.

### Phase 1 — Harden v1 without expanding it

- Value-initialize all v1 messages/union storage and all instance fields.
- Always tag `ERR_RESP`, including failed INIT.
- Distinguish invalid configuration from channel exhaustion.
- Validate input/output capacities before copying.
- Propagate connect/send/start/push failures to GStreamer ERROR and failed flow/state.
- Implement STOP; send GOODBYE only after destruction.
- Bound sends/timeouts and reap completed client threads.

Exit: invalid 680x382 cannot masquerade as a frame; malformed input cannot access out of bounds; rejected INIT/START never reports Running; teardown is a tested barrier.

### Phase 2 — Shared layout and safe daemon staging

- Extract/generalize `raw420_layout.*`; add GStreamer and AL adapters.
- Implement checked direct/bulk/row/convert copy planning.
- Replace the daemon bulk copy with visible-plane staging.
- Initialize pool padding once.
- Remove artificial divisibility gate; use codec validators.
- Retain `NUMCORE_AUTO`.
- Correct sample comments claiming both dimensions must be multiples of 8.

Exit: 680x382 and 672x384 AVC/HEVC encode through the legacy tight adapter with exact decoded dimensions and passing PSNR; ASan/canaries are clean; aligned controls remain compatible. This phase establishes universal correctness; later phases optimize it.

### Phase 3 — Common session engine and IPC v2

- Implement v2 parser/serializer, golden tests, and fuzzing.
- Implement CREATE/START/DRAIN/STOP/DESTROY/PING.
- Return input layout, output capacity, features, pools, and credits.
- Add registration/token ownership and correlated typed events.
- Put policy/lifecycle in one session engine with v1/v2 adapters.
- Serve a new v2 endpoint while retaining v1.

Exit: state/illegal-command tests pass; delayed responses cannot acknowledge the wrong request; destroy returns after ownership; one stalled client cannot block another; old clients work with the dual-stack daemon.

Deploy daemon first, then client/plugin.

### Phase 4 — Integrate v2 into `neatencoder`

- Explicit HELLO/features with narrow v1 fallback.
- Transactional bounded caps/session setup.
- Reusable registered input/output pools and negotiated output sizing.
- Structured configuration/resource/transport/internal errors.
- Preserve `input-layout-aware` for Core compatibility.

Exit: unsupported caps fail before successful startup; daemon restart/timeout/exhaustion are distinct; failed push cannot return `GST_FLOW_OK`; mixed-version matrix passes.

### Phase 5 — One-copy and zero-copy input

- Advertise meta/pitch/padding/pool in allocation query.
- Allocate hardware-shaped CMA staging buffers.
- Combine I420/padded input into the final staging surface.
- Register/attach once, not per frame.
- Extract shared decoder CMA loan wrapper.
- Enable direct import only after layout/coherency validation.
- Track hardware source release.
- Add direct/bulk/row/conversion/rejection/copied-byte/credit counters.

Exit: compatible CMA is zero-copy; ordinary/padded/I420 is at most one copy; legacy is one bounded daemon copy; no early release; aligned throughput regression is within agreed budget (recommended initial gate 2%); non-aligned 1080p/4K-adjacent cases sustain target FPS without per-frame allocation.

Direct import remains independently disableable; staging is always the fallback.

### Phase 6 — Core E2E, docs, and rollout

- Parameterize `video_sender_adaptive_ingress_e2e_test.cpp` geometry.
- Add 680x382 plus boundary/control shapes.
- Verify RTP/decode, exact dimensions, PTS, frame count, PSNR, and source immutability.
- Document codec validity versus storage layout and error behavior.
- Add release notes and installed-package compatibility tests.
- Deploy Internals before Core tests/docs requiring the capability.

Exit: installed DevKit artifacts pass using canonical paths; required coverage does not silently skip; old Internals exercises Core fallback; close #526 with exact evidence.

### Phase 7 — Optional live geometry changes

Treat caps/resolution changes as separate. Codec support still requires layout renegotiation, pool replacement, registration turnover, and ownership barriers. Add only after static arbitrary-shape support is stable and measured.

## Required tests

### Pure layout/copy

- canonical/padded NV12 and I420-to-pitched-NV12;
- positive and negative host strides;
- bad plane count, offset, stride, rows, overlap, truncation, overflow;
- zero, odd, min/max, and exact-end geometry;
- neutral padding and path classification;
- failure leaves destination/ownership unchanged;
- source/destination canaries and ASan/UBSan.

### Geometry/device matrix

| Category | Shapes |
|---|---|
| Controls | 640x360, 1280x720, 704x384 |
| Reported regression | 680x382 |
| Corruption regression | 672x384 |
| Boundaries | 642x480, 680x384, 688x384; below/at/above width modulo 64 and height modulo 8 boundaries |
| Invalid chroma | odd width and odd height |
| Limits | codec-specific accepted min/max and one outside each |

Generate a practical residue matrix across even width modulo 64 and even height modulo 8. Run AVC and HEVC with representative profiles, levels, rates, FPS, automatic core selection, and channel exhaustion.

### Source/layout matrix

- tight/padded NV12 and I420;
- SystemMemory and SiMa CMA/segment memory;
- `GstVideoMeta` and default `GstVideoInfo` layout;
- separated planes where supported;
- native decoder and RTSP decode-to-encode;
- negotiated pool and deliberately noncompliant fallback.

### IPC/lifecycle

- golden serialization and parser fuzzing;
- correlation with delay/out-of-order/timeout;
- all state transitions, duplicates, and illegal commands;
- INIT/START rejection with no false Ready/Running;
- invalid/double/in-flight tokens and undersized buffers;
- attach/map/process failure;
- DRAIN/STOP/DESTROY/disconnect/crash/callback-close races;
- exactly-once token return;
- multi-client saturation, stalled reader, channel exhaustion, churn, bounds/backpressure.

### End-to-end assertions

Accepted cases: at least three decoded frames, exact visible dimensions, monotonic timestamps, agreed PSNR (35 dB is a conservative initial NV12 regression floor), immutable input, stable daemon/socket, and correct selected-path counter.

Rejected cases: fail before successful Running or at the exact operation, carry correct structured status, post one fatal GStreamer error, return failed flow/state, claim no output, and release all resources.

### Performance gates

Track copies/bytes per frame, attach/map churn, CPU/bandwidth, latency percentiles, FPS/backpressure, CMA high-water mark, and multi-session fairness.

Recommended gates:

- no more than 2% throughput regression on aligned controls;
- no steady-state per-frame allocation/registration;
- zero input copies for verified CMA and at most one for other v2 raw input;
- bounded memory/queues under overload;
- no cross-session stall from a non-reading client.

## Rejected alternatives

- **Pad caps and crop later:** leaks storage into semantics and duplicates codec conformance crop.
- **Relax `%32/%8` only:** exposes the 15,864-byte over-read and existing row corruption.
- **Put alignment in Core/plugin:** duplicates resource- and codec-dependent policy.
- **Keep v1 and implicitly agree on padding:** v1 cannot describe/validate layout or readiness.
- **Require zero-copy for correctness:** coherency/layout/lifetime are stricter; checked staging must remain universal.
- **Rewrite as `GstVideoEncoder` now:** current allocation-query hook suffices; a base-class rewrite is unrelated risk.
- **Static supported-resolution list:** complete configuration and live resources determine admission.
- **Use source crop/resize:** changes image semantics; coded-canvas conformance crop already exists.

## Definition of done

1. 680x382 AVC/HEVC produces RTP, decodes to exactly 680x382, emits at least three frames, and meets PSNR.
2. 672x384, 642x480, and boundary shapes are not corrupt.
3. Odd/unsupported configurations fail synchronously with the actual field/reason.
4. ASan/canaries find no over-read/overwrite.
5. Ownership is exactly once through stop/drain/destroy/disconnect/failure.
6. Core APIs, graph names, serialization, and old-plugin fallback remain compatible.
7. Aligned controls stay within performance budget.
8. Compatible CMA is zero-copy; other v2 raw input is at most one copy.
9. Installed DevKit tests use canonical paths without silent required-test skips.
10. Docs distinguish visible, coded, input-layout, and hardware-storage geometry.

## PR decomposition and merge order

1. **Internals A — layout/contracts/tests:** shared library, extracted packing, adapters, characterization and safety tests.
2. **Internals B — daemon correctness/v1 hardening:** plane staging, codec validation, auto cores, capacity/lifecycle fixes, device regressions.
3. **Internals C — codec IPC v2:** parser, session engine, state/ownership, dual endpoints, behavioral/fuzz tests.
4. **Internals D — plugin v2:** atomic setup, errors, registered pools, output sizing, compatibility.
5. **Internals E — allocation/direct import:** GStreamer advertisement, one-copy staging, shared CMA loan, zero-copy, telemetry/benchmarks.
6. **Core — E2E/docs:** parameterized shapes, diagnostics, docs, release notes.

A and B deliver correctness without waiting for zero-copy. C through E improve reuse, ABI safety, lifecycle, and performance. Land Core only after corresponding Internals artifacts reach its device lane.

## External references

- [GStreamer exact layout negotiation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html#negotiating-the-exact-layout-of-video-buffers)
- [GStreamer `GstVideoMeta`](https://gstreamer.freedesktop.org/documentation/video/gstvideometa.html)
- [GStreamer caps negotiation](https://gstreamer.freedesktop.org/documentation/additional/design/negotiation.html)
- [GStreamer buffer pool design](https://gstreamer.freedesktop.org/documentation/additional/design/bufferpool.html)
- [GStreamer `gst_video_info_align_full()`](https://gstreamer.freedesktop.org/documentation/video/video-info.html#gst_video_info_align_full)
- [Linux V4L2 dimensions, pitch, and allocation](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/pixfmt-v4l2.html)
- [Linux V4L2 multi-planar layout](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/pixfmt-v4l2-mplane.html)
- [Linux V4L2 encoder visible rectangle](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/dev-encoder.html#initialization)
- [ITU-T H.264 recommendation](https://www.itu.int/rec/t-rec-h.264)

These references reinforce the design already present in the local codec library: visible geometry and encoded semantics are distinct from pitch, offsets, padding, and allocation extent.
