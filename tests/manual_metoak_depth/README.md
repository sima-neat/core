# Metoak depth qualification

This explicit hardware probe checks `Input` → `MetoakDepth` → `Output`, not the
robot stack. Use an exclusively owned Modalix DevKit, the matching graph-20 firmware
and host package set, and a reviewed firmware restoration/recovery procedure.
Never run target ARM64 binaries on an x86 build host.

`MetoakDepthOptions` accepts even width 8–2048 and height 8–1536, batch one.
The camera's native shape is 640×360; publication resizing is separate.
The six canonical inputs are Y/U/V UInt8 planes, UInt16 disparity, FP32 BF `[1]`
and FP32 projection `[3]` (`fx_fy`, `cx`, `cy`). Keep canonical names. Disparity
scale is fixed at 32. All three outputs are published in order: RGB UInt8 HWC,
depth UInt16 HW in millimeters, XYZ FP32 HWC in meters. Depth is the primary
boundary description, not the only output. Invalid disparity produces depth zero
and XYZ NaNs. BF and focal length must be positive and finite; principal point
must be finite.

Build this directory as a standalone CMake project against the same installed
Core development package used by the application (cross-compilation requires the
SDK toolchain). This probe is not registered in automatic CTest runs.

Generate at least two same-geometry fixtures with the independent Internals
`test/simor_depth_reference.py`, changing seed and calibration values. Pass their
directories to the probe on the guarded native DevKit:

```bash
./metoak_depth_qualification --guarded-hardware /path/to/frame1 /path/to/frame2
```

The probe requires exact RGB/depth bytes, matching XYZ invalid masks, and finite
XYZ error within `1e-5 + 1e-5 * abs(reference)`. It holds earlier outputs while
processing later frames and rechecks them to catch buffer reuse/lifetime errors.
Use additional fixture batches for minimum geometry, non-vector-width tails,
zero/high disparities, saturation and repeated changing calibration. Geometry
changes need separate probe runs.

On an execution exception the probe prints **PARKED** and retains its inputs,
outputs and Run state. Do not blindly kill/retry it: first use the guarded recovery
procedure to establish DMA-safe quiescence. Successful compilation, component
discovery and `import pyneat` do not replace this numerical qualification.

Every probe input is a device-backed subview with a nonzero source byte offset.
Include a 10×14 fixture pair: its odd-sized chroma planes require padding between
named segments so UInt16 disparity and FP32 calibration remain aligned.

Before dispatch qualification, build the existing
`unit_sample_packed_parent_alignment_padding_test` target and run its explicit
memory-only gate on the native target with the matching candidate libraries:

```bash
file ./unit_sample_packed_parent_alignment_padding_test
./unit_sample_packed_parent_alignment_padding_test --device-bundle
```

This gate does not dispatch firmware. It requires real device allocations and
checks six mixed-type inputs, nonzero offsets, exact copied bytes, bus alignment,
logical metadata, injected source-map failures, and subsequent recovery. Missing
device support is a failure, not a skipped test. Destination cleanup on failures
is protected by RAII; this check does not independently count allocator leaks.

The probe blocks SIGINT, SIGTERM, and SIGHUP before creating pipeline threads.
A shell timeout or disconnect cannot safely be treated as cancellation. The
external guardian must retain the device locks until hardware quiescence is
established, then use SIGKILL if a parked probe must be terminated. Fatal signals
and internal runtime cleanup before an exception are not made safe by this
caller-side guard; treat any abnormal exit as an ambiguous DMA outcome.

## Qualified cohort — 2026-09-30

The following exact cohort passed fresh source/package builds, offline dependency
audits and guarded synthetic-tensor qualification on Modalix **2.1.3 B4837**:

| Component | Product source commit |
|---|---|
| Internals host and fresh EV74 firmware/runtime | `9b86e8c500867cb0d753e7694418c93a994c735a` |
| Core and Python wheel | `5b5152a0779c77abff8696a8ffbb6ae5199ebde7` |
| LLiMa | `de7937e79defaf4e769c1ae2c61ecc8996704955` |

- Raw firmware SHA256: `3a403a400765e6c28f1446db60323b577d299013e8101f6575a4631953154011`.
- Complete `final-cohort-v3.tar.gz` SHA256: `53190bf92c1f5cc1ff6bae9fcb58c4fdfaea6f210da7a6a1d94b192c20c40f3e`.

This is the legacy RPMsg profile with 32-bit libmetal physical-address/page-mask
ABI, **not** the 3.0 `/dev/cvu` profile. Internals
`0d089531ca8f5f6dad2fcb2a968d4efd0201810b` is a subsequent tests-only correction;
it is not the firmware or package source identity above.

The guarded trial recorded 15 explicit no-pending-DMA completion markers across
baseline/restored casts, candidate productized regressions, rejected-request
recovery, standalone graph20, production Core and the actual camera-wrapper
processing source. Core passed A→B→A cases at 640×360, 10×14 and 8×8; the wrapper
passed 640×360 and 10×14, including calibration-not-ready handling. Original
firmware was restored and functionally retested; M4 was unchanged. These results
do not qualify other firmware hashes, sustained workloads or live camera input.

### Required isolation for this qualification

Default system-plugin discovery blocked in a `/dev/media0`
`MEDIA_IOC_G_TOPOLOGY` ioctl. The unit-test parent was proven to be in plugin
initialization before dispatch and was canceled; the blocked scanner was left
untouched. This platform media-enumeration issue remains unresolved.

The passing synthetic tests used the exact candidate Neat plugin directory and a
separate, root-owned, hash-verified standard-plugin directory containing only
`libgstcoreelements.so` and `libgstapp.so`. Both `GST_PLUGIN_SYSTEM_PATH` and
`GST_PLUGIN_SYSTEM_PATH_1_0` pointed exclusively to that directory. The guardian
kept `SIMA_GST_NEAT_ONLY=1`, set `SIMA_GST_PLUGIN_DIR` to the candidate plugins,
and used a fresh registry. `SIMA_GST_ALLOW_SYSTEM_PLUGINS=1` preserved the explicit
curated directory instead of rebuilding the broader default filtered directory;
it did not enable unrestricted system discovery.

The default Neat plugin filter prevents namespace conflicts, not all device
probing. Do not treat the curated profile as a general camera configuration or
hide missing factories by reopening the full system plugin path. Reproduce it
only through the reviewed exclusive guardian, with pinned plugin hashes and the
existing DMA ownership/restoration safeguards.

**Scope:** synthetic C++ Metoak processing and isolated Python import/ABI checks.
Metoak has no Python binding in this cohort. These results do not establish default
GStreamer startup, camera acquisition, ROS/robot deployment or installation
readiness. Consult the matching acceptance receipt for subsequent ROS and
publication status.
