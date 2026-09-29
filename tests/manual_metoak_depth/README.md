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
