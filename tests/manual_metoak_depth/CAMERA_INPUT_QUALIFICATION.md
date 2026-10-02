# CameraInput copy-mode qualification

This qualification is separate from the graph-20 EV74 numerical tests. It covers
`CameraInput(camera)` with `camera.zero_copy = false`, followed by
`Output`. It does not invoke an EV kernel or change the ROS application.

## Capture contract

- Legacy SIMOR: request `BA81`, wire dimensions 1920 × 360.
- Verified driver result: stride 1920, allocation size 691232; captured payload
  691200 bytes. The output is an owned flat UInt8 tensor, not decoded RGB/depth.
- Configure the media-controller links and pad formats with the board's BSP
  before capture. Discover entity names from the actual topology, not a script
  written for another carrier/CSI port. Neat does not reconfigure the media graph.
- Newer kernels advertising `MORD` are not covered by this eight-bit wire-format
  backend. Do not assume their negotiated geometry matches a legacy request.

## Lifecycle acceptance

Run only with exclusive camera ownership and a recovery contact available. Use
an in-process duration limit and an external **observational** watchdog. A watchdog
expiry must not kill the camera owner or free its buffers while DMA completion
is unknown. Save diagnostics before recovery; do not automatically retry.

1. Run the hardware-isolated capture/source tests first. These cover partial
   preparation and queueing failures, rejected output caps, failed start/stop,
   cancellation, queue-release ordering, retained tensors and blocked restart.
2. Capture for at least 60 seconds; explicitly stop and reopen.
3. Repeat 100 start/frame-delivery/stop cycles. Check device ownership and kernel
   diagnostics between cycles; stop the campaign on a lifecycle failure.
4. Run the public Neat graph for at least five continuous minutes. Record payload
   shape, negotiated caps, frame count and observed shutdown duration.
5. Retain both outputs with a two-buffer CPU pool, then stop while acquisition is
   blocked. The retained tensors must remain readable and byte-identical after
   stop and while a new capture session runs.
6. Verify that no camera descriptors or stuck tasks remain after the tests.

Shutdown first interrupts the source task, then retires V4L2 capture, unmaps and
releases the capture queue, and closes the device before retiring the CPU pool.
Output tensors retain independent CPU storage. Device-retirement failures post
an error on the GStreamer bus and prevent restarting the source. Software source
deactivation alone is not evidence of hardware recovery.

## Scope and limitations

The tested board uses Linux `6.18.3-modalix`, built September 11, 2026, with
installed kernel package `6.18.3-4837` and libcamera `2.1.3`. Its exact source
commit is not established; the newer public Metoak branch must not be assumed to
match it. Kernel, EV firmware, installed packages and ROS source remain unchanged.
Only the transient depth media-path configuration and isolated test files change.

Normal-operation qualification does not establish safe teardown after arbitrary
kernel/DMA faults, sensor unplugging, or forced process termination. Known CSI
warnings are recorded separately from DMA-stop failures. Metadata-magic checks
are not an independent depth-accuracy validation. CPU payload inspection also
means these measurements are functional evidence, not peak-throughput benchmarks.

## Recorded results — 2026-10-01

- Camera-only development run: 1334 frames over 60 seconds, zero sequence gaps,
  explicit stop/buffer release/close in 74 ms.
- 100 separate start/frame-delivery/stop cycles passed; no detected DMA-stop or
  kernel-lifecycle errors. These first two stages used an isolated capture
  qualification harness, not the public Graph API.
- Final public Python `CameraInput` → `Output` graph, default camera queue:
  8002 tensors over six minutes; all 8001 frames checked after the first frame
  passed metadata magic validation. Stop completed in 131 ms, retained Tensor
  bytes were unchanged, no camera fd remained, and reopening produced output.
- Final two-buffer output-pool test: both tensors retained across stop, byte
  integrity preserved, camera fd released, and a second graph produced output.
- Seven targeted ARM64 C++ tests, seven Python API tests, and native ASan/UBSan
  capture tests passed. An incompatible-caps negative test confirms rejection
  before QBUF/STREAMON.
- No reboot or firmware replacement was required.

The final Core shared-library SHA-256 was
`55edd137f14aaa09e0773880beb1dac17f69e53b1be35d55f3b849bce1f84d70`.
Runtime binaries were staged in RAM rather than installed over the board's
packages. Full logs, source hashes and artifact hashes are retained in the
`camera-copy/lifecycle` evidence directory for this investigation.

## Earlier unified CameraInput qualification (2026-10-02)

These trials used the earlier experimental backend-selector API and an isolated
ABI-6 test build. They predate the current profile-only API and ABI-5 packaging;
their results are historical evidence, not qualification of a later source head.
The current manifest remains ABI 5, but the changed public options/node layouts
still require rebuilding consumers and the matching Python wheel together.

Three isolated public Python Graph trials on the same B4837 board completed:

| Selection | Frames | STREAMOFF/stop observation | Retention and reopen |
| --- | ---: | ---: | --- |
| Explicit `MetoakSimor` profile and `/dev/video1`, 30 seconds | 659 | 121 ms | Passed |
| `/dev/video1` with automatic profile/backend, 30 seconds | 667 | 141 ms | Passed |
| `MetoakSimor` profile without device, both pool slots retained | 2 | 91 ms | Passed |

All 1325 frames checked after the initial frame of each trial contained the
expected SIMOR metadata magic. Output remained an owned `UInt8[691200]` tensor;
negotiated stride was 1920 and allocation size 691232. Identification returned
`/dev/video1` both with and without a requested endpoint using only media metadata.
Discovery does not open capture/subdevice nodes, allocate camera buffers or
configure media links. Multiple candidates or mixed-sensor routes are rejected.

Construction/lifecycle coverage: four guarded ARM64 C++ tests, nine Python API
tests, and the native topology fixture under ASan/UBSan passed. The non-Linux
preprocessor paths compile; this is not a native macOS runtime qualification.
The default libcamera hardware path was not rerun because this board's attached
camera is SIMOR. Existing libcamera defaults and source/factory compatibility
are covered by hardware-isolated tests. No firmware, installed runtime package,
kernel, media-controller configuration, or ROS source was changed.

These short trials qualify the unified route; they do not replace the longer
lifecycle evidence above or prove recovery from arbitrary driver/DMA faults.


## Profile-only CameraInput qualification (2026-10-02)

The manifest, generated CMake package and shared-library SONAME now use ABI 5.
Backend selection is private: `profile=Default` preserves libcamera;
`profile=MetoakSimor` selects the qualified owned-copy SIMOR route; `profile=Raw`
requires an explicit device and wire configuration. A device path alone is
rejected rather than selecting Metoak. No public `backend` option remains.
These public layout changes still require a matching rebuild of C++ consumers
and the Python wheel, despite keeping the ABI number.

The ARM64 Core library, Python extension and four focused C++ targets were
rebuilt and staged together in RAM. Four hardware-guarded C++ tests passed
(including 54 node checks); all 11 focused Python camera/API checks passed.
The topology fixture passed native ASan/UBSan. Non-Linux conditional paths
compiled; this does not establish native macOS runtime qualification. All 103
packaging/tooling regressions and formatting, include/CMake hygiene,
localization, navigation and source-policy checks passed.

| Profile-only public Graph trial | Frames | Observed stop | Retention and reopen |
| --- | ---: | ---: | --- |
| `MetoakSimor` with `/dev/video1`, 30 seconds | 654 | 131 ms | Passed |
| `MetoakSimor` without device, both output-pool slots retained | 2 | 81 ms | Passed |

All 654 post-initial frames checked across the two trials matched SIMOR metadata
magic. Output remained owned `UInt8[691200]` with `BA81`, 1920×360 wire geometry,
stride 1920 and allocation size 691232. The default libcamera route was covered
by construction/fragment tests, not a new live libcamera hardware trial. Its
strict external-buffer support guard remains intact.

Postflight found no camera/EV owners, D-state tasks or kernel lifecycle failures.
Boot identity and original firmware hash were unchanged. No firmware, kernel,
installed package, media-controller configuration or ROS source was modified.
The staged Core library SHA-256 was
`0d862b750b901bd121464db17d4f1ed676bc04cca71b286c792919e1609b068d`.
Logs and artifact hashes are retained in this investigation's `camera-profile`
evidence directory. These short route-selection trials supplement, not replace,
the earlier sustained lifecycle evidence or its kernel-fault limitations.
