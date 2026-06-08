---
title: Troubleshooting
description: Symptom-first fixes for the errors new Neat users hit most
sidebar_position: 7.7
---

# Troubleshooting

Each entry is **Symptom → Cause → Fix**. The symptom headings are the exact
error strings — search this page (Ctrl-F) for the message you're seeing. Every
entry is verified against the current source or reproduced on a DevKit.

If you're not sure where to start, jump to
[When you're stuck: diagnostics](#when-youre-stuck-diagnostics).

## Install & environment

### 1. `pyneat is not importable. Either Neat is not installed, or the venv is not activated.`

:::info Cause
The `pyneat` virtual environment isn't active, or the wheel isn't installed in the environment you're running.
:::

:::tip Fix
Activate the DevKit environment before running anything Python:
```bash
source ~/pyneat/bin/activate
```
:::

### 2. GST plugin fails to load: `undefined symbol: _ZN16simaaidispatcher14DispatcherBase14submitPrepared...`

:::info Cause
The Neat runtime shared libraries aren't on the dynamic-loader path, so the GStreamer plugins can't resolve runtime symbols at load time.
:::

:::tip Fix
Put the runtime directory on `LD_LIBRARY_PATH` before launching:
```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/neat/runtime:$LD_LIBRARY_PATH
```
:::

### 3. Model archive missing — `sima-cli modelzoo` not yet run

:::info Cause
The `.tar.gz` model archive referenced by your code (or by `SIMA_YOLO_TAR` / `SIMA_RESNET50_TAR` / `SIMA_MODEL_TAR`) doesn't exist on disk.
:::

:::tip Fix
Download it from the Model Zoo:
```bash
sima-cli modelzoo get yolo_v8s     # or resnet_50, etc.
```
:::

## Build

### 4. `find_package(SimaNeat CONFIG)` cannot find the package

:::info Cause
CMake can't locate `SimaNeatConfig.cmake` (installed under `lib/cmake/SimaNeat/`). On native DevKit installs it's on the default system prefix; in SDK cross-builds the sysroot isn't on `CMAKE_PREFIX_PATH`.
:::

:::tip Fix
Export `SYSROOT` and let your `CMakeLists` add it to the prefix path (the [Hello Neat template](/getting-started/minimal_example/minimal) does this):
```cmake
if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu")
endif()
find_package(SimaNeat REQUIRED CONFIG)
```
:::

## Loading a model & configuring it

### 5. `failed to read image: <path>`

:::info Cause
OpenCV (`cv2.imread` / `cv::imread`) returned null — the file doesn't exist, isn't readable, or isn't a decodable image.
:::

:::tip Fix
Verify the path and that the file is a valid JPEG/PNG before constructing the input tensor.
:::

### 6. `reason=topk must be > 0` (from `boxdecode`)

:::info Cause
A detection model's `ModelOptions.top_k` was left at `0`; the box-decode stage requires a positive cap.
:::

:::tip Fix
Set a positive `top_k` (the tutorials use `100`):
```python
opt.top_k = 100
```
*(Message originates in the EV74 box-decode plugin.)*
:::

### 7. `preproc_upsample_not_supported`

:::info Cause
The source image is smaller than the model's input resolution, so preprocess would have to **upsample** — which older EV74 preprocess firmware does not do (it is downsample-only).
:::

:::tip Fix
Feed a source image at least as large as the model input (e.g. ≥ 640×640 for YOLOv8), or update `neat-ev74-firmware` to a build with the upsample kernel.
*(Message originates in the EV74 preprocess plugin/firmware.)*
:::

### 8. Low `score_threshold` → postprocess latency spikes

:::info Cause
The lower the detection threshold, the more candidate boxes survive thresholding, and NMS cost grows with roughly the **square** of the surviving-box count.
:::

:::tip Fix
Lower the threshold only as far as you need to catch weak detections, and cap the worst case with `top_k`. See [Read Detection Boxes](/tutorials/006-read-detection-boxes).
:::

## Running inference

### 9. `misconfig.caps … Internal data stream error … reason not-negotiated (-4)`

:::info Cause
For raw-image input, the preprocess stage wasn't enabled / the input kind wasn't declared, so caps can't negotiate between the appsrc and the first stage.
:::

:::tip Fix
Declare image input and a preprocess preset on `ModelOptions`:
```python
opt.preprocess.kind   = neat.InputKind.Image
opt.preprocess.preset = neat.NormalizePreset.COCO_YOLO
```
:::

### 10. `No channel available (all candidate channel opens failed)`

:::info Cause
The EV74 dispatcher tried to schedule a kernel the loaded firmware doesn't implement — usually because `neat-runtime` and `neat-ev74-firmware` are **not the same build** (mismatched internals hash), e.g. a partial update.
:::

:::tip Fix
Install the matched `neat-*` set (same hash) together; confirm runtime and firmware report the same hash. See [Compatibility → the version-matched set](/getting-started/compatibility#the-version-matched-set-firmware--runtime).
*(Message originates in the EV74 dispatcher.)*
:::

### 11. `frame=N rtsp_timeout`

:::info Cause
An RTSP pull timed out — the URL is wrong or the stream isn't delivering frames.
:::

:::tip Fix
Verify the RTSP URL is reachable and actively streaming; check transport (TCP vs UDP). See [Consume an RTSP Stream](/tutorials/017-consume-rtsp-stream).
:::

## Tensors & Python interop

### 12. `… expects a TensorList; pass [tensor] instead of a single Tensor`

:::info Cause
A bare `Tensor` (or `Sample`) was passed to `run` / `push` / `build`; the API requires an explicit list — this is deliberate, not a bug.
:::

:::tip Fix
Wrap it: `model.run([tensor])`, `run.push([tensor])`, `graph.build([tensor])`.
:::

### 13. `image-mode Tensor input requires explicit image format metadata`

:::info Cause
An image-input model received a tensor with no pixel format, so Neat can't interpret the byte layout.
:::

:::tip Fix
Build the tensor with an explicit format: `neat.Tensor.from_numpy(arr, image_format=neat.PixelFormat.RGB)`.
:::

### 14. `byte_format tensors cannot also specify image_format`

:::info Cause
A tensor was constructed with both `byte_format=` (opaque bytes) and `image_format=` (pixels) — they're mutually exclusive.
:::

:::tip Fix
Pass one or the other, not both.
:::

## Coming from another stack

- **"Where's my `.engine` / `.blob` / `.dlc` / `.hef`?"** — Neat loads a `.tar.gz` model archive; that's the equivalent compiled artifact.
- **"How do I pin work to a CUDA stream / OpenCL queue?"** — you don't; decouple producer/consumer with async `push`/`pull` and tune `RunOptions` instead.
- **"Why is throughput below the headline TOPS?"** — usually host overhead, queue starvation, or drop policy rather than the accelerator. See [Tune Throughput and Queue Depth](/tutorials/015-tune-throughput-and-queues).

## When you're stuck: diagnostics

Reach for these before guessing.

**Inspect the pipeline / run (Python and C++):**
- `graph.validate()` → a `GraphReport` — validates wiring against built-in contracts before you build. Check its `error_code`.
- `graph.describe()` → the resolved pipeline as text (node names + caps chain).
- `run.stats()` → a `RunStats` (headline: `avg_latency_ms`).
- `run.report()` → a formatted text report of the run.

**Turn on framework debug output** with `SIMA_DEBUG_PROFILE` — a comma-separated list of components to trace. Use `all` for everything, or narrow it:
```bash
export SIMA_DEBUG_PROFILE=all                 # everything
export SIMA_DEBUG_PROFILE=graph,gst,pipeline  # just these areas
```
Known components: `pipeline`, `graph`, `gst`, `appsink`, `inputstream`, `tensor`. Unset by default (no debug output).

**Dump the GStreamer graph** for visual inspection of where caps break:
```bash
export SIMA_GST_DOT_DIR=/tmp     # writes .dot graphs on build/failure; default: off
```

## Error codes

`NeatError` (and `GraphReport::error_code` / `PullError::code`) report a
`domain.reason` code. The framework defines exactly these — switch on the code,
read the message for specifics.

| Code | Raised when |
|---|---|
| `io.open` | A file or device path couldn't be opened — missing file, permission denied, or kernel device absent (e.g. `/dev/rpmsg*`). |
| `io.parse` | JSON/config parse error — typically a bad MPK contract or per-stage config. |
| `misconfig.pipeline_shape` | Pipeline geometry is wrong — bad sink count, a cycle, or a missing terminal `Output`. |
| `misconfig.caps` | Caps/format negotiation failed between adjacent elements (resolution, format, framerate, layout). |
| `misconfig.input_shape` | Input tensor violates the model's contract (rank, spatial dims, channel count). |
| `misconfig.runtime_abi_mismatch` | Framework/runtime plugin ABI mismatch — usually mixed `pyneat` and runtime artifacts. |
| `build.parse_launch` | `gst_parse_launch` couldn't parse the generated pipeline string (missing plugin/property). |
| `runtime.pull` | A runtime-side error during `pull` — downstream EOS, bus error, or appsink failure. |
| `infra.dispatcher_unavailable` | An MLA/EV74/A65 dispatcher couldn't be acquired — firmware not loaded, missing license, or hardware fault. No CPU fallback. |

<!-- NOTE for maintainers: concepts/error_codes.md documents io.not_found / mpk.* / plan.* / caps.* families that are NOT in include/pipeline/ErrorCodes.h. That doc is out of sync with the code — reconcile separately. This table reflects the actual constants. -->

These nine are the full set in [`ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h).
