---
title: Troubleshooting
description: Symptom-first fixes for the errors new Neat users hit most
sidebar_position: 5
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
Export `SYSROOT` and let your `CMakeLists` add it to the prefix path (the [Hello Neat template](/develop-apps/hello-neat/minimal) does this):
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
Lower the threshold only as far as you need to catch weak detections, and cap the worst case with `top_k`. See [Read Detection Boxes](/tutorials/read-detection-boxes).
:::

## Running inference

### 9. `misconfig.caps … Internal data stream error … reason not-negotiated (-4)`

:::info Cause
For raw-image input, the preprocess stage wasn't enabled / the input kind wasn't declared, so caps can't negotiate between the appsrc and the first stage.
:::

:::tip Fix
Declare image input and a preprocess preset on `ModelOptions`:
```python
opt.preprocess.kind = pyneat.InputKind.Image
opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
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
Verify the RTSP URL is reachable and actively streaming; check transport (TCP vs UDP). See [Consume an RTSP Stream](/tutorials/consume-rtsp-stream).
:::

### 12. Graph throughput is low, or live frames get dropped

:::info Cause
The graph is backpressured. Common causes are a pull loop that cannot keep up, output samples held too long, per-frame logging in the hot path, a queue policy that does not match the source, or a live stream with no explicit drop/freshness policy.
:::

:::tip Fix
Use a reusable `Run`, then make the runtime policy explicit:

- Use `RunPreset::Realtime` / `pyneat.RunPreset.Realtime` for live inputs where freshness matters.
- Use `RunPreset::Reliable` / `pyneat.RunPreset.Reliable` for batch or file processing where every input matters.
- Use `try_push(...)` when the app should not block on a full queue.
- Set `on_input_drop` to count drops by `stream_id`, `frame_id`, `port_name`, and reason.
- Pull continuously. A full output queue can throttle the whole graph.
- Release or copy outputs before pushing more if the app may hold runtime-backed buffers.

For multistream graphs, preserve `stream_id` and `frame_id` and check per-stream output counts. Aggregate FPS can hide one starving stream. See [Run a Graph → Tune throughput without lying to yourself](/develop-apps/development-workflow/pipeline#tune-throughput-without-lying-to-yourself).
:::

### 13. `unknown input/output name`, `no unambiguous default input`, or `no unambiguous default output`

:::info Cause
The graph has named endpoints and the app pushed or pulled the wrong name, or used unnamed `push(...)` / `pull(...)` on a graph with more than one possible endpoint.
:::

:::tip Fix
Inspect names before pushing or pulling:

```python
run = graph.build()
print("inputs:", run.input_names())
print("outputs:", run.output_names())
```

Then use the exact endpoint name:

```python
run.push("image", [tensor])
sample = run.pull("detections", timeout_ms=2000)
```

`Graph("name")` is a diagnostic label. It does not create an endpoint. Endpoints come from `nodes.input("name")` and `nodes.output("name")`.
:::

### 14. `pull(...)` returns no output before the timeout

:::info Cause
No sample reached the requested output before the timeout. The graph might still be running, the output name might be wrong, the input may be backpressured, the graph may be closed, or a runtime error may have occurred.
:::

:::tip Fix
Separate timeout, closed, and error. In C++, use the structured pull overload:

```cpp
simaai::neat::Sample sample;
simaai::neat::PullError error;

switch (run.pull("detections", /*timeout_ms=*/1000, sample, &error)) {
case simaai::neat::PullStatus::Ok:
  break;
case simaai::neat::PullStatus::Timeout:
  // Keep waiting, push more input, or report timeout.
  break;
case simaai::neat::PullStatus::Closed:
  // End of stream.
  break;
case simaai::neat::PullStatus::Error:
  std::cerr << error.code << ": " << error.message << "\n";
  break;
}
```

Also check `run.last_error()`, endpoint names, input dtype/layout/format, and whether your app is continuously pulling from every output branch.
:::

### 15. Old snippets fail with `push_timeout_ms`, `pull_or_throw`, root-level `input_max_*`, or `boxdecode_original_*`

:::info Cause
The snippet was written against an older option surface or a private/internal path. Current app code should use the public `ModelOptions`, `RunOptions`, and `Run` APIs.
:::

:::tip Fix
Use the current public names:

- Use `RunOptions.queue_depth`, `overflow_policy`, and `try_push(...)` for input pressure.
- Use `pull(...)` or the structured `PullStatus` overload instead of `pull_or_throw`.
- If an old snippet sets root-level `input_max_*` fields, move dynamic input limits under `ModelOptions.preprocess.input_max_width`, `input_max_height`, and `input_max_depth`, and set them only when you intentionally need bounds.
- For BoxDecode coordinate mapping, prefer preprocess metadata. Do not set deprecated original-size fields in new examples.

If the page you copied from still shows the old spelling, treat it as stale docs and file a docs bug so the next reader does not hit the same trap.
:::

## Tensors & Python interop

### 16. `… expects a TensorList; pass [tensor] instead of a single Tensor`

:::info Cause
A bare `Tensor` (or `Sample`) was passed to `run` / `push` / `build`; the API requires an explicit list — this is deliberate, not a bug.
:::

:::tip Fix
Wrap it: `model.run([tensor])`, `run.push([tensor])`, `graph.build([tensor])`.
:::

### 17. `image-mode Tensor input requires explicit image format metadata`

:::info Cause
An image-input model received a tensor with no pixel format, so Neat can't interpret the byte layout.
:::

:::tip Fix
Build the tensor with an explicit format: `pyneat.Tensor.from_numpy(arr, image_format=pyneat.PixelFormat.RGB)`.
:::

### 18. `byte_format tensors cannot also specify image_format`

:::info Cause
A tensor was constructed with both `byte_format=` (opaque bytes) and `image_format=` (pixels) — they're mutually exclusive.
:::

:::tip Fix
Pass one or the other, not both.
:::

## Coming from another stack

- **"Where's my `.engine` / `.blob` / `.dlc` / `.hef`?"** — Neat loads a `.tar.gz` model archive; that's the equivalent compiled artifact.
- **"How do I pin work to a CUDA stream / OpenCL queue?"** — you don't; decouple producer/consumer with async `push`/`pull` and tune `RunOptions` instead.
- **"Why is throughput below the headline TOPS?"** — usually host overhead, queue starvation, output backpressure, or drop policy rather than the accelerator. See [Run a Graph](/develop-apps/development-workflow/pipeline).

## When you're stuck: diagnostics

Reach for these before guessing.

**Inspect the pipeline / run (Python and C++):**
- `graph.validate()` → a `GraphReport` — validates wiring against built-in contracts before you build. Check its `error_code`.
- `graph.describe()` → the resolved pipeline as text (node names + caps chain).
- `run.input_names()` / `run.output_names()` → the names accepted by runtime push/pull calls.
- `run.start_measurement()` / `MeasureReport` → counters, latency, input-stream telemetry, plugin/edge timing, and optional power.
- `run.json(...)` / `run.save_json(...)` or C++ `save_run_json(...)` → run evidence after samples have moved.
- `NeatError::report()` → structured failure details when a run throws.


### Collect a support packet

If you need help from another developer or SiMa.ai support, send evidence another developer can replay. Include:

- Neat version/build information: Python `pyneat.build_info()` or C++ `sima_neat_version()`, `sima_neat_platform_version()`, and `sima_neat_abi_version()`;
- the model artifact name, model path, and how it was produced;
- the smallest runnable snippet that reproduces the failure;
- input shape, dtype, layout, pixel format, payload family, and whether the graph is app-pushed or source-owned;
- endpoint names from `run.input_names()` and `run.output_names()`;
- `GraphReport` JSON from `graph.validate()` or `NeatError::report()`;
- run export JSON from `run.save_json(...)` or C++ `save_run_json(...)` after samples have moved through the run;
- measurement output when the issue is latency, throughput, drops, or power.

For multistream issues, also include per-stream input counts, accepted counts, output counts, and drop counts. Aggregate FPS can hide one starving stream.

When you collect a `GraphReport`, keep the fields that explain what happened:

- `error_code` and `repro_note`;
- `pipeline_string`;
- `bus`;
- `repro_gst_launch` and `repro_env`;
- `dot_paths` and `caps_dump`;
- `boundaries` / `BoundaryFlowStats` when boundary probes are present;
- `build_adaptation` for seeded `build(input, ...)` failures;
- run export JSON for after-execution counters and metrics.

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

<!-- NOTE for maintainers: reference/error-codes.md documents io.not_found / mpk.* / plan.* / caps.* families that are NOT in include/pipeline/ErrorCodes.h. That doc is out of sync with the code — reconcile separately. This table reflects the actual constants. -->

These nine are the full set in [`ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h).
