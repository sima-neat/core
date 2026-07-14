---
title: Use a MIPI Camera
description: Add an overlay-supported MIPI camera to a Neat Graph with CameraInput, model preprocessing, MLA inference, optional EV74 BoxDecode, and output pulling.
sidebar_position: 4
slug: /develop-apps/advanced-concepts/mipi-camera-input
---

# Use a MIPI Camera

Use `CameraInput` when a Graph should read frames directly from a MIPI camera on a Modalix DevKit. `CameraInput` is the Neat boundary between a validated libcamera stream and an accelerator-first model graph:

```text
CameraInput -> model-managed CVU preproc -> MLA -> Output
CameraInput -> model-managed CVU preproc -> MLA -> EV74 BoxDecode -> Output
```

No `appsrc`. No `ostosima` in user code. No CPU `videoconvert` or `videoscale` in the production path unless you add them yourself.

## Two gates: bring-up first, then Neat

MIPI CSI-2 is the camera link. It does not make every sensor plug-and-play. A working camera path also needs the right board overlay, sensor driver, libcamera pipeline, ISP behavior, and caps.

Treat MIPI camera work as two gates:

1. **Board bring-up:** the Modalix DevKit sees the sensor and libcamera can stream the requested mode.
2. **Neat graph bring-up:** `CameraInput` feeds those frames into model-managed CVU/MLA stages.

Neat starts at gate 2. It does not select `.dtbo` overlays, load sensor drivers, or tune the ISP. For overlay setup, supported overlay names, and `cam` validation, use the [Modalix DevKit MIPI camera interface guide](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces).

## What Neat supports

`CameraInput` supports cameras that are already brought up by the platform camera stack:

- the camera is connected to a Modalix DevKit MIPI port while the board is powered off;
- the correct board overlay is active;
- the kernel driver and libcamera pipeline expose the camera;
- `libcamerasrc` can negotiate the requested `video/x-raw` caps, typically `NV12`;
- the caps match the model preprocessing you configure in Neat.

If those conditions are not true yet, fix the camera stack first. A Neat graph can be precise, but it cannot turn an unbound sensor into a stream.

## Validate the camera stream

Validate the camera at the libcamera/GStreamer layer before you build the graph. If this layer does not work, the Neat graph cannot fix it.

On the DevKit, confirm that `libcamerasrc` is present:

<ShellCommand prompt="devkit">
gst-inspect-1.0 libcamerasrc
</ShellCommand>

If `cam` is available, list cameras and inspect modes:

<ShellCommand prompt="devkit">
cam -l
cam -c 1 -I
</ShellCommand>

Then try the exact caps you plan to request from Neat:

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! fakesink
</ShellCommand>

For a visual smoke test only, encode a few frames to JPEG:

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! videoconvert ! jpegenc ! \
  multifilesink location=/tmp/mipi-frame-%03d.jpg
</ShellCommand>

`videoconvert` and `jpegenc` are fine for this one-off debug check. Keep them out of the model pipeline when throughput matters.

## Build a raw MLA smoke graph

Start with a raw MLA route. It proves that camera frames reach CVU preprocessing and MLA inference before you add model-specific postprocessing.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::CameraInputOptions camera;
camera.width = 1920;
camera.height = 1080;
camera.framerate_num = 30;
camera.framerate_den = 1;
camera.format = "NV12";
camera.buffer_name = "camera0";
camera.allow_cpu_fallback = true;

neat::Model::Options model_options;
model_options.preprocess.kind = neat::InputKind::Image;
model_options.preprocess.input_max_width = static_cast<int>(camera.width);
model_options.preprocess.input_max_height = static_cast<int>(camera.height);
model_options.preprocess.input_max_depth = 3;
model_options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
model_options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
model_options.preprocess.resize.enable = neat::AutoFlag::On;
model_options.preprocess.resize.width = 640;
model_options.preprocess.resize.height = 640;
model_options.preprocess.resize.mode = neat::ResizeMode::Letterbox;
model_options.preprocess.resize.pad_value = 114;
model_options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
model_options.advanced_execution.preprocess_target = "EV74";
model_options.inference_terminal.mla_only = true;

neat::Model model("/models/yolo.tar.gz", model_options);

neat::Model::RouteOptions route;
route.include_input = false;
route.include_output = true;
route.upstream_name = camera.buffer_name;
route.buffer_name = camera.buffer_name;
route.name_suffix = "_camera0";
route.advanced_execution.preprocess_target = "EV74";

neat::Graph graph("camera_mla_smoke");
graph.add(neat::nodes::CameraInput(camera));
graph.add(model.graph(route));

neat::Run run = graph.build();
std::optional<neat::Sample> output = run.pull(/*timeout_ms=*/5000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

camera = pyneat.CameraInputOptions()
camera.width = 1920
camera.height = 1080
camera.framerate_num = 30
camera.framerate_den = 1
camera.format = "NV12"
camera.buffer_name = "camera0"
camera.allow_cpu_fallback = True

model_options = pyneat.ModelOptions()
model_options.preprocess.kind = pyneat.InputKind.Image
model_options.preprocess.input_max_width = int(camera.width)
model_options.preprocess.input_max_height = int(camera.height)
model_options.preprocess.input_max_depth = 3
model_options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
model_options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
model_options.preprocess.resize.enable = pyneat.AutoFlag.On
model_options.preprocess.resize.width = 640
model_options.preprocess.resize.height = 640
model_options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
model_options.preprocess.resize.pad_value = 114
model_options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
model_options.advanced_execution.preprocess_target = "EV74"
model_options.inference_terminal.mla_only = True

model = pyneat.Model("/models/yolo.tar.gz", model_options)

route = pyneat.ModelRouteOptions()
route.include_input = False
route.include_output = True
route.upstream_name = camera.buffer_name
route.buffer_name = camera.buffer_name
route.name_suffix = "_camera0"
route.advanced_execution.preprocess_target = "EV74"

graph = pyneat.Graph("camera_mla_smoke")
graph.add(pyneat.nodes.camera_input(camera))
graph.add(model.graph(route))

run = graph.build()
output = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

`inference_terminal.mla_only = true` is deliberate. It keeps the smoke path at `CameraInput -> CVU preproc -> MLA -> Output`, so you can debug camera and inference movement before debugging detection decode.

## Add EV74 BoxDecode when the model needs it

For YOLO-style detection models, enable BoxDecode explicitly and keep postprocessing on EV74. The decode token must match the MPK's output contract.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
model_options.inference_terminal.mla_only = false;
model_options.decode_type = neat::BoxDecodeType::YoloV9Seg;
model_options.advanced_execution.postprocess_target = "EV74";
model_options.score_threshold = 0.25f;
model_options.nms_iou_threshold = 0.45f;
model_options.top_k = 100;

route.advanced_execution.postprocess_target = "EV74";
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
model_options.inference_terminal.mla_only = False
model_options.decode_type = pyneat.BoxDecodeType.YoloV9Seg
model_options.advanced_execution.postprocess_target = "EV74"
model_options.score_threshold = 0.25
model_options.nms_iou_threshold = 0.45
model_options.top_k = 100

route.advanced_execution.postprocess_target = "EV74"
```

</CodeTab>
</CodeTabs>

Leave `decode_type` unset and keep `mla_only = true` when you want raw MLA tensors. Set `decode_type` only when the model route should emit decoded detections or segmentation data.

## Choose the right memory mode

`CameraInput` has two modes:

| Mode | Use when | Behavior |
| --- | --- | --- |
| Strict zero-copy | Your `libcamerasrc` exposes SiMaAI camera zero-copy properties and the memory library supports DMA-BUF export. | Neat requests device/SiMaAI camera buffers and fails if the source cannot provide them. |
| Adaptive fallback (default) | You want the graph to run on current camera stacks. | Neat accepts OS/libcamera buffers, copies them into pooled SiMaAI memory for CVU/MLA handoff, and passes through SiMaAI buffers when the source already provides them. |

Current Modalix DevKit images use adaptive fallback by default. Set `camera.allow_cpu_fallback = false` only after confirming both that `libcamerasrc` exposes the SiMaAI zero-copy properties and that the installed memory library supports DMA-BUF export. The fallback copy is a bridge into the accelerator pipeline; it is not permission to add CPU color conversion or scaling to the hot path.

## Keep preprocessing on CVU/EV74

For model pipelines, prefer model-managed preprocessing:

- set `Model::Options::preprocess` or `pyneat.ModelOptions.preprocess` for resize, color conversion, normalization, quantization, and tessellation;
- keep model-managed CVU pre/post targets on `EV74` when the model route supports them;
- avoid inserting `VideoConvert`, `VideoScale`, or GStreamer `videoconvert`/`videoscale` before the model unless you are building a debug-only graph.

The camera gives you frames. The CVU should do the frame math. The CPU should not become your accidental image-processing engine.

## Troubleshooting quick map

| Symptom | First check |
| --- | --- |
| No camera appears | Confirm the selected `.dtbo`, cable orientation, power cycle, and kernel/libcamera logs. |
| `libcamerasrc` is missing | Install the matching Neat/runtime camera image or camera packages for the DevKit build. |
| `misconfig.caps` or `not-negotiated` | Validate the exact `format,width,height,framerate` with `gst-launch-1.0`. Try a known supported mode such as `NV12 1920x1080@30`. |
| Strict zero-copy fails | Set `allow_cpu_fallback = true`, or use a camera stack that exposes SiMaAI zero-copy properties. |
| Output colors look wrong | Confirm the frame is interpreted as `NV12`, not RGB/BGR. If a JPEG produced directly from `libcamerasrc` is also wrong, debug camera ISP/tuning before debugging Neat. |
| Throughput is low | Remove CPU video conversion/scaling, pull continuously, and use a live-source queue policy that favors freshness. |

For more symptoms, see [Troubleshooting](/reference/troubleshooting).
