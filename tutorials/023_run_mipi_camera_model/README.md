# 023 Run a MIPI Camera Model

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | User-provided camera-compatible model |
| Labels | mipi, camera, live-input, model, ev74 |

## Concept

Attach a Modalix DevKit MIPI camera to a `Graph` with `CameraInput`, feed the live `NV12` frames into model-managed preprocessing, and pull model outputs. This is the direct camera path for deployed vision apps: the camera owns the source, CVU/EV74 handles image preprocessing, MLA runs inference, and the application consumes the result.

## Walkthrough

This chapter assumes the camera already works through the board overlay and libcamera. Neat does not select `.dtbo` files or tune the ISP; it consumes frames once `libcamerasrc` can produce them. Before running the tutorial, validate the camera with the [hardware MIPI guide](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces) and a GStreamer caps check.

Think of the tutorial as gate 2. Gate 1 is camera bring-up: overlay, driver, libcamera, ISP, and exact caps. Gate 2 is the Neat graph: camera frames into CVU preprocessing, MLA inference, optional EV74 BoxDecode, and output pulling.

### Configure the camera source {#step-configure-camera}

`CameraInputOptions` describes the source caps Neat requests from `libcamerasrc`: resolution, frame rate, format, and an optional libcamera camera name. Set `allow_cpu_fallback = true` for current camera stacks that do not expose SiMaAI zero-copy buffers yet. Strict zero-copy is still available through `--strict-zero-copy` when your `libcamerasrc` supports it.

### Configure the model route {#step-configure-model}

The model sees camera frames as `NV12` images. Configure model-managed preprocessing for color conversion, resize, normalization, quantization, and tessellation. The example pins model-managed CVU preprocessing to `EV74` so a production graph does not quietly become a CPU image pipeline. With `--decode none`, the route terminates at the MLA and returns raw model tensors. With a YOLO `--decode` token, BoxDecode runs as the model-managed EV74 postprocess stage.

### Compose the source-owned graph {#step-compose-graph}

Add `CameraInput` first, then add the model route with `include_input = false`. There is no public `Input` node because frames originate inside the running pipeline. `include_output = true` keeps a pull endpoint for detections or tensors.

### Pull outputs {#step-pull-output}

Build the graph and pull a fixed number of outputs. A timeout means no model output reached the app before `--pull-timeout-ms`; the camera may have stopped, caps may not have negotiated, or a downstream stage such as BoxDecode may be backpressured. Print tensor counts and the first tensor shape so you can confirm data is moving before adding application logic.

## Run

Run this tutorial directly on a Modalix DevKit with a configured MIPI camera. Run prebuilt commands from the Neat install root; run build-from-source commands from the repo root. The model archive must match the preprocessing and optional `--decode` mode you request.

The default pull timeout is 15 seconds. Increase `--pull-timeout-ms` when you are collecting first-run diagnostics on a cold board.

**Python:**
<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/model.tar.gz --frames 5 --decode none
</ShellCommand>

**C++ (prebuilt):**
<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /path/to/model.tar.gz --frames 5 --decode none
</ShellCommand>

For YOLO-style models with a supported BoxDecode route, choose a decode token such as `yolov8` or `yolov9seg`:

<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /path/to/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

**C++ (build from source):**
<ShellCommand prompt="devkit">
./build.sh --target tutorial_023_run_mipi_camera_model
</ShellCommand>

<ShellCommand prompt="devkit">
./build/tutorials-standalone/tutorial_023_run_mipi_camera_model \
  --model /path/to/model.tar.gz --frames 5 --decode none
</ShellCommand>

Expected output shape depends on the model and decode route. Raw MLA output usually contains model-specific tensors:

```text
frame=0 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=1 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=2 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=3 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=4 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
[OK] 023_run_mipi_camera_model
```

With a supported BoxDecode route, the output changes to decoded detection or segmentation tensors. Use the tensor count and first shape as a movement check, not as a universal contract.

If you see `output_timeout`, validate the camera with `gst-launch-1.0`, then inspect the generated backend with `--print-backend`. For BoxDecode routes, confirm the model archive, `--decode` token, and thresholds match the model.

## In Practice

Use `--print-backend` when you need to inspect the generated GStreamer path. The production path should contain `libcamerasrc`, `neatcamerabridge` when fallback is enabled, `neatprocesscvu`, `neatprocessmla`, optional EV74 postprocess, and `appsink`. It should not contain `appsrc`, `ostosima`, `videoconvert`, or `videoscale` unless you intentionally added a debug-only path.

## Source Files
- C++: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.cpp`
- Python: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py`
