# 024 Run Your First Model over PCIe

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, inference, tensor, image, detection |

## Concept

The PCIe host API accepts either a model-ready tensor or decoded image pixels.
Tensor mode keeps preprocessing on the host. Image mode sends the original
pixels and lets the Modalix card resize, convert color, and normalize them.
Adding box decode keeps the image input but replaces six raw YOLO output tensors
with one compact list of detections.

## Walkthrough

Run three independent programs with the same YOLOv8s archive and 640x480 street
scene. Each program demonstrates one mode, uses queue 0 synchronously, and
closes one model. This keeps every example short enough to copy on its own.

### Run a model-ready tensor {#step-tensor-mode}

The host letterboxes the image to the model's reported `[640, 640, 3]` input,
converts BGR to RGB, and scales pixels to `[0, 1]`. `Model.run()` sends that FP32
tensor without card-side image preprocessing and prints all six raw YOLO output
routes.

### Move preprocessing to the card {#step-image-mode}

Set `preprocess.kind` to `Image`, identify the incoming pixels as BGR, and select
the `COCO_YOLO` preset. The host now sends decoded pixels while the card performs
letterbox resize, BGR-to-RGB conversion, and normalization. The program prints
the six raw output route names and shapes so you can compare them with tensor
mode.

### Decode detections on the card {#step-decode-boxes}

Add `BoxDecodeType.YoloV8`, a score threshold, NMS threshold, and output limit.
The returned BBOX tensor begins with a detection count followed by fixed-size
records containing `(x, y, width, height, score, class_id)`. The example parses
and prints the first ten records in source-image coordinates.

### Parse the BBOX tensor {#step-parse-boxes}

Validate that box decode returned one populated tensor, read its leading count,
and reject a count that exceeds the payload. Each remaining 24-byte record is
then converted to one detection for printing.

## Run

Install the PCIe host package and download the tutorial bundle as described in
[Tutorial Setup](/tutorials/before-you-run). Run the following
commands from the extracted PCIe extras root:

```bash
sima-cli modelzoo get yolo_v8s
```

The programs require `yolo_v8s_mpk.tar.gz` in this directory. Model Zoo output
names and locations can vary. If the command did not create that exact path,
copy the downloaded archive into place and verify it:

```bash
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_tensor_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_boxdecode.py
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_024_run_tensor_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_boxdecode
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_024_run_tensor_mode
./build.sh --target tutorial_024_run_image_mode
./build.sh --target tutorial_024_run_image_boxdecode

./build/tutorials-standalone/tutorial_024_run_tensor_mode
./build/tutorials-standalone/tutorial_024_run_image_mode
./build/tutorials-standalone/tutorial_024_run_image_boxdecode
```

The matching C++ and Python programs print the same six raw output contracts
for tensor and image mode, followed by decoded people, cars, or other visible
objects:

```text
Tensor mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_tensor_mode
Image mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_image_mode
Image + boxdecode detections=...
  person score=... box=(...)
[OK] 024_run_image_boxdecode
```

The default is card 0 and queue 0. Pass `--card N` only when using another card;
its management address is derived automatically.

## In Practice

Use tensor mode when your application already produces exactly the dtype,
shape, layout, color order, and numeric range reported by `model.info()`. Use
image mode when the application naturally owns decoded pixels and you want the
card to apply repeatable model preprocessing. Enable box decode when the
application needs detections rather than raw feature maps.

Every mode uses the same `pcie::Model`/`pyneatpcie.Model` lifecycle. Only
`ModelOptions` and the submitted payload change. Continue with
[Run PCIe Inference Async](/tutorials/run-pcie-inference-async)
to overlap submission and completion with `push()` and `pull()`.

## Source Files

- `run_tensor_mode.cpp`
- `run_tensor_mode.py`
- `run_image_mode.cpp`
- `run_image_mode.py`
- `run_image_boxdecode.cpp`
- `run_image_boxdecode.py`
- `../assets/street-scene.png`
