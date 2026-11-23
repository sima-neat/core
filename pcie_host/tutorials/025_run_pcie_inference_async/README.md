# 025 Run PCIe Inference Async

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, asynchronous, throughput, detection |

## Concept

Synchronous `run()` is ideal for a first inference, but it waits for each result
before submitting the next image. A producer calling `push()` while a consumer
drains `pull()` keeps one PCIe model busy. Throughput must be calculated from
completed results after warm-up—not merely from images offered to the model.

## Walkthrough

This tutorial reuses the YOLOv8s image-plus-boxdecode configuration and 640x480
street scene from tutorial 024. It submits one repeated image so storage and
image decode do not distort the PCIe measurement.

### Configure one detection model {#step-configure-model}

Load the image once, configure card-side COCO preprocessing and YOLOv8 box
decode, then build one `Model` on queue 0. Missing files and card startup errors
stop the program before measurement begins.

### Warm up the pipeline {#step-warm-up}

Run a few complete detections without timing them. Warm-up removes model startup
and first-buffer effects from the reported workload.

### Submit and retrieve concurrently {#step-push-pull}

One thread submits images with `push()` while another retrieves BBOX outputs
with a finite-timeout `pull()`. A small application-owned FIFO stores the start
time of each ordered submission. Any rejection, timeout, or malformed result
closes the model and wakes the other thread.

The example relies only on the normal `Model` flow-control behavior; there is no
queue-depth tuning in the application.

### Report completed work {#step-report-results}

Stop timing only after both threads finish and all accepted images have been
retrieved. Frames per second uses the number of completed outputs. Average
latency measures from each submission attempt until its matching ordered result
arrives.

## Run

Install the PCIe host package and download the tutorial bundle as described in
[Tutorial Setup](/tutorials/before-you-run). Run the following
commands from the extracted PCIe extras root.
On Windows, set up the tutorials and run the programs as described in
[Run applications on Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows).

The program requires the exact path `yolo_v8s_mpk.tar.gz` in this directory.
If it is not already present, download YOLOv8s from the
[Model Browser](/tools/model-browser/), extract the archive from the downloaded
ZIP file, and save it under that name:

```bash
sima-cli models download --id yolov8s --variant w8a8 --output models --force
unzip -o models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

On Windows, download the model in PowerShell and extract the archive with
`Expand-Archive`:

```powershell
sima-cli models download --id yolov8s --variant w8a8 --output models --force
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8 -Force
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

If the download is a `<id>_<variant>_mpk.tar.gz` file instead of a ZIP file, as
for a model last built before Model Browser packaged models as ZIP files, copy
that file from `models/<id>/<variant>` instead of extracting it.

Run Python:

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/025_run_pcie_inference_async/run_pcie_inference_async.py
```

Run the prebuilt C++ tutorial:

```bash
./lib/sima-pcie-host/tutorials/tutorial_025_run_pcie_inference_async
```

Or rebuild it:

```bash
./build.sh --target tutorial_025_run_pcie_inference_async
./build/tutorials-standalone/tutorial_025_run_pcie_inference_async
```

On Windows, rebuild it with CMake from an **x64 Native Tools Command Prompt**
in your copy of the tutorials. Replace `C:\opencv\build` with the `build`
directory of the OpenCV package described in
[Run applications on Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows):

```bat
call "%ProgramFiles%\SimaPCIeHost\bin\sima-pcie-host-env.cmd"
set "OpenCV_DIR=C:\opencv\build"
cmake -S share\sima-pcie-host\tutorials -B build
cmake --build build --config Release --target tutorial_025_run_pcie_inference_async
set "PATH=%OpenCV_DIR%\x64\vc16\bin;%PATH%"
build\Release\tutorial_025_run_pcie_inference_async.exe
```

The exact timing depends on the host and card, but both programs use the same
measurement boundaries and print:

```text
completed=1000
elapsed_seconds=...
throughput_fps=...
average_latency_ms=...
total_detections=...
[OK] 025_run_pcie_inference_async
```

The tutorial always warms up with five frames, then measures 1,000 completed
frames. Pass `--card N` only when using another card.

## In Practice

Keep submission and retrieval balanced. If an application pushes indefinitely
without pulling, normal backpressure eventually slows submission. A dedicated
consumer also makes failures straightforward: a finite timeout identifies a
stalled result, and closing the model releases queue 0 even if the producer is
waiting.

For a representative benchmark, replace the repeated frame with a fixed image
set and keep disk reads outside the timed region. Continue with
[Run Multiple Models](/tutorials/run-multiple-models)
to execute two different models concurrently.

## Source Files

- `run_pcie_inference_async.cpp`
- `run_pcie_inference_async.py`
- `../assets/street-scene.png`
