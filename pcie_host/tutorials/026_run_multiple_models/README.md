# 026 Run Multiple Models

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | resnet_50, yolo_v8s |
| Labels | PCIe, queues, concurrency, classification, detection |

## Concept

A Modalix PCIe Card exposes co-processing queues 0 through 5. Each active `Model` owns one
queue, so independent models can execute concurrently by assigning a different
`ConnectionOptions.queue` to each instance. This tutorial uses ResNet-50 on
queue 0 and YOLOv8s on queue 1 without adding a global coordinator.

## Walkthrough

The two models intentionally use different images: ResNet-50 classifies a clear
Labrador photograph, while YOLOv8s detects people and cars in a busy street
scene.

### Load model-specific images {#step-load-assets}

Validate both model archives and decode both packaged assets before occupying a
queue. Keeping the images separate makes each result meaningful and avoids
using a classification portrait as an object-detection workload.

### Assign one model to each queue {#step-assign-queues}

Create two ordinary `Model` objects. Configure ResNet-50 with ImageNet image
preprocessing on queue 0 and YOLOv8s with COCO image preprocessing plus box
decode on queue 1. Build errors identify the queue and model that failed; an
already-built model is closed if the second build fails.

### Run both queues concurrently {#step-run-concurrently}

Start one blocking image inference per model in separate host threads. Each
call still uses the simple synchronous `run` behavior, but the calls overlap
because they target different physical queues.

### Interpret each result independently {#step-read-results}

Queue 0 returns one FP32 classification tensor and prints its top-scoring
ImageNet class. Queue 1 returns decoded BBOX records and prints detection class,
confidence, and source-image coordinates. Closing either model releases only
its assigned queue.

## Run

Install the PCIe host package and download the tutorial bundle as described in
[Tutorial Setup](/tutorials/before-you-run). Run the following
commands from the extracted PCIe extras root.
On Windows, set up the tutorials and run the programs as described in
[Run applications on Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows).

The program requires the exact paths `resnet_50_mpk.tar.gz` and
`yolo_v8s_mpk.tar.gz` in this directory. Download both models from the
[Model Browser](/tools/model-browser/), extract the archives from the downloaded
ZIP files, and save them under those names:

```bash
sima-cli models download --id resnet_50 --variant w8a8 --output models
sima-cli models download --id yolov8s --variant w8a8 --output models
unzip models/resnet_50/w8a8/*.zip resnet_50_w8a8_mpk.tar.gz
unzip models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
mv resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

On Windows, download the models in PowerShell and extract the archives with
`Expand-Archive`:

```powershell
sima-cli models download --id resnet_50 --variant w8a8 --output models
sima-cli models download --id yolov8s --variant w8a8 --output models
Expand-Archive models\resnet_50\w8a8\*.zip -DestinationPath models\resnet_50\w8a8
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8
Copy-Item models\resnet_50\w8a8\resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

Run Python:

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/026_run_multiple_models/run_multiple_models.py
```

Run the prebuilt C++ tutorial:

```bash
./lib/sima-pcie-host/tutorials/tutorial_026_run_multiple_models
```

Or rebuild it:

```bash
./build.sh --target tutorial_026_run_multiple_models
./build/tutorials-standalone/tutorial_026_run_multiple_models
```

On Windows, rebuild it with CMake from an **x64 Native Tools Command Prompt**
in your copy of the tutorials. Replace `C:\opencv\build` with the `build`
directory of the OpenCV package described in
[Run applications on Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows):

```bat
call "C:\Program Files\SimaPCIeHost\bin\sima-pcie-host-env.cmd"
set "OpenCV_DIR=C:\opencv\build"
cmake -S share\sima-pcie-host\tutorials -B build
cmake --build build --config Release --target tutorial_026_run_multiple_models
set "PATH=%OpenCV_DIR%\x64\vc16\bin;%PATH%"
build\Release\tutorial_026_run_multiple_models.exe
```

With the documented models and assets, both versions print output similar to:

```text
queue=0 model=resnet_50 output_shape=[1, 1000] top1=208 (Labrador retriever)
queue=1 model=yolo_v8s detections=...
  person score=... box=(...)
[OK] 026_run_multiple_models
```

The tutorial deliberately fixes ResNet-50 to queue 0 and YOLOv8s to queue 1.
Pass `--card N` only when using another card.

## In Practice

Queue assignment is an application resource decision: two live models cannot
own the same physical queue. Build models before starting work, report the
specific queue on failure, and close every successfully built model on both
normal and error paths. Separate `Model` instances keep results and failures
isolated while remaining easy to reason about.

For deployment diagnostics, continue with the
[PCIe model workflow](/develop-apps/development-workflow/pcie-model/) and the
[troubleshooting guide](/reference/troubleshooting/).

## Source Files

- `run_multiple_models.cpp`
- `run_multiple_models.py`
- `../assets/labrador.jpg`
- `../assets/street-scene.png`
