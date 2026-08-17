# 025 Measure PCIe Detection Throughput

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

From the extracted PCIe host extras root, download YOLOv8s if it is not already
present:

```bash
sima-cli modelzoo get yolo_v8s
```

Run Python:

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/025_measure_pcie_detection_throughput/measure_pcie_detection_throughput.py
```

Run the prebuilt C++ tutorial:

```bash
./lib/sima-pcie-host/tutorials/tutorial_025_measure_pcie_detection_throughput
```

Or rebuild it:

```bash
./build.sh --target tutorial_025_measure_pcie_detection_throughput
./build/tutorials-standalone/tutorial_025_measure_pcie_detection_throughput
```

The exact timing depends on the host and card, but both programs use the same
measurement boundaries and print:

```text
completed=1000
elapsed_seconds=...
throughput_fps=...
average_latency_ms=...
total_detections=...
[OK] 025_measure_pcie_detection_throughput
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
[Run Models on Multiple PCIe Queues](/tutorials/run-models-on-multiple-pcie-queues)
to execute two different models concurrently.

## Source Files

- `measure_pcie_detection_throughput.cpp`
- `measure_pcie_detection_throughput.py`
- `../assets/street-scene.png`
