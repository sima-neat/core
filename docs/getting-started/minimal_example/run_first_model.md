---
title: Run a Model
description: Run a model on a sample image with Neat
sidebar_position: 2
---

# Run a Model

## What You’ll Build

This page builds a small object detection app that runs a YOLOv8 model on a sample image.
The application loads a compiled model package, prepares the image, runs inference, decodes bounding boxes, and prints how many detections were found. You will build from the same working directory you created in [Hello Neat!](./), so the project structure and build commands stay familiar.

This page introduces two Neat concepts:

* [`Model`](/reference/programming-model/model) loads the compiled model package and gives you the `run(...)` entry point.
* [`ModelOptions`](/tutorials/004-configure-model-options) tells Neat how to prepare the image and decode the detector output.

You do not need to master the full API yet; for now, focus on how `Model` and `ModelOptions` work together to run a compiled model.

![Hello Neat YOLOv8 flow](../../images/hello-neat-yolov8-flow.svg)

## Build and Run

**To start:**

1. **Make an assets directory** where we will store our model and input image:
    ```bash
    mkdir -p assets
    cd assets
    ```
2. **Download the model:**
    ```bash
    sima-cli modelzoo -v 2.0.0 get yolo_v8s
    ```
    :::note sima-cli model download
    If `sima-cli` writes the model somewhere other than the `assets` directory, copy that file into `assets/yolo_v8s_mpk.tar.gz`.
    :::
3. **Download the sample image:**
    ```bash
    curl -L \
      -o tutorial_sample_image.png \
      https://docs.sima-neat.com/images/tutorial_sample_image.png
    ```

    You can also [open or download the sample image](../../images/tutorial_sample_image.png) from the docs.
4. **Return to your project directory:**
    ```bash
    cd ..
    ```

**Edit the script:**

Now that we have created a directory for the model and the input image, we will edit the script from the [Hello Neat!](./)
introduction to load the model, run it, and count the number of detections.

:::note
The main focus should be on the highlighted lines of the code. That is where the application creates the `Model()`, a
`Tensor()` is generated from the image, and the model is run using `Model.run()`.
:::

<div class="minimal-tabs">
  <input type="radio" name="hello-step2-tabs-lang" id="hello-step2-tab-py" checked>
  <label for="hello-step2-tab-py">Python</label>
  <input type="radio" name="hello-step2-tabs-lang" id="hello-step2-tab-cpp">
  <label for="hello-step2-tab-cpp">C++</label>

  <div class="minimal-tab-panel">
<p>Keep the same <code>CMakeLists.txt</code> from <a href="./">Hello Neat!</a> It already links the app with Neat and OpenCV for this example.</p>

Replace `main.cpp` with:

```cpp {46-48}
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

cv::Mat load_sample_image() {
  cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png", cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to load sample image");

  // YOLOv8s expects a 640 x 640 input in this tutorial.
  cv::resize(bgr, bgr, cv::Size(640, 640));
  return bgr;
}

simaai::neat::Model::Options yolo_model_options() {
  simaai::neat::Model::Options opt;
  opt.format = "BGR";
  opt.input_max_width = 640;
  opt.input_max_height = 640;
  opt.input_max_depth = 3;
  opt.preproc.normalize = true;
  opt.preproc.channel_mean = std::array<float, 3>{0.485f, 0.456f, 0.406f};
  opt.preproc.channel_stddev = std::array<float, 3>{0.229f, 0.224f, 0.225f};
  opt.decode_type = "yolov8";
  opt.score_threshold = 0.55f;
  opt.nms_iou_threshold = 0.5f;
  opt.top_k = 100;
  opt.original_width = 640;
  opt.original_height = 640;
  return opt;
}

int main() {
  // 1. Load the sample image and resize it for the model.
  cv::Mat bgr = load_sample_image();

  // 2. Load the compiled model package and run inference.
  simaai::neat::Model::Options options = yolo_model_options();
  simaai::neat::Model yolo("assets/yolo_v8s_mpk.tar.gz", options);
  simaai::neat::Sample output = yolo.run(bgr, /*timeout_ms=*/2000);

  // 3. The BBOX output starts with a uint32 detection count.
  std::uint32_t detections = 0;
  if (output.tensor.has_value()) {
    simaai::neat::Mapping view = output.tensor->map_read();
    if (view.size_bytes >= sizeof(detections))
      std::memcpy(&detections, view.data, sizeof(detections));
  }
  std::cout << "detections=" << detections << "\n";
  std::cout << "[OK] YOLOv8 completed\n";
  return 0;
}
```

Rebuild with the same commands:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Run:**

* **On the DevKit**
    ```bash
    ./build/sima_neat_hello
    ```
* **On the Neat SDK from host**
  ```bash
  dk build/sima_neat_hello
  ```
  </div>

  <div class="minimal-tab-panel">
<p>Replace <code>hello_neat.py</code> with:</p>

```python {53-56}
#!/usr/bin/env python3
from __future__ import annotations

import struct
import sys
from pathlib import Path

try:
    import pyneat
except ImportError:
    sys.exit(
        "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
        "Run: source ~/pyneat/bin/activate"
    )

import cv2


def load_image(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    # YOLOv8s expects a 640 x 640 input in this tutorial.
    return cv2.resize(bgr, (640, 640))


def yolo_model_options():
    opt = pyneat.ModelOptions()
    opt.format = "BGR"
    opt.input_max_width = 640
    opt.input_max_height = 640
    opt.input_max_depth = 3
    opt.decode_type = "yolov8"
    opt.score_threshold = 0.55
    opt.nms_iou_threshold = 0.5
    opt.top_k = 100
    opt.original_width = 640
    opt.original_height = 640
    opt.preproc.normalize = True
    opt.preproc.channel_mean = [0.485, 0.456, 0.406]
    opt.preproc.channel_stddev = [0.229, 0.224, 0.225]
    return opt


def main() -> int:
    mpk = Path("assets/yolo_v8s_mpk.tar.gz")
    image = Path("assets/tutorial_sample_image.png")

    # 1. Load the sample image and resize it for the model.
    bgr = load_image(image)

    # 2. Load the compiled model package and run inference.
    options = yolo_model_options()
    model = pyneat.Model(str(mpk), options)
    tensor = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)
    sample = model.run(tensor, timeout_ms=2000)

    # 3. The BBOX output starts with a uint32 detection count.
    if sample.tensor is not None:
        payload = bytes(sample.tensor.to_numpy(copy=False))
        detections = struct.unpack_from("<I", payload, 0)[0] if len(payload) >= 4 else 0
        print(f"detections={detections}")
    else:
        print(f"raw_output_heads={len(sample.fields or [])}")

    print("[OK] YOLOv8 completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Run:**

* **On the DevKit**
  ```bash
  source ~/pyneat/bin/activate
  python3 hello_neat.py
  ```
* **On the Neat SDK from host**
  ```bash
  dk hello_neat.py
  ```
  </div>
</div>

You should see a detection summary similar to:

```text
detections=3
[OK] YOLOv8 completed
```

The exact count can vary by model pack and runtime version. The important part is that the app builds, runs, and reaches `[OK] YOLOv8 completed`.

## What you built

This example follows the same high-level path used by larger Neat applications:

- Load a compiled model package (`.tar.gz`) as a `Model`.
- Convert the input image into the format the model expects.
- Run inference through Neat runtime stages.
- Decode raw detector output into bounding boxes.

For a deeper explanation of box decoding, thresholds, NMS, and detector output structure, continue with [Read Detection Boxes from Model Output](/tutorials/006-read-detection-boxes).

## Next Steps

Once YOLOv8 runs, continue with [Run an App](./run_an_app) to place the same model inside a small Graph application.

From there, continue with broader SiMa Neat learning resources:

- Learn the [core programming model](/reference/programming-model/overview), which explains the main Neat concepts such as sessions, models, pipeline stages, and graph execution.
- Follow the [tutorials](/tutorials/), which walk through specific concepts and workflows step by step.
- Explore curated applications on the [apps portal](https://apps.sima-neat.com/portal), with source code in the [apps repository on GitHub](https://github.com/sima-neat/apps).
