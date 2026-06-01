---
title: Run an App
description: Compose a small Graph app around a model
sidebar_position: 3
---

# Run an App

## What You’ll Build

In [Run a Model](./run_first_model), you used `Model.run(...)` as the shortest path from a compiled model package and input image to detection output.
That direct path is useful for checking that a model loads, testing inputs and options, and running simple single-model inference.

Most real applications need more structure: named inputs, one or more models, custom logic, branches, and outputs.
`Graph` is the Neat API for authoring that production-level application pipeline.
It gives developers a way to compose those pieces in application code while Neat handles the runtime wiring underneath.

This page uses the same model and image from [Run a Model](./run_first_model), but places the model inside a small `Graph` application.
For this first app, we keep the shape intentionally simple:

- A named _input_ (`nodes.input()`) marks where data enters the app.
- A _model_ (`graph.add(model)`) runs the model as one step in the pipeline.
- A named _output_ (`nodes.output()`) marks where your application reads the result.

![Assembling your Graph](../../images/hello-neat-graph-add-animation.svg)

The same API can scale to much more complex applications later.
Here, the goal is just to learn the core composition pattern before adding more pieces.

## Build and Run

:::note Before you start
This page reuses the same working directory, model, and image from [Run a Model](./run_first_model).
If you skipped that page, create the `assets` directory and download `assets/yolo_v8s_mpk.tar.gz` and `assets/tutorial_sample_image.png` first.
:::

<div class="minimal-tabs">
  <input type="radio" name="hello-step3-tabs-lang" id="hello-step3-tab-py" checked>
  <label for="hello-step3-tab-py">Python</label>
  <input type="radio" name="hello-step3-tabs-lang" id="hello-step3-tab-cpp">
  <label for="hello-step3-tab-cpp">C++</label>

  <div class="minimal-tab-panel">
<p>Keep the same <code>CMakeLists.txt</code> from <a href="./">Hello Neat!</a> It already links the app with Neat and OpenCV for this example.</p>

Replace `main.cpp` with:

```cpp {50-64}
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

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
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.input_max_width = 640;
  opt.preprocess.input_max_height = 640;
  opt.preprocess.input_max_depth = 3;
  opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.resize.width = 640;
  opt.preprocess.resize.height = 640;
  opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  opt.preprocess.color_convert.output_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.normalize.mean = std::array<float, 3>{0.485f, 0.456f, 0.406f};
  opt.preprocess.normalize.stddev = std::array<float, 3>{0.229f, 0.224f, 0.225f};
  opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  opt.score_threshold = 0.55f;
  opt.nms_iou_threshold = 0.5f;
  opt.top_k = 100;
  return opt;
}

int main() {
  // 1. Load the sample image and resize it for the model.
  cv::Mat bgr = load_sample_image();

  // 2. Build the app: input -> model -> output.
  simaai::neat::Model::Options options = yolo_model_options();
  simaai::neat::Model model("assets/yolo_v8s_mpk.tar.gz", options);

  // A Graph is the application pipeline. Each add(...) appends the next step.
  simaai::neat::Graph graph("hello_neat_app");
  graph.add(simaai::neat::nodes::Input("image"));
  graph.add(model);
  graph.add(simaai::neat::nodes::Output("detections"));

  // Build once, then push into the named input and pull from the named output.
  auto run = graph.build(std::vector<cv::Mat>{bgr});
  // Send the image into the Graph input named "image".
  run.push("image", std::vector<cv::Mat>{bgr});
  // Read the detection result from the Graph output named "detections".
  simaai::neat::Sample output = run.pull_samples("detections", 20000);
  run.close();

  // 3. The BBOX output starts with a uint32 detection count.
  std::uint32_t detections = 0;
  simaai::neat::Mapping view = simaai::neat::require_single_tensor(output).map_read();
  if (view.size_bytes >= sizeof(detections))
    std::memcpy(&detections, view.data, sizeof(detections));

  std::cout << "detections=" << detections << "\n";
  std::cout << "[OK] Graph app completed\n";
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

```python {58-73}
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
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.input_max_width = 640
    opt.preprocess.input_max_height = 640
    opt.preprocess.input_max_depth = 3
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = 640
    opt.preprocess.resize.height = 640
    opt.preprocess.color_convert.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.normalize.enable = pyneat.AutoFlag.On
    opt.preprocess.normalize.mean = [0.485, 0.456, 0.406]
    opt.preprocess.normalize.stddev = [0.229, 0.224, 0.225]
    opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = 0.55
    opt.nms_iou_threshold = 0.5
    opt.top_k = 100
    return opt


def main() -> int:
    mpk = Path("assets/yolo_v8s_mpk.tar.gz")
    image = Path("assets/tutorial_sample_image.png")

    # 1. Load the sample image and resize it for the model.
    bgr = load_image(image)

    # 2. Build the app: input -> model -> output.
    options = yolo_model_options()
    model = pyneat.Model(str(mpk), options)

    # A Graph is the application pipeline. Each add(...) appends the next step.
    graph = pyneat.Graph("hello_neat_app")
    graph.add(pyneat.nodes.input("image"))
    graph.add(model)
    graph.add(pyneat.nodes.output("detections"))

    # Build once, then push into the named input and pull from the named output.
    run = graph.build([bgr], copy=True, image_format=pyneat.PixelFormat.BGR)
    try:
        # Send the image into the Graph input named "image".
        run.push("image", [bgr], copy=True, image_format=pyneat.PixelFormat.BGR)
        # Read the detection result from the Graph output named "detections".
        sample = run.pull_samples("detections", 20000)
    finally:
        run.close()

    # 3. The BBOX output starts with a uint32 detection count.
    payload = bytes(sample.tensors[0].to_numpy(copy=False))
    detections = struct.unpack_from("<I", payload, 0)[0] if len(payload) >= 4 else 0

    print(f"detections={detections}")
    print("[OK] Graph app completed")
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
[OK] Graph app completed
```

The exact count can vary by model pack and runtime version. The important part is that the app builds, runs, and reaches `[OK] Graph app completed`.

## What you built

This example turns the direct model call from [Run a Model](./run_first_model) into a small Neat application:

![Hello Neat Graph app flow](../../images/hello-neat-graph-app-flow.svg)

The APIs map directly to that shape:

- `Graph` holds the application pipeline.
- `graph.add(...)` appends each step in order, so the three calls build the linear flow above.
- The named input and output become the runtime endpoints: `run.push("image", ...)` and `run.pull_samples("detections", ...)`.

For deeper graph composition, continue with the [Graph programming model](/reference/programming-model/graph).
