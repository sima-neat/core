---
title: Run an App
description: Run a YOLOv8 model and compose it into a small Graph app
sidebar_position: 2
mdx:
  format: mdx
---

# Run an App

![YOLOv8 detections from Neat on the source image](../../images/first_inference_hook.png)

## What You’ll Build

This page builds a small object-detection app: it runs a YOLOv8 model on a sample image and prints how many objects were detected.

The shortest path to run a model is to call `Model.run(...)` directly on a list of inputs — great for testing that a model loads and producing output. Most real applications need more structure, though: named inputs, one or more models, custom logic, branches, and outputs. [`Graph`](/reference/programming-model/graph) is the Neat API for authoring that application pipeline, and that is what we build here.

For this first app we keep the shape intentionally simple:

- A named _input_ (`nodes.input()`) marks where data enters the app.
- A _model_ (`graph.add(model)`) runs the model as one step in the pipeline.
- A named _output_ (`nodes.output()`) marks where your application reads the result.

![Assembling your Graph](../../images/hello-neat-graph-add-animation.svg)

The same API scales to much more complex applications later; here the goal is to learn the core composition pattern.

:::tip Pick your language
Use the **Python / C++** tabs in any code block below — your choice follows the site-wide language selector, so every snippet and the full program switch together.
:::

## Set up the project

1. **Create an assets directory** for the model and the input image:
    ```bash
    mkdir -p assets
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
    curl -L -o assets/tutorial_sample_image.png \
      https://docs.sima-neat.com/images/tutorial_sample_image.png
    ```

    You can also [open or download the sample image](../../images/tutorial_sample_image.png) from the docs.

## Walk through the code

Each step below is one part of the program. Read them in order, then copy the [full program](#full-program) at the end to build and run.

### 1. Load the image

OpenCV reads the file; Neat enters at the next step. YOLOv8s expects a 640&nbsp;×&nbsp;640 input, so resize first.

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
import cv2

bgr = cv2.imread("assets/tutorial_sample_image.png", cv2.IMREAD_COLOR)
bgr = cv2.resize(bgr, (640, 640))
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png", cv::IMREAD_COLOR);
cv::resize(bgr, bgr, cv::Size(640, 640));
```

</CodeTab>
</CodeTabs>

### 2. Configure `ModelOptions` and load the model

`ModelOptions` declares how the input is preprocessed and how the detector output is decoded. `Model` then reads the `.tar.gz`, validates its **MPK contract** against those options, and instantiates the pipeline.

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
opt = pyneat.ModelOptions()
opt.preprocess.kind = pyneat.InputKind.Image
opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
opt.preprocess.input_max_width = 640
opt.preprocess.input_max_height = 640
opt.preprocess.input_max_depth = 3
opt.preprocess.normalize.enable = pyneat.AutoFlag.On
opt.preprocess.normalize.mean = [0.485, 0.456, 0.406]
opt.preprocess.normalize.stddev = [0.229, 0.224, 0.225]
opt.decode_type = pyneat.BoxDecodeType.YoloV8
opt.score_threshold = 0.55
opt.nms_iou_threshold = 0.5
opt.top_k = 100
opt.boxdecode_original_width = 640
opt.boxdecode_original_height = 640

model = pyneat.Model("assets/yolo_v8s_mpk.tar.gz", opt)
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::Model::Options opt;
opt.preprocess.kind = simaai::neat::InputKind::Image;
opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
opt.preprocess.input_max_width = 640;
opt.preprocess.input_max_height = 640;
opt.preprocess.input_max_depth = 3;
opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
opt.preprocess.normalize.mean = std::array<float, 3>{0.485f, 0.456f, 0.406f};
opt.preprocess.normalize.stddev = std::array<float, 3>{0.229f, 0.224f, 0.225f};
opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
opt.score_threshold = 0.55f;
opt.nms_iou_threshold = 0.5f;
opt.top_k = 100;
opt.boxdecode_original_width = 640;
opt.boxdecode_original_height = 640;

simaai::neat::Model model("assets/yolo_v8s_mpk.tar.gz", opt);
```

</CodeTab>
</CodeTabs>

### 3. Compose the Graph

A `Graph` is the application pipeline. Each `add(...)` appends the next step: a named input, the model, and a named output.

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
graph = pyneat.Graph("hello_neat_app")
graph.add(pyneat.nodes.input("image"))
graph.add(model)
graph.add(pyneat.nodes.output("detections"))
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::Graph graph("hello_neat_app");
graph.add(simaai::neat::nodes::Input("image"));
graph.add(model);
graph.add(simaai::neat::nodes::Output("detections"));
```

</CodeTab>
</CodeTabs>

### 4. Build, push, and pull

Build the graph once, then push your image into the named input and pull the result from the named output.

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
run = graph.build([bgr], copy=True, image_format=pyneat.PixelFormat.BGR)
try:
    run.push("image", [bgr], copy=True, image_format=pyneat.PixelFormat.BGR)
    sample = run.pull_samples("detections", 20000)
finally:
    run.close()
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
auto run = graph.build(std::vector<cv::Mat>{bgr});
run.push("image", std::vector<cv::Mat>{bgr});
simaai::neat::Sample output = run.pull_samples("detections", 20000);
run.close();
```

</CodeTab>
</CodeTabs>

### 5. Read the detections

The YOLOv8 BBOX output is a `uint8` buffer that begins with a `uint32` detection count.

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
payload = bytes(sample.tensors[0].to_numpy(copy=False))
detections = struct.unpack_from("<I", payload, 0)[0] if len(payload) >= 4 else 0
print(f"detections={detections}")
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
std::uint32_t detections = 0;
simaai::neat::Mapping view = simaai::neat::require_single_tensor(output).map_read();
if (view.size_bytes >= sizeof(detections))
  std::memcpy(&detections, view.data, sizeof(detections));
std::cout << "detections=" << detections << "\n";
```

</CodeTab>
</CodeTabs>

## Full program

Create the files in your project directory, then build and run.

<CodeTabs>
<CodeTab label="Python" lang="python">

Create `hello_neat.py`:

```python
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


def yolo_model_options():
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.input_max_width = 640
    opt.preprocess.input_max_height = 640
    opt.preprocess.input_max_depth = 3
    opt.preprocess.normalize.enable = pyneat.AutoFlag.On
    opt.preprocess.normalize.mean = [0.485, 0.456, 0.406]
    opt.preprocess.normalize.stddev = [0.229, 0.224, 0.225]
    opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = 0.55
    opt.nms_iou_threshold = 0.5
    opt.top_k = 100
    opt.boxdecode_original_width = 640
    opt.boxdecode_original_height = 640
    return opt


def main() -> int:
    bgr = cv2.imread("assets/tutorial_sample_image.png", cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to read assets/tutorial_sample_image.png")
    bgr = cv2.resize(bgr, (640, 640))

    model = pyneat.Model("assets/yolo_v8s_mpk.tar.gz", yolo_model_options())

    # A Graph is the application pipeline. Each add(...) appends the next step.
    graph = pyneat.Graph("hello_neat_app")
    graph.add(pyneat.nodes.input("image"))
    graph.add(model)
    graph.add(pyneat.nodes.output("detections"))

    # Build once, then push into the named input and pull from the named output.
    run = graph.build([bgr], copy=True, image_format=pyneat.PixelFormat.BGR)
    try:
        run.push("image", [bgr], copy=True, image_format=pyneat.PixelFormat.BGR)
        sample = run.pull_samples("detections", 20000)
    finally:
        run.close()

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

</CodeTab>
<CodeTab label="C++" lang="cpp">

Create `CMakeLists.txt` and `main.cpp`:

```cmake title="CMakeLists.txt"
cmake_minimum_required(VERSION 3.16)
project(sima_neat_hello LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Supports both DevKit/native installs (system paths) and
# cross builds with SYSROOT exported (SDK sysroot paths).
if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH
    "$ENV{SYSROOT}/usr"
    "$ENV{SYSROOT}/usr/lib"
    "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
endif()

find_package(SimaNeat REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)

add_executable(sima_neat_hello main.cpp)
target_link_libraries(sima_neat_hello
  PRIVATE
    SimaNeat::sima_neat
    PkgConfig::OPENCV
)
```

```cpp title="main.cpp"
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

  // YOLOv8s expects a 640 x 640 input in this example.
  cv::resize(bgr, bgr, cv::Size(640, 640));
  return bgr;
}

simaai::neat::Model::Options yolo_model_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  opt.preprocess.input_max_width = 640;
  opt.preprocess.input_max_height = 640;
  opt.preprocess.input_max_depth = 3;
  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.normalize.mean = std::array<float, 3>{0.485f, 0.456f, 0.406f};
  opt.preprocess.normalize.stddev = std::array<float, 3>{0.229f, 0.224f, 0.225f};
  opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  opt.score_threshold = 0.55f;
  opt.nms_iou_threshold = 0.5f;
  opt.top_k = 100;
  opt.boxdecode_original_width = 640;
  opt.boxdecode_original_height = 640;
  return opt;
}

int main() {
  cv::Mat bgr = load_sample_image();

  simaai::neat::Model model("assets/yolo_v8s_mpk.tar.gz", yolo_model_options());

  // A Graph is the application pipeline. Each add(...) appends the next step.
  simaai::neat::Graph graph("hello_neat_app");
  graph.add(simaai::neat::nodes::Input("image"));
  graph.add(model);
  graph.add(simaai::neat::nodes::Output("detections"));

  // Build once, then push into the named input and pull from the named output.
  auto run = graph.build(std::vector<cv::Mat>{bgr});
  run.push("image", std::vector<cv::Mat>{bgr});
  simaai::neat::Sample output = run.pull_samples("detections", 20000);
  run.close();

  std::uint32_t detections = 0;
  simaai::neat::Mapping view = simaai::neat::require_single_tensor(output).map_read();
  if (view.size_bytes >= sizeof(detections))
    std::memcpy(&detections, view.data, sizeof(detections));

  std::cout << "detections=" << detections << "\n";
  std::cout << "[OK] Graph app completed\n";
  return 0;
}
```

**Build:**

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

</CodeTab>
</CodeTabs>

You should see a detection summary similar to:

```text
detections=3
[OK] Graph app completed
```

The exact count can vary by model pack and runtime version. The important part is that the app builds, runs, and reaches `[OK] Graph app completed`.

## What you built

![Hello Neat Graph app flow](../../images/hello-neat-graph-app-flow.svg)

The APIs map directly to that shape:

- `Graph` holds the application pipeline.
- `graph.add(...)` appends each step in order, so the three calls build the linear flow above.
- The named input and output become the runtime endpoints: `run.push("image", ...)` and `run.pull_samples("detections", ...)`.

## Next steps

For deeper graph composition, continue with the [Graph programming model](/reference/programming-model/graph).

From there, continue with broader SiMa Neat learning resources:

- Learn the [core programming model](/reference/programming-model/overview), which explains the main Neat concepts such as graphs, models, pipeline stages, and graph execution.
- Follow the [tutorials](/tutorials/), which walk through specific concepts and workflows step by step.
- Explore curated applications on the [apps portal](https://apps.sima-neat.com/portal), with source code in the [apps repository on GitHub](https://github.com/sima-neat/apps).
