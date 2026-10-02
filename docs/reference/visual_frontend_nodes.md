---
title: EV74 Visual Frontend Nodes
description: "Neat Graph usage for FeatureHistogram, GriderFast, TrackDescriptor, TrackKLT, and MetoakDepth"
sidebar_position: 8
---

# EV74 visual frontend Nodes

Neat exposes the EV74 visual-frontend graphs as normal `Graph` Nodes.  Use the
public Node factories and option structs; do not call `processcvu`, ConfigManager,
or dispatcher APIs directly from application code.

| Node factory | Graph name | Graph ID | Purpose |
| --- | --- | ---: | --- |
| `nodes::FeatureHistogram` / `pyneat.nodes.feature_histogram` | `feature_histogram` | 235 | Grayscale image histogram |
| `nodes::GriderFast` / `pyneat.nodes.grider_fast` | `grider_fast` | 236 | Grid-distributed FAST features |
| `nodes::TrackDescriptor` / `pyneat.nodes.track_descriptor` | `track_descriptor` | 237 | FAST features plus descriptors |
| `nodes::TrackKLT` / `pyneat.nodes.track_klt` | `track_klt` | 238 | Pyramidal KLT tracking, optionally with detected replacement features |
| `nodes::MetoakDepth` | `simor_depth_map` | 20 | I420 and disparity to RGB, metric depth, and XYZ |

Graph IDs are useful for diagnostics and firmware/package parity checks.  They
are not required in application code.

## Tensor contract

Feature and tracking tensors use **logical batch shapes**.  If `batch_size == B`, a grayscale
image is `[B,H,W]`, not `[B*H,W]`.  The runtime handles any EV74 transport packing
internally.

| Node | Inputs | Public outputs |
| --- | --- | --- |
| `FeatureHistogram` | `input_image`: UInt8 `[B,H,W]` | `output_hist`: Int32 `[B,256]` |
| `GriderFast` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]` |
| `TrackDescriptor` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]`; `output_descriptors`: Int32 `[B,max_features,8]` |
| `TrackKLT` | `prev_image`: UInt8 `[B,H,W]`; `cur_image`: UInt8 `[B,H,W]`; `input_points`: Int32 `[B,num_points,2]` | `output_points`: Float32 `[B,num_points,2]`; `output_status`: Int32 `[B,num_points,1]`; plus `output_features`: Int32 `[B,1 + max_features*3]` only when `detect_new_features != 0` |

Feature-list tensors use this per-batch layout:

```text
[count, x0, y0, score0, x1, y1, score1, ...]
```

The descriptor graph currently requires `descriptor_words == 8`.  Changing that
is an EV74 ABI change and is rejected before dispatch.

## C++ quick start

```cpp
#include <neat.h>

#include <cstdint>
#include <vector>

using namespace simaai::neat;

Tensor make_gray_batch(int width, int height, int batch) {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * batch);
  // Fill pixels in batch-major order: b*height*width + y*width + x.
  auto tensor = Tensor::from_vector(pixels, {batch, height, width}, TensorMemory::EV74);
  tensor.layout = TensorLayout::HW;
  tensor.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::H, TensorAxisSemantic::W};
  tensor.route.name = "input_image";
  tensor.route.segment_name = "input_image";
  return tensor;
}

int main() {
  constexpr int width = 320;
  constexpr int height = 240;
  constexpr int batch = 2;

  Graph graph;

  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::UINT8;
  input.width = width;
  input.height = height;
  input.depth = 1;
  input.max_width = width;
  input.max_height = height * batch; // transport capacity; public tensor remains [B,H,W]
  input.max_depth = 1;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.buffer_name = "input_image";

  graph.add(nodes::Input(input));

  GriderFastOptions fast;
  fast.width = width;
  fast.height = height;
  fast.batch_size = batch;
  fast.max_features = 64;
  fast.threshold = 30;
  graph.add(nodes::GriderFast(fast));

  graph.add(nodes::Output());

  RunOptions run_opt;
  run_opt.output_memory = OutputMemory::Owned;

  Tensor image = make_gray_batch(width, height, batch);
  Run run = graph.build({image}, run_opt);
  TensorList outputs = run.run({image}, /*timeout_ms=*/30000);
  run.close();
}
```

## KLT with three inputs

`TrackKLT` consumes a tensor-set: previous image, current image, and input
points.  Name the routes to match the option fields.

```cpp
TrackKLTOptions klt;
klt.width = 320;
klt.height = 240;
klt.batch_size = 2;
klt.num_points = 32;
klt.max_features = 64;
klt.detect_new_features = 1; // publish output_features as the third output

graph.add(nodes::TrackKLT(klt));
```

Expected public outputs when `detect_new_features == 1`:

```text
output_points   Float32 [2,32,2]
output_status   Int32   [2,32,1]
output_features Int32   [2,193]
```

When `detect_new_features == 0`, Neat publishes only `output_points` and
`output_status`; the EV-visible features buffer remains an internal runtime
allocation.

## Metoak depth with six inputs

`MetoakDepth` is a C++-only Node using `simor_depth_map` (graph 20). It consumes decoded I420 planes, raw disparity, and per-frame calibration—not a raw SIMOR camera frame. Your application or ROS adapter must unpack SIMOR and select calibration before this Node; Neat does not replace that adapter.

Use even `width` in `[8,2048]` and even `height` in `[8,1536]`. S315 native depth geometry is `640x360`. Batch is fixed at one, with no leading batch dimension. Keep the following canonical route names and input order; aliases are rejected during contract compilation.

| Input route | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `y_src` | UInt8 | `[H,W]` | I420 Y |
| `u_src` | UInt8 | `[H/2,W/2]` | I420 U |
| `v_src` | UInt8 | `[H/2,W/2]` | I420 V |
| `disp_src` | UInt16 | `[H,W]` | Raw disparity; fixed subpixel scale 32 |
| `bf_mm_src` | Float32 | `[1]` | Calibrated baseline × focal length, in mm |
| `proj_src` | Float32 | `[3]` | Projection `{fx_fy,cx,cy}` |

| Output route | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `rgb_dst` | UInt8 | `[H,W,3]` | Interleaved RGB |
| `depth_dst` | UInt16 | `[H,W]` | Depth in mm; 0 means invalid |
| `points_dst` | Float32 | `[H,W,3]` | Interleaved XYZ in meters; NaN means invalid |

All three outputs are published together; `depth_dst` is the primary boundary description, not an output selector. Calibration BF and focal length must be positive and finite, and principal-point coordinates must be finite.

Build the Graph using six named tensors in EV74 memory matching the input table. This example configures the Node; supply actual decoded frame and calibration tensors from your adapter.

```cpp
#include <neat.h>

using namespace simaai::neat;

// inputs contains the six named, decoded EV74 tensors from the table above.
Run build_metoak_depth(const TensorList& inputs) {
  Graph graph;
  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.caps_override =
      "application/vnd.simaai.tensor, representation=(string)tensor-set, storage=(string)tensorbuffer";
  graph.add(nodes::Input(input));

  MetoakDepthOptions depth;
  depth.width = 640;
  depth.height = 360;
  graph.add(nodes::MetoakDepth(depth));
  graph.add(nodes::Output());

  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  return graph.build(inputs, options);
}
```

Run this Graph only with matching Internals and EV74 firmware containing graph 20. The feature/tracking validation command below covers the other four graphs, not `MetoakDepth`.

## Python surface

The four feature/tracking Nodes have Python bindings that mirror the C++ options/factory style. `MetoakDepth` currently has no Python binding. Create an options object, set public configuration, and add the Node to a `Graph`.

```python
import numpy as np
import pyneat

width, height, batch = 320, 240, 2

opt = pyneat.GriderFastOptions()
opt.width = width
opt.height = height
opt.batch_size = batch
opt.max_features = 64
print(opt.summary())

graph = pyneat.Graph()
input_opt = pyneat.InputOptions()
input_opt.payload_type = pyneat.PayloadType.Tensor
input_opt.format = pyneat.Format.UINT8
input_opt.width = width
input_opt.height = height
input_opt.max_width = width
input_opt.max_height = height * batch
input_opt.memory_policy = pyneat.InputMemoryPolicy.Ev74
input_opt.buffer_name = "input_image"

graph.add(pyneat.nodes.input(input_opt))
graph.add(pyneat.nodes.grider_fast(opt))
graph.add(pyneat.nodes.output())

image_np = np.zeros((batch, height, width), dtype=np.uint8)
image = pyneat.Tensor.from_numpy(image_np, memory="ev74")
image.layout = pyneat.TensorLayout.HW
# If setting route metadata from Python in a custom app, keep it aligned with
# the option names used above.
```

## Safety checks

The four feature/tracking Nodes validate graph envelopes before EV dispatch. They reject:

- non-positive dimensions or counts;
- unsupported batch sizes;
- thresholds outside `[0,255]`;
- duplicate/empty tensor names;
- `TrackDescriptorOptions.descriptor_words != 8`;
- invalid KLT window, level, and detect-mode values;
- undersized runtime tensors during pre-dispatch negotiation.

This matters because illegal buffers can wedge EV74.  Keep validation failures as
host-side errors and do not bypass the Node contract path.

## Fast validation command

The fast customer-style DevKit gate is:

```bash
ctest --test-dir /workspace/core_graph_changes/build/tests \
  -R visual_frontend_ --output-on-failure
```

It runs:

- all four visual graphs with `320x240`, `batch_size=2`, `detect_new_features=1`;
- a targeted KLT no-detect ABI check;
- a negative pre-dispatch guard check that confirms illegal batch input is
  rejected before EV74.
