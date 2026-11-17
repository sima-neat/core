---
title: "EV74 Visual Frontend Nodes"
description: "Public Graph contracts for FeatureHistogram, GriderFast, TrackDescriptor, and TrackKLT"
sidebar_position: 8
---

# EV74 Visual Frontend Nodes

Use these EV74 operations through the public `Graph` node factories. Install a matching Core, Scheduler, Runner, Kernels and EV74 firmware package set. Legacy graph IDs are not a native firmware-compatibility check; applications do not need dispatcher IDs.

The quality options preserve existing source usage and default output positions, but change the C++ layout of `TrackKLTOptions`; existing C++ binaries are not ABI-compatible. Rebuild C++ applications and the Python extension against the matching Core version. Use a matching Kernels, Runner, Scheduler and EV74 firmware cohort for v2 tracker requests. Existing native v1 requests remain supported; this does not make v2 requests compatible with older firmware.

| Node factory | Purpose |
| --- | --- |
| `nodes::FeatureHistogram` / `pyneat.nodes.feature_histogram` | Grayscale histogram |
| `nodes::GriderFast` / `pyneat.nodes.grider_fast` | Grid-distributed FAST corners |
| `nodes::TrackDescriptor` / `pyneat.nodes.track_descriptor` | FAST corners and BRIEF descriptors |
| `nodes::TrackKLT` / `pyneat.nodes.track_klt` | Pyramidal KLT tracking and optional current-frame feature detection |

## Tensor contracts

All image tensors have logical shape `[B,H,W]`, where `B` is `batch_size`. Do not flatten the batch into image height. Device-backed inputs may use separate allocations or valid shared-storage views; no CPU-copy workaround is required.

| Node | Inputs | Default public outputs |
| --- | --- | --- |
| `FeatureHistogram` | UInt8 `[B,H,W]` | Int32 `[B,256]` |
| `GriderFast` | UInt8 `[B,H,W]` | Int32 `[B,1+3*max_features]` |
| `TrackDescriptor` | UInt8 `[B,H,W]` | Int32 `[B,1+3*max_features]`; Int32 `[B,max_features,8]` |
| `TrackKLT` | UInt8 `[B,H,W]`; UInt8 `[B,H,W]`; Int32 `[B,num_points,2]` | Float32 `[B,num_points,2]`; Int32 `[B,num_points,1]` |

The packed feature format is `[count,x0,y0,score0,...]` for each batch item. Read only the first `count` records. Unused capacity is not a feature list. TrackDescriptor requires `descriptor_words == 8`; each descriptor row belongs to the corresponding selected feature row.

## Grid selection and boundaries

`grid_x` and `grid_y` divide the whole image into cells. For capacity `K` and `M` cells, each cell gets `K / M` slots; the first `K % M` row-major cells get one extra. Select strongest scores per cell, then order winners by decreasing score with ties in increasing `(y,x)` order. Sparse cells do not lend slots. `min_px_dist` suppresses closer points across cell boundaries; exact distance equality is accepted. Suppression does not refill discarded candidates, so the result can contain fewer than `max_features` points.

Plain FAST includes centers at `x=3`, `y=3`, `x=W-4` and `y=H-4`. TrackDescriptor instead needs its complete 31×31 BRIEF patch: centers must be at least 15 pixels from each edge. Missing row-3 descriptors are therefore intentional.

## KLT inputs and outputs

Pass an ordered `TensorList`: **previous image, current image, input points**. On this direct public multi-input route, a nonempty logical `route.name` must match its configured slot; it does not reorder the list. Correctly ordered unnamed tensors remain valid. Missing inputs, contradictory names, wrong types/shapes and inaccessible spans are rejected before submission, including on subsequent frames.

When reusing a compatible output tensor, shallow-copy it and rename or clear only `route.name` for the new public slot. Keep its storage, offsets and physical routing metadata intact. This rule does not rename internal graph links or replace an explicit model input-route processor.

By default, TrackKLT publishes points and status. `detect_new_features != 0` appends detected features from the **current** image. Disabled detection does not allocate a hidden feature output. `publish_error = true` appends Float32 `[B,num_points,1]` after the existing outputs: index 2 without detection, index 3 with detection. Existing output positions do not move.

### C++ example

Provide two `640×480` UInt8 image vectors and `64×2` Int32 seed coordinates. Each input below gets independent EV74 storage. The returned tuple is points, status, error.

```cpp
#include <neat.h>
#include <cstdint>
#include <vector>

using namespace simaai::neat;

TensorList track_pair(const std::vector<std::uint8_t>& previous,
                     const std::vector<std::uint8_t>& current,
                     const std::vector<std::int32_t>& xy) {
  constexpr int width = 640, height = 480, count = 64;
  auto image = [&](const auto& pixels, const char* name) {
    auto tensor = Tensor::from_vector(pixels, {1, height, width}, TensorMemory::EV74);
    tensor.layout = TensorLayout::HW;
    tensor.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::H,
                            TensorAxisSemantic::W};
    tensor.route.name = name;
    return tensor;
  };
  auto points = Tensor::from_vector(xy, {1, count, 2}, TensorMemory::EV74);
  points.route.name = "input_points";
  TensorList inputs{image(previous, "prev_image"), image(current, "cur_image"), points};

  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::UINT8;
  input.width = input.max_width = width;
  input.height = input.max_height = height;
  input.depth = input.max_depth = 1;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.buffer_name = "prev_image";
  input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                        "storage=(string)tensorbuffer";

  TrackKLTOptions klt;
  klt.width = width;
  klt.height = height;
  klt.num_points = count;
  klt.win_half = 7;
  klt.max_level = 2;
  klt.max_iters = 20;
  klt.publish_error = true;

  Graph graph;
  graph.add(nodes::Input(input));
  graph.add(nodes::TrackKLT(klt));
  graph.add(nodes::Output());
  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  Run run = graph.build(inputs, options);
  TensorList outputs;
  try {
    outputs = run.run(inputs, 30000);
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  return outputs;
}
```

## Quality controls

| Option | Default | Meaning |
| --- | --- | --- |
| `publish_error` | `false` | Publish final-coordinate mean absolute intensity residual. `output_error_name` defaults to `output_error`. |
| `error_threshold` | `-1.0` | Disabled at `-1`; otherwise a finite threshold in `[0,255]`. Reject when residual is greater than the threshold; equality passes. Filtering also works without publishing error. |
| `min_eig_threshold` | `0.0` | Disabled at `0`; otherwise a finite positive normalized minimum-eigenvalue threshold, applied to texture conditioning. |
| `require_convergence` | `false` | Require the final full-resolution update to meet the convergence criterion. Iteration-cap exhaustion does not imply failure when this option is false. |

Residual is an intensity difference, not pixel displacement or confidence. It is evaluated at the returned coordinate; `+inf` means no valid residual was evaluated. The texture measure divides the smallest central-gradient matrix eigenvalue by window area and `255²`; its scale is not interchangeable with OpenCV thresholds. Choose thresholds on representative calibration data, then validate retained-track recall as well as incorrect-track rejection.

Inspect status before using coordinates. Pre-solve invalid seeds retain their original coordinates with status 0. A finite computed out-of-image point, or a candidate rejected by a quality threshold, can retain its computed coordinates with status 0. Default status is solver/geometry validity, not proof of a correct real-world correspondence.

## Motion, borders and fixed counts

The current-image tile is recentered and reloaded when refinement leaves it; motion is not silently clamped to a ±4-pixel search box. Reloads and iterations remain bounded. Border tiles replicate edge pixels, using only in-image DMA. This removes the old pyramid-dependent excluded band, but does not guarantee tracking under arbitrary motion, occlusion or ambiguous texture.

Core seeds remain Int32; results are Float32. `num_points` is fixed when building the graph. For an unused slot, supply an invalid coordinate such as `(-1,-1)` and inspect status. `(0,0)` is a valid edge coordinate, not an inactive marker. Both image pyramids are rebuilt per call; previous-frame pyramid reuse is not implied.

## Python options

Python exposes the same options. Create input tensors with `Tensor.from_numpy(..., memory="ev74")`, preserve the ordered input contract and build the public node as in C++.

```python
import pyneat

options = pyneat.TrackKLTOptions()
options.width, options.height = 640, 480
options.num_points = 64
options.win_half, options.max_level, options.max_iters = 7, 2, 20
options.publish_error = True
options.error_threshold = -1.0
options.min_eig_threshold = 0.0
options.require_convergence = False
node = pyneat.nodes.track_klt(options)
```

## Validation

Configured port names must be nonempty and unique. Invalid dimensions, counts, windows, pyramid levels, descriptor sizes, and nonfinite/out-of-range quality parameters are rejected. Do not bypass these checks or reinterpret a tensor dtype to convert its values.

## Run the correctness tests

On the DevKit, set `CORE_TEST_DIR` to the installed Core test directory or the native build’s test directory, then run:

```bash
ctest --test-dir "$CORE_TEST_DIR" \
  -R '^(grider_fast|track_descriptor|track_klt)_ev_test$' --output-on-failure
```

Each registration executes its complete per-kernel matrix. GriderFast covers grid/distance and FAST edges; TrackDescriptor also checks exact descriptor alignment; TrackKLT covers independent inputs, frame reuse, validation, quality, motion, borders and its detector. The separate fast presets are histogram/KLT-no-detect/batch-guard smoke tests, not a substitute for this numerical coverage.

## Metoak depth (C++)

`nodes::MetoakDepth` restores the existing `MetoakDepthOptions` API. It consumes
six named tensors and publishes RGB, depth and XYZ together. This node performs
EV74 computation, not camera capture; synthetic tensors work without a camera.

Unlike the batched visual nodes above, Metoak always uses batch one, with no
leading batch dimension. Set even `W` in `[8,2048]` and even `H` in `[8,1536]`.
Keep the canonical route names and supply the inputs in the order below.

| Tensor | Dtype | Shape |
| --- | --- | --- |
| y_src | UInt8 | [H,W] |
| u_src, v_src | UInt8 | [H/2,W/2] |
| disp_src | UInt16 | [H,W] |
| bf_mm_src | Float32 | [1] |
| proj_src | Float32 | [3] |
| rgb_dst | UInt8 | [H,W,3] |
| depth_dst | UInt16 | [H,W] |
| points_dst | Float32 | [H,W,3] |

`y_src`, `u_src` and `v_src` are the separate I420 planes. Disparity is scaled by 32.
Supply positive finite `bf_mm_src` (baseline × focal length, in mm) and
`proj_src = {fx_fy, cx, cy}` for each frame; `fx_fy` must be positive and all
three values finite. Calibration can change between frames.

Depth is rounded to millimetres and saturated to UInt16; XYZ uses unrounded
depth in metres. Zero disparity produces zero depth and NaN XYZ. All three
outputs remain available even though depth is the primary output description.

```cpp
MetoakDepthOptions depth;
depth.width = 640;
depth.height = 360;
graph.add(nodes::MetoakDepth(depth));
```

The existing `debug`, `num_buffers` and `element_name` fields remain available.
`num_buffers = 0` uses the graph's default capacity; a positive value selects
this stage's execution capacity. C++ is the restored public surface; this change
does not add a Python Metoak factory.
