---
title: Sample features with a dynamic grid
description: Run A65 NEON GridSample2D through a Neat graph
sidebar_position: 9
---

# Sample features with a dynamic grid

`SimaGridSample2D` performs bilinear sampling on A65. Each input sample holds
features `[N,H,W,C]` (FP32 or BF16), then a grid `[N,Ho,Wo,2]` (FP32), with
normalized `(x,y)` coordinates. The output is `sampled` `[N,Ho,Wo,C]` with the
feature dtype and channels-last layout. All dimensions must be positive.

## Build and run

Use a matching Kernels, Runner, Scheduler and Core package set containing
`grid_sample_2d`. This C++ fragment runs inside an application linked to
`SimaNeat::sima_neat`; provide contiguous NHWC feature values and NHW2 grid values
in the two vectors before building the graph.

```cpp
#include <neat.h>
using namespace simaai::neat;

std::vector<float> features(1 * 16 * 16 * 16, 1.0f);
std::vector<float> grid(1 * 32 * 8 * 2, 0.0f);
auto feature = Tensor::from_vector(features, {1,16,16,16}, TensorMemory::CPU);
feature.layout = TensorLayout::HWC;
auto coordinates = Tensor::from_vector(grid, {1,32,8,2}, TensorMemory::CPU);

InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::SystemMemory;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";
SimaGridSample2DOptions sampling;
sampling.padding = SimaGridSample2DPadding::Zeros;
sampling.align_corners = false;
Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaGridSample2D(sampling));
graph.add(nodes::Output());
auto run = graph.build(TensorList{feature, coordinates});
auto output = run.run(TensorList{feature, coordinates}, 30000);
run.close();
```

Change grid values on each submission while keeping shapes and dtypes fixed.
Use the normal asynchronous push/pull API for multiple in-flight samples.
Retained zero-copy outputs hold producer-buffer loans; release them to let the
pool progress. Owned outputs remain valid independently of the run.

## Sampling options and precision

`SimaGridSample2DPadding::Zeros` substitutes zero for invalid feature taps;
`SimaGridSample2DPadding::Border` clamps coordinates to edge pixel centers.
`align_corners=false` uses half-pixel centers; true maps normalized -1/+1 to
corner pixel centers. NaN grid components mean normalized -1. Infinite components
produce a zero sample with zero padding, or select the edge with border padding.

Coordinates, weights and accumulation use FP32. Each sum uses an initial multiply
followed by three explicit fused multiply-adds. BF16 features widen exactly and
the final output rounds once to nearest-even. This precision differs from an MLA
implementation that rounds grid values or intermediate coordinates to BF16.

NCHW, tiled storage, non-dense views and nonidentity quantization require explicit
conversion before this node. Nearest, cubic and reflection modes are unsupported.
Adding this node does not replace GridSample operations inside compiled MLA models.
For performance comparisons, use identical saved inputs and timing boundaries;
ELF profiling with unknown grid initialization has unconfirmed grid provenance.
