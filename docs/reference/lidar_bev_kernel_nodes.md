---
title: LiDAR and BEV kernel nodes
description: Run the SimaVoxelize, SimaPillarScatter, SimaBEVPoolPlan, and SimaBEVPool kernels-library operations as graph nodes
sidebar_position: 8
---

# LiDAR and BEV kernel nodes

`SimaVoxelize`, `SimaPillarScatter`, `SimaBEVPoolPlan`, and `SimaBEVPool` run the non-MLA stages
of PointPillars and BEVFusion. Each node runs one SiMa kernels-library operation
on the target you select. A node never falls back to another target or copies a
tensor to change its layout. If the kernels library rejects a request, the graph
build fails with the kernels' reason.

| Node | Operation | Targets |
| --- | --- | --- |
| `SimaVoxelize` | LiDAR points to the decorated, quantized pillar tensor | A65, EV74 |
| `SimaPillarScatter` | Pillar features to the dense BEV canvas | A65, EV74 |
| `SimaBEVPoolPlan` | Camera calibration to the BEV pooling plan | A65 |
| `SimaBEVPool` | Camera depth and context features to the BEV grid | A65, EV74 |

```cpp
#include <neat.h>

using namespace simaai::neat;

SimaVoxelizeOptions voxelize;
voxelize.voxel_size = {0.25f, 0.25f, 8.0f};
voxelize.point_cloud_range = {-50.0f, -50.0f, -5.0f, 50.0f, 50.0f, 3.0f};
voxelize.quantization.multiplier = 2.720007f; // 1 / the consuming MLA input's scale
voxelize.quantization.zero_point = 8;
voxelize.max_pillars = 40000;
voxelize.max_points_per_pillar = 64;
voxelize.target = SimaKernelTarget::EV74;

// One sample holds every input tensor of the node, in order.
InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::Ev74;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";

std::vector<float> xyzt = read_sweep();  // x, y, z, time offset per point
const auto n = static_cast<std::int64_t>(xyzt.size() / 4);
auto points = Tensor::from_vector(xyzt, {n, 4}, TensorMemory::EV74);
auto point_count = Tensor::from_vector(std::vector<std::int32_t>{static_cast<std::int32_t>(n)},
                                       {1}, TensorMemory::EV74);

Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaVoxelize(voxelize));
graph.add(nodes::Output());
auto run = graph.build(TensorList{points, point_count});
```

Python exposes the same option types and the `pyneat.nodes.sima_voxelize`,
`pyneat.nodes.sima_pillar_scatter`, `pyneat.nodes.sima_bev_pool_plan`, and
`pyneat.nodes.sima_bev_pool` factories.

## Input and output contracts

Each node reads the tensors of its input sample in the order below and publishes
its outputs in the order below. The tensor count, data types, and shapes are
checked when the graph is built.

| Node | Inputs | Outputs |
| --- | --- | --- |
| `SimaVoxelize` | FP32 `points` `[N,4]` (x, y, z, time offset); INT32 `point_count` `[1]` | INT8 or BF16 `decorated` `[1,M,T,10]`; INT32 `coordinates` `[M,2]` as (row, column); INT32 `count` `[1]` |
| `SimaPillarScatter` | INT8 or BF16 `features` `[M,C]`; INT32 `coordinates` `[M,2]`; INT32 `count` `[1]` | `canvas` `[1,H,W,C]` of the same type as `features` |
| `SimaBEVPoolPlan` | FP32 `camera2lidar`, `camera_intrinsics`, and `img_aug_matrix` `[N,4,4]`; FP32 `lidar_aug_matrix` `[4,4]` | UINT8 `plan` `[P]` |
| `SimaBEVPool` | INT8 or BF16 `depth`; `context` of the same type; UINT8 `plan` `[P]` | INT8 or BF16 `bev` `[1,nZ*C,nX,nY]` |

- **SimaVoxelize** reads only the first `point_count` points. `max_pillars` (`M`) and
  `max_points_per_pillar` (`T`) are fixed capacities; points and pillars beyond them
  are dropped. Unused rows hold the quantization zero point and zero coordinates.
- **SimaPillarScatter** reads only the first `count` pillars. It also accepts features
  with extra singleton axes, such as `[1,M,1,C]` from a pillar-feature model. `C`
  must be a multiple of 16. Empty cells hold `zero_point`, which must be the INT8
  code that the consuming model's input quantization assigns to 0.0. An INT8
  canvas's bytes are in dense NHWC order.
- **SimaBEVPoolPlan** takes its frustum (`xs`, `ys`, `ds`) and grid (`bx`, `dx`, `nx`)
  from the checkpoint's view transform. `SimaBEVPoolPlanFormat::V1`, the default,
  groups the points by BEV cell; use `SimaBEVPoolPlanFormat::V2` with
  `context_channels` for an EV74 `SimaBEVPool`. By default `P` is the worst-case plan
  size; bytes past the plan's used length are not written.
- **SimaBEVPool** sums `depth x context` over each cell's plan points and quantizes the
  sum with `quantization`. Set `cells` to the plan's `nx`, `plan_format` to the
  plan's `format`, and, for INT8 inputs, `quantization.multiplier` to
  `depth_scale * context_scale / output_scale`. With the default
  `SimaBEVPoolStorage::ChannelsLast`, inputs arrive as `[N,fH,fW,D]` and `[N,fH,fW,C]`,
  as camera models produce them. The output is published as logical
  `[1,nZ*C,nX,nY]` with channels-last strides, so its bytes are NHWC
  `[1,nX,nY,nZ*C]`. `SimaBEVPoolStorage::Contiguous` uses `[N,D,fH,fW]` and
  `[N,C,fH,fW]` inputs and a dense output.

INT8 outputs carry their quantization in `Tensor.semantic.quant` when the node
knows it: `scale = 1 / quantization.multiplier` for `SimaVoxelize` and for a
`SimaBEVPool` with BF16 inputs;
`depth_scale * context_scale / quantization.multiplier` for a `SimaBEVPool` whose
INT8 inputs both carry a scale; and the features' quantization for
`SimaPillarScatter`. Otherwise no quantization is published.

## BF16 tensors

- **SimaVoxelize** with `output_dtype = TensorDType::BFloat16` rounds each binary32
  decoration to nearest-even BF16, stores every NaN as `0x7FC0`, and fills empty
  slots with +0.0. Leave `quantization` at its default; any other value is rejected.
- **SimaPillarScatter** keeps the type of its features. BF16 features need
  `zero_point = 0`, so empty cells hold +0.0. A BF16 canvas is stored in the MLA tiled
  BF16 layout: each cell holds the low bytes of its `C` channels, then their high
  bytes. It is not dense NHWC, and the published tensor does not mark the
  difference: its dtype is BF16 and its shape `[1,H,W,C]`, so
  `copy_dense_bytes_tight()` and DLPack views such as `numpy.from_dlpack` expose
  the stored bytes, not BF16 values in NHWC order. Pass the canvas to an MLA model
  that reads this layout, or read it as bytes and deinterleave each cell.
- **SimaBEVPoolPlan** takes `input_dtype`, the type of the `SimaBEVPool` inputs. A
  `V2` plan budgets its EV74 caches in bytes, so a BF16 plan can be larger. A `V1`
  plan is the same for both types.
- **SimaBEVPool** accepts BF16 `depth` and `context` with zero points of 0. Each
  cell's sum is binary32, taken in the plan's point order, so `V1` and `V2` plans can
  give different BF16 bytes. A BF16 `bev` stores each sum rounded to
  nearest-even, with NaN as `0x7FC0`, and needs the default `quantization`. An INT8
  `bev` from BF16 inputs quantizes the sum: set `quantization.multiplier` to
  `1 / output_scale`. INT8 inputs cannot produce a BF16 `bev`.

## Connect the nodes into a pipeline

These nodes are ordinary graph stages, but their inputs come from more than one
upstream stage:

- In PointPillars, `SimaPillarScatter` needs the pillar features from the
  pillar-feature model together with `coordinates` and `count` from `SimaVoxelize`.
- In BEVFusion, `SimaBEVPool` needs `depth` and `context` from the camera model together
  with the plan from `SimaBEVPoolPlan`.

Join these tensors into one input sample, in the order above, before the node.
For example, pull both results in your application and push them to the node's
graph together.

`SimaPillarScatter` copies feature bytes without requantizing them, so the features
must already use the input quantization of the model that reads the canvas.
`SimaBEVPool` takes all cameras in one `[N,...]` tensor.

## Plans and state

A plan depends only on the calibration and the `SimaBEVPoolPlanOptions`. With
`cache_plan = true`, a frame whose calibration bytes equal those of the previous
frame copies that frame's plan instead of computing it again. The cache lives in
the kernels-library context that the runtime owns for each in-flight frame, so a
hit is possible but not guaranteed when several frames are in flight. The plan
bytes are the same either way.

`SimaBEVPool` keeps no state between frames. It checks the plan's header and
contents on every execution and rejects a plan that does not match its shapes.

## Target limits

EV74 adds restrictions to the A65 contract:

- `SimaBEVPool` needs channels-last `context` with a multiple of 16 channels and a
  channels-last `bev`. A `V2` plan also needs channels-last `depth`.
- `SimaBEVPoolPlan` runs on A65 only.

A request that one target does not support fails on that target. The node does
not choose another target for you.
