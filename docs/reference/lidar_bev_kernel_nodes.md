---
title: LiDAR and BEV kernel nodes
description: Run the SimaVoxelize, SimaPillarScatter, SimaBEVPoolPlan, SimaBEVPool, and SimaCenterHeadDecode kernels-library operations as graph nodes
sidebar_position: 8
---

# LiDAR and BEV kernel nodes

`SimaVoxelize`, `SimaPillarScatter`, `SimaBEVPoolPlan`, and `SimaBEVPool` run the non-MLA stages
of PointPillars and BEVFusion. `SimaCenterHeadDecode` turns the head outputs of
CenterPoint-style models into 3D boxes. Each node runs one SiMa kernels-library operation
on the target you select. A node never falls back to another target or copies a
tensor to change its layout. If the kernels library rejects a request, the graph
build fails with the kernels' reason.

| Node | Operation | Targets |
| --- | --- | --- |
| `SimaVoxelize` | LiDAR points to the decorated, quantized pillar tensor | A65, EV74 |
| `SimaPillarScatter` | Pillar features to the dense BEV canvas | A65, EV74 |
| `SimaBEVPoolPlan` | Camera calibration to the BEV pooling plan | A65 |
| `SimaBEVPool` | Camera depth and context features to the BEV grid | A65, EV74 |
| `SimaCenterHeadDecode` | CenterHead outputs to 3D boxes | A65 |

This graph-construction fragment belongs inside your application. Implement
`read_sweep()` to return interleaved x, y, z and time-offset values: a nonempty
vector of `4*N` floats, with `N` representable as INT32. The function is an
application-provided reader, not a Neat API. The fragment builds the graph; it
does not submit a frame or collect an output.

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
`pyneat.nodes.sima_pillar_scatter`, `pyneat.nodes.sima_bev_pool_plan`,
`pyneat.nodes.sima_bev_pool`, and `pyneat.nodes.sima_centerhead_decode` factories.

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
| `SimaCenterHeadDecode` | FP32 or BF16 head maps on two to six ports | FP32 `boxes` `[K,9]` or `[K,7]`; FP32 `scores` `[K]`; INT32 `labels` `[K]`; INT32 `count` `[1]` |

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
- **SimaCenterHeadDecode** places the heads by `heatmap_input` to `vel_input`. By default
  three ports hold the heatmap; reg, height, dim and vel; and rot. `K` is
  `tasks.size() * post_max_per_task`. `boxes` hold x, y and z of the bottom centre, dx, dy,
  dz, yaw and, with `velocity`, vx and vy. `labels` are heatmap channels, counted across
  the head tasks. Rows from `count` on are zero.

### Logical axes and physical storage

For `SimaBEVPool`, logical axis order and physical byte order are separate.
A tensor tagged `TensorLayout::HWC` has logical `[N,H,W,K]` axes; a tensor tagged
`TensorLayout::CHW` has logical `[N,K,H,W]` axes. The node honors that tag and
checks its strides against `input_storage`. With `ChannelsLast`, HWC inputs
must be dense, while CHW inputs must have channels-last strides. With
`Contiguous`, inputs must be dense CHW tensors; HWC tags are rejected.
You can also declare the same canonical axes through `Tensor.axis_semantics`;
when both annotations are present, they must agree.

For untagged `ChannelsLast` inputs, the compatibility convention is dense NHWC
or non-dense logical NCHW with channels-last strides. If singleton dimensions
make a logical NCHW view also dense, set `Tensor.layout = TensorLayout::CHW`
explicitly. Strides alone cannot identify the logical axes in that case. Pool
outputs carry CHW logical axes and the strides for `output_storage`; the node
does not transpose or copy payload bytes to change layouts.

### Quantization metadata

INT8 outputs carry their quantization in `Tensor.semantic.quant` when the node
knows it: `scale = 1 / quantization.multiplier` for `SimaVoxelize` and for a
`SimaBEVPool` with BF16 inputs;
`depth_scale * context_scale / quantization.multiplier` for a `SimaBEVPool` whose
INT8 inputs both carry a scale; and the features' quantization for
`SimaPillarScatter`. Otherwise no quantization is published.

For INT8 Scatter inputs, per-channel quantization is supported only on the final
feature axis `C`, including shapes such as `[1,M,1,C]`. Provide `C` finite positive
scales and `C` INT8 zero points. Every zero point must equal the Scatter
`zero_point` used to fill empty cells. The canvas preserves those values and remaps the quantization axis
to `3` in `[1,H,W,C]`. Other quantization axes are rejected; pillar-axis parameters
cannot describe a fixed canvas after coordinates scatter the pillars.

`SimaBEVPool` does not support per-channel quantization metadata and rejects it
rather than silently removing it. Existing per-tensor metadata and explicit
zero-point options keep their behavior; absent input metadata remains unknown.

## BF16 tensors

- **SimaVoxelize** with `output_dtype = TensorDType::BFloat16` rounds each binary32
  decoration to nearest-even BF16, stores every NaN as `0x7FC0`, and fills empty
  slots with +0.0. Leave `quantization` at its default; any other value is rejected.
- **SimaPillarScatter** keeps the type of its features. BF16 features need
  `zero_point = 0`, so empty cells hold +0.0. A BF16 canvas is stored in the MLA tiled
  BF16 layout: each cell holds the low bytes of its `C` channels, then their high
  bytes. `semantic.tess` describes that storage explicitly: `slice_shape = [1,1,1,C]`,
  `axes = [0,1,2,3]`, `format = "MLA_TILED_BF16_COMPACT"`, and `storage_bytes` is the
  checked payload extent. D/H/W/C tiles are ordered C/D/H/W; each tile payload is
  D/H/W/C with per-cell low/high BF16 byte planes and 16-byte tile alignment.
  `copy_dense_bytes_tight()`, typed `data_ptr()` and DLPack reject this non-dense
  layout rather than misrepresent its values. `copy_payload_bytes()` reads the
  bounded stored bytes; `clone()` and Owned graph output preserve those bytes and
  metadata without detessellation. A layout-aware consumer such as `SimaArgMax`
  receives the producer's checked layout directly within the same graph. Dense-only
  consumers and unsupported tiled graph ingress reject it.
  INT8 full-channel cell tiles remain byte-identical to dense NHWC and keep ordinary
  dense-copy/DLPack behavior. Other `TessSpec` format-only annotations retain their
  established behavior; a nonzero `storage_bytes` denotes explicit tiled storage.
- **SimaBEVPoolPlan** takes `input_dtype`, the type of the `SimaBEVPool` inputs. A
  `V2` plan budgets its EV74 caches in bytes, so a BF16 plan can be larger. A `V1`
  plan is the same for both types.
- **SimaBEVPool** accepts BF16 `depth` and `context` with zero points of 0. Each
  cell's sum is binary32, taken in the plan's point order, so `V1` and `V2` plans can
  give different BF16 bytes. A BF16 `bev` stores each sum rounded to
  nearest-even, with NaN as `0x7FC0`, and needs the default `quantization`. An INT8
  `bev` from BF16 inputs quantizes the sum: set `quantization.multiplier` to
  `1 / output_scale`. INT8 inputs cannot produce a BF16 `bev`.

The appended `TessSpec::axes` and `TessSpec::storage_bytes` fields have defaults, so existing two-field aggregate initializers remain source-compatible. The public struct's binary layout changes: rebuild C++ consumers and Python bindings against matching Core headers and libraries; do not reuse binaries built with the earlier layout.

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

## Decode CenterHead outputs

`SimaCenterHeadDecode` is mmdetection3d's CenterPoint box coder followed by NMS per
head task, for models such as CenterPoint and BEVDet. Directly after `model.inference()`
it reads the MLA's BF16 head maps where the MLA wrote them, so the model's post stage does
not run.

This graph-construction fragment belongs inside your application. The values are
CenterPoint's nuScenes configuration.

```cpp
#include <neat.h>

using namespace simaai::neat;

Model model("centerpoint_mpk.tar.gz");

const auto circle = [](int classes, float radius) {
  SimaCenterHeadTask task;
  task.classes = classes;
  task.nms = SimaCenterHeadNms::Circle;
  task.nms_radius = radius;
  return task;
};
SimaCenterHeadDecodeOptions decode;
decode.tasks = {circle(1, 4.0f), circle(2, 12.0f), circle(2, 10.0f),
                circle(1, 1.0f), circle(2, 0.85f), circle(2, 0.175f)};
decode.out_size_factor = 4;
decode.voxel_size = {0.2f, 0.2f};
decode.range_min = {-51.2f, -51.2f};
decode.score_threshold = 0.1f;
decode.top_k = 500;
decode.post_max_per_task = 83;
decode.post_center_range = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};

Graph graph;
graph.add(nodes::Input());
graph.add(model.inference());
graph.add(nodes::SimaCenterHeadDecode(decode));
graph.add(nodes::Output());
```

Each pulled Sample holds `boxes`, `scores`, `labels` and `count`, in this order. Read the
first `count` rows. `model.inference()` starts at the MLA, so the graph's input must already
be in the format the MLA reads. The node's parallel work shares the graph's A65 workers
(`GraphOptions.kernel_runner_a65_workers`).

### Head maps

Port `i` is the `i`-th upstream tensor. Every port is an FP32 or BF16 channels-last map
`[H,W,C]` or `[1,H,W,C]`, and all ports have the same H and W.

- The heatmap has a port of its own, with one channel per class, head task after head
  task.
- Every other port holds one or more box heads, head task after head task. Each head
  task's heads come in this order: reg (2 channels: the x and y offsets), height (1), dim
  (3: the logarithms of dx, dy and dz), rot (2: sine and cosine) and, with `velocity`,
  vel (2).
- The defaults fit a model with three outputs: the heatmap; reg, height, dim and vel; and
  rot. For CenterPoint on nuScenes, with six head tasks and ten classes, they have 10, 48
  and 12 channels. A model with one output per head sets `heatmap_input` to `vel_input`
  to its own order.

A model with one output per head task and head has more than six outputs. Concatenate
them per head when you export the model.

The maps are dense, or BF16 with gaps after cells and rows. After `model.inference()`,
the MLA of an SDK 3.0 package publishes its heads that way, BF16 padded to 16 channels
per cell, and the node reads them where they are. The boxes do not depend on how the
heads are stored.

A package whose MLA output is quantized or tiled publishes integers after
`model.inference()`, and the graph build fails with the node's reason. Use
`model.graph()` there instead of `model.inference()`. It adds the model's own
conversions around the MLA, so the graph takes the model's input and the node gets FP32
head maps.

### Options

The options are the decode settings of the model's mmdetection3d configuration.

| Option | mmdetection3d configuration |
| --- | --- |
| `tasks[i].classes` | Number of `class_names` of head task `i` |
| `tasks[i].nms`, `tasks[i].nms_radius` | `test_cfg.nms_type = 'circle'`, `test_cfg.min_radius[i]` |
| `tasks[i].nms`, `tasks[i].nms_iou_threshold` | `test_cfg.nms_type = 'rotate'`, `test_cfg.nms_thr` |
| `nms_size_scale` | `test_cfg.nms_rescale_factor`, as one value per class |
| `out_size_factor`, `voxel_size` | The same keys of `bbox_coder` |
| `range_min` | x and y minimum of `bbox_coder.pc_range` |
| `score_threshold`, `top_k`, `post_center_range` | `score_threshold`, `max_num` and `post_center_range` of `bbox_coder` |
| `post_max_per_task` | `test_cfg.post_max_size` |
| `velocity` | The head has a `vel` branch |

Every head task states its NMS. Circle NMS compares the squared distance of two box
centres with `nms_radius` itself, as mmdetection3d does. Rotate NMS compares the
bird's-eye-view IoU of two rotated boxes with `nms_iou_threshold`. When set,
`nms_size_scale` has one value per heatmap channel, and 1 for the classes of circle head
tasks. `labels` are heatmap channels, so the first class of the second head task follows
the last class of the first.

In Python, pass a head task's values to its constructor and assign whole lists. A list
read from an options object is a copy, so appending to `decode.tasks` changes nothing.

```python
import pyneat

circle = pyneat.SimaCenterHeadNms.Circle
decode = pyneat.SimaCenterHeadDecodeOptions()
decode.tasks = [
    pyneat.SimaCenterHeadTask(classes, circle, nms_radius=radius)
    for classes, radius in [(1, 4.0), (2, 12.0), (2, 10.0), (1, 1.0), (2, 0.85), (2, 0.175)]
]
decode.out_size_factor = 4
decode.voxel_size = [0.2, 0.2]
decode.range_min = [-51.2, -51.2]
decode.score_threshold = 0.1
decode.top_k = 500
decode.post_max_per_task = 83
decode.post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
node = pyneat.nodes.sima_centerhead_decode(decode)
```

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
- `SimaBEVPoolPlan` and `SimaCenterHeadDecode` run on A65 only.

A request that one target does not support fails on that target. The node does
not choose another target for you.
