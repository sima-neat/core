"""SimaVoxelize, SimaPillarScatter, SimaBEVPoolPlan, SimaBEVPool and SimaCenterHeadDecode bindings.

Node factories are build-time descriptions: these check option round trips,
defaults, construction and structural validation without a runtime. Numerical
results are covered by the C++ sima_*_ev_test and sima_*_a65_test suites on a DevKit.
"""

from __future__ import annotations

import pytest
import pyneat


def test_factories_present():
  for name in ("sima_voxelize", "sima_pillar_scatter", "sima_bev_pool_plan", "sima_bev_pool",
               "sima_centerhead_decode"):
    assert hasattr(pyneat.nodes, name), name


def test_shared_kernel_types():
  assert pyneat.SimaKernelTarget.A65 != pyneat.SimaKernelTarget.EV74
  q = pyneat.SimaKernelQuantization()
  assert q.multiplier == 1.0
  assert q.zero_point == 0
  assert q.rounding == pyneat.SimaQuantRounding.NearestEven
  q.multiplier = 0.25
  q.zero_point = -3
  q.rounding = pyneat.SimaQuantRounding.TowardNegative
  assert (q.multiplier, q.zero_point, q.rounding) == (
      0.25, -3, pyneat.SimaQuantRounding.TowardNegative)


def test_voxelize_options_and_node():
  opt = pyneat.SimaVoxelizeOptions()
  assert opt.target == pyneat.SimaKernelTarget.A65
  assert opt.max_pillars == 0
  assert opt.output_dtype == pyneat.TensorDType.Int8
  opt.voxel_size = [0.25, 0.25, 8.0]
  opt.point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
  q = pyneat.SimaKernelQuantization()
  q.multiplier = 2.5
  q.zero_point = 8
  opt.quantization = q
  opt.max_pillars = 40000
  opt.max_points_per_pillar = 64
  opt.target = pyneat.SimaKernelTarget.EV74
  assert list(opt.voxel_size) == [0.25, 0.25, 8.0]
  assert list(opt.point_cloud_range)[3] == 50.0
  assert opt.quantization.zero_point == 8
  opt.output_dtype = pyneat.TensorDType.BFloat16
  assert opt.output_dtype == pyneat.TensorDType.BFloat16
  node = pyneat.nodes.sima_voxelize(opt)
  assert isinstance(node, pyneat.Node)
  assert node.kind() == "SimaVoxelize"
  with pytest.raises(ValueError, match="max_pillars"):
    pyneat.nodes.sima_voxelize(pyneat.SimaVoxelizeOptions())
  opt.output_dtype = pyneat.TensorDType.Float32
  with pytest.raises(ValueError, match="output_dtype must be Int8 or BFloat16"):
    pyneat.nodes.sima_voxelize(opt)


def test_pillar_scatter_options_and_node():
  opt = pyneat.SimaPillarScatterOptions()
  assert (opt.rows, opt.columns, opt.zero_point) == (0, 0, 0)
  opt.rows = 400
  opt.columns = 400
  opt.zero_point = -128
  node = pyneat.nodes.sima_pillar_scatter(opt)
  assert node.kind() == "SimaPillarScatter"
  with pytest.raises(ValueError, match="rows and columns"):
    pyneat.nodes.sima_pillar_scatter(pyneat.SimaPillarScatterOptions())


def test_bev_pool_plan_options_and_node():
  opt = pyneat.SimaBEVPoolPlanOptions()
  assert (opt.format, opt.workers, opt.tile_h, opt.tile_w) == (
      pyneat.SimaBEVPoolPlanFormat.V1, 4, 8, 8)
  assert opt.cache_plan is False
  assert opt.plan_capacity_bytes == 0
  assert opt.input_dtype == pyneat.TensorDType.Int8
  opt.xs = [0.0, 8.0, 16.0]
  opt.ys = [0.0, 8.0]
  opt.ds = [1.0, 1.5]
  opt.bx = [-51.0, -51.0, 0.0]
  opt.dx = [0.4, 0.4, 20.0]
  opt.nx = [256, 256, 1]
  opt.format = pyneat.SimaBEVPoolPlanFormat.V2
  opt.context_channels = 80
  opt.cache_plan = True
  assert list(opt.nx) == [256, 256, 1]
  opt.input_dtype = pyneat.TensorDType.BFloat16
  assert opt.input_dtype == pyneat.TensorDType.BFloat16
  assert opt.format == pyneat.SimaBEVPoolPlanFormat.V2
  assert pyneat.nodes.sima_bev_pool_plan(opt).kind() == "SimaBEVPoolPlan"


def test_bev_pool_options_and_node():
  opt = pyneat.SimaBEVPoolOptions()
  assert opt.input_storage == pyneat.SimaBEVPoolStorage.ChannelsLast
  assert opt.output_storage == pyneat.SimaBEVPoolStorage.ChannelsLast
  assert opt.output_dtype == pyneat.TensorDType.Int8
  assert opt.plan_format == pyneat.SimaBEVPoolPlanFormat.V1
  opt.cells = [256, 256, 1]
  opt.depth_zero_point = -128
  opt.context_zero_point = -5
  opt.plan_format = pyneat.SimaBEVPoolPlanFormat.V2
  opt.output_storage = pyneat.SimaBEVPoolStorage.Contiguous
  assert list(opt.cells) == [256, 256, 1]
  assert pyneat.nodes.sima_bev_pool(opt).kind() == "SimaBEVPool"
  opt.output_dtype = pyneat.TensorDType.Float32
  with pytest.raises(ValueError, match="output_dtype"):
    pyneat.nodes.sima_bev_pool(opt)


def test_centerhead_decode_options_and_node():
  opt = pyneat.SimaCenterHeadDecodeOptions()
  assert opt.target == pyneat.SimaKernelTarget.A65
  assert opt.velocity is True
  assert (opt.heatmap_input, opt.reg_input, opt.height_input, opt.dim_input, opt.rot_input,
          opt.vel_input) == (0, 1, 1, 1, 2, 1)
  task = pyneat.SimaCenterHeadTask()
  assert task.nms == pyneat.SimaCenterHeadNms.Unspecified
  task.classes = 2
  opt.tasks = [task]
  opt.out_size_factor = 4
  opt.top_k = 500
  opt.post_max_per_task = 83
  with pytest.raises(ValueError, match="state its NMS"):
    pyneat.nodes.sima_centerhead_decode(opt)

  circle = pyneat.SimaCenterHeadTask(2, pyneat.SimaCenterHeadNms.Circle, nms_radius=0.85)
  assert (circle.classes, circle.nms, circle.nms_iou_threshold) == (
      2, pyneat.SimaCenterHeadNms.Circle, 0.0)
  assert circle.nms_radius == pytest.approx(0.85)
  rotate = pyneat.SimaCenterHeadTask(1, pyneat.SimaCenterHeadNms.Rotate, nms_iou_threshold=0.2)
  assert rotate.nms == pyneat.SimaCenterHeadNms.Rotate and rotate.nms_radius == 0.0
  assert rotate.nms_iou_threshold == pytest.approx(0.2)
  opt.tasks = [circle]
  opt.voxel_size = [0.2, 0.2]
  opt.range_min = [-51.2, -51.2]
  opt.score_threshold = 0.1
  opt.post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
  opt.nms_size_scale = [1.0, 1.0]
  assert opt.tasks[0].classes == 2
  assert list(opt.post_center_range)[3] == pytest.approx(61.2)
  assert list(opt.nms_size_scale) == [1.0, 1.0]
  assert pyneat.nodes.sima_centerhead_decode(opt).kind() == "SimaCenterHeadDecode"
