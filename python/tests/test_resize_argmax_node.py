"""ResizeArgMax and ModelOptions.semantic_segmentation bindings.

Node factories are build-time descriptions: these check option round trips,
defaults, construction and structural validation without a runtime. Numerical
results are covered by the C++ resize_argmax_ev_test suite on a DevKit.
"""

from __future__ import annotations

import pytest
import pyneat


def test_resize_argmax_options_and_node():
  opt = pyneat.ResizeArgMaxOptions()
  assert (opt.output_width, opt.output_height, opt.align_corners) == (0, 0, False)
  assert not hasattr(opt, "target")
  opt.output_width = 1280
  opt.output_height = 720
  opt.align_corners = True
  assert (opt.output_width, opt.output_height, opt.align_corners) == (1280, 720, True)
  node = pyneat.nodes.resize_argmax(opt)
  assert isinstance(node, pyneat.Node)
  assert node.kind() == "ResizeArgMax"


@pytest.mark.parametrize("width,height", [(0, 0), (0, 720), (1280, -1)])
def test_resize_argmax_rejects_empty_class_map(width, height):
  opt = pyneat.ResizeArgMaxOptions()
  opt.output_width = width
  opt.output_height = height
  with pytest.raises(ValueError, match="output_width and output_height must be positive"):
    pyneat.nodes.resize_argmax(opt)


def test_model_semantic_segmentation_options():
  options = pyneat.ModelOptions()
  seg = options.semantic_segmentation
  assert isinstance(seg, pyneat.ModelSemanticSegmentationOptions)
  assert (seg.enable, seg.output_width, seg.output_height, seg.align_corners) == (
      False, 0, 0, False)
  assert not hasattr(seg, "target")
  seg = pyneat.ModelSemanticSegmentationOptions()
  seg.enable = True
  seg.output_width = 1920
  seg.output_height = 1080
  seg.align_corners = True
  options.semantic_segmentation = seg
  assert options.semantic_segmentation.enable is True
  seg = options.semantic_segmentation
  assert (seg.output_width, seg.output_height, seg.align_corners) == (1920, 1080, True)
