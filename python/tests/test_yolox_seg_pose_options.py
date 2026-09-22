"""Decoder-specific configuration follows the existing SuperPoint option pattern."""
import pyneat
import pytest


@pytest.mark.parametrize("make_options", [
    pyneat.ModelOptions,
    lambda: pyneat.BoxDecodeOptions(pyneat.BoxDecodeType.YoloXSegPose),
])
def test_yolox_seg_pose_nested_options(make_options):
    options = make_options()
    assert options.yolox_seg_pose.pose_classes == []
    options.yolox_seg_pose.pose_classes = [0, 5]
    assert options.yolox_seg_pose.pose_classes == [0, 5]
    assert not hasattr(options, "pose_classes")
    replacement = pyneat.YoloXSegPoseOptions()
    replacement.pose_classes = [2]
    options.yolox_seg_pose = replacement
    assert options.yolox_seg_pose.pose_classes == [2]
