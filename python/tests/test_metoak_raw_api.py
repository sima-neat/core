"""Public raw-camera graph construction: no devices or EV execution."""
import math

import pytest
import pyneat as n


def depth_options():
    options = n.MetoakDepthOptions()
    options.width = 640
    options.height = 360
    return options


def test_planar_and_raw_factories_are_distinct_and_additive():
    options = depth_options()
    assert isinstance(n.nodes.metoak_depth(options), n.Node)
    raw = n.MetoakRawInputOptions()
    assert math.isnan(raw.cx) and math.isnan(raw.cy)
    with pytest.raises((ValueError, RuntimeError), match="calibration"):
        n.nodes.metoak_depth(options, raw)
    raw.cx, raw.cy = 319.5, 179.5
    assert isinstance(n.nodes.metoak_depth(options, raw), n.Node)
    options.width = 638
    with pytest.raises((ValueError, RuntimeError), match="640x360"):
        n.nodes.metoak_depth(options, raw)


def test_raw_capture_requires_explicit_transport_not_bayer_image_conversion():
    options = n.CameraInputOptions()
    options.width, options.height = 1920, 360
    options.format = "RAW8"
    options.buffer_name = "raw_src"
    backend = n.CameraV4L2Options()
    backend.device = "/dev/video-test-not-opened"
    backend.fourcc = "BA81"
    assert isinstance(n.nodes.camera_input(options, backend_options=backend), n.Node)
    options.format = "RGB"
    with pytest.raises((ValueError, RuntimeError)):
        n.nodes.camera_input(options, backend_options=backend)
