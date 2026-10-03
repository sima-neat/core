"""Unified CameraInput construction checks; never open cameras or dispatch firmware."""
import sys

import pytest
import pyneat as neat


def options():
    camera = neat.CameraInputOptions()
    camera.profile = neat.CameraProfile.MetoakSimor
    camera.device = "/dev/video1"
    return camera


def test_only_unified_camera_api_is_public():
    camera = options()
    assert not hasattr(neat, "CameraV4L2Options")
    assert not hasattr(neat.nodes, "CameraInputWithV4L2")
    assert not hasattr(camera, "fourcc")
    assert camera.zero_copy is None
    assert camera.output_buffer_count == 8
    assert camera.frame_timeout_ms == 2000
    with pytest.raises(TypeError):
        neat.nodes.camera_input(camera, backend_options=object())
    assert neat.nodes.camera_input(neat.CameraInputOptions()) is not None
    if sys.platform == "linux":
        assert neat.nodes.camera_input(camera) is not None
        camera.zero_copy = False
        assert neat.nodes.camera_input(camera) is not None
        camera.zero_copy = None
        assert neat.nodes.camera_input(camera, capture_buffer_count=12) is not None
    else:
        with pytest.raises(RuntimeError, match="requires Linux"):
            neat.nodes.camera_input(camera)


@pytest.mark.parametrize("field,value", [
    ("zero_copy", True), ("format", "RGB"), ("width", 640), ("height", 480),
    ("capture_buffer_count", 3), ("capture_buffer_count", 129),
    ("output_buffer_count", 1), ("output_buffer_count", 129),
    ("frame_timeout_ms", 0), ("frame_timeout_ms", 60001),
    ("queue_depth", 0), ("queue_depth", 8), ("buffer_name", ""),
    ("camera_name", "wrong-source"), ("allow_cpu_fallback", True),
])
def test_invalid_options_fail_before_discovery(field, value):
    camera = options()
    camera.device = ""  # Invalid configuration must not trigger a metadata scan.
    setattr(camera, field, value)
    if sys.platform == "linux":
        with pytest.raises(ValueError):
            neat.nodes.camera_input(camera)
    else:
        with pytest.raises(RuntimeError, match="requires Linux"):
            neat.nodes.camera_input(camera)


def test_generic_raw_format_does_not_select_metoak():
    camera = neat.CameraInputOptions()
    camera.format = "RAW8"
    camera.zero_copy = False
    assert camera.profile == neat.CameraProfile.Default
    assert neat.nodes.camera_input(camera) is not None


def test_libcamera_copy_policy_is_preserved():
    config = neat.CameraInputOptions()
    assert config.zero_copy is None
    assert config.allow_cpu_fallback is False
    config.zero_copy = False
    assert neat.nodes.camera_input(config) is not None
    config.zero_copy = True
    assert neat.nodes.camera_input(config) is not None
    config.allow_cpu_fallback = True
    with pytest.raises(ValueError, match="conflicts"):
        neat.nodes.camera_input(config)


def test_default_profile_does_not_infer_metoak_from_device():
    config = neat.CameraInputOptions()
    assert config.profile == neat.CameraProfile.Default
    assert not hasattr(config, "backend")
    assert not hasattr(neat, "CameraBackend")
    assert not hasattr(neat.CameraProfile, "Raw")
    config.device = "/dev/not-a-camera-must-not-be-opened"
    with pytest.raises(ValueError, match="MetoakSimor profile"):
        neat.nodes.camera_input(config)
