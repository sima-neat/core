"""Construction-only checks; never open a camera or dispatch firmware."""
import sys

import pytest
import pyneat as neat


def options():
    camera = neat.CameraInputOptions()
    camera.width, camera.height, camera.format = 1920, 360, "RAW8"
    backend = neat.CameraV4L2Options()
    backend.device, backend.fourcc = "/dev/video1", "BA81"
    return camera, backend


def test_owned_camera_and_legacy_api():
    camera, backend = options()
    assert backend.zero_copy is False
    assert backend.output_buffer_count == 8
    assert backend.frame_timeout_ms == 2000
    assert neat.nodes.camera_input(neat.CameraInputOptions()) is not None
    if sys.platform == "linux":
        assert neat.nodes.camera_input(camera, backend) is not None
    else:
        with pytest.raises(RuntimeError, match="requires Linux"):
            neat.nodes.camera_input(camera, backend)


@pytest.mark.parametrize("field,value", [
    ("zero_copy", True), ("fourcc", "NV12"), ("device", ""),
    ("capture_buffer_count", 3), ("output_buffer_count", 1),
    ("frame_timeout_ms", 0),
])
def test_invalid_camera_options(field, value):
    camera, backend = options()
    setattr(backend, field, value)
    if sys.platform == "linux":
        with pytest.raises(ValueError):
            neat.nodes.camera_input(camera, backend)
    else:
        with pytest.raises(RuntimeError, match="requires Linux"):
            neat.nodes.camera_input(camera, backend)


def test_metoak_profile_and_generic_libcamera_options():
    camera, _ = options()
    # A raw format alone does not select the Metoak/V4L2 profile.
    camera.zero_copy = False
    assert camera.profile == neat.CameraProfile.Default
    assert neat.nodes.camera_input(camera) is not None
    profile = neat.CameraInputOptions()
    profile.device = "/dev/video1"
    profile.profile = neat.CameraProfile.MetoakSimor
    profile.zero_copy = False
    if sys.platform != "linux":
        with pytest.raises(RuntimeError, match="requires Linux"):
            neat.nodes.camera_input(profile)
        return
    assert neat.nodes.camera_input(profile) is not None
    profile.capture_buffer_count = 12
    assert neat.nodes.camera_input(profile) is not None
    profile.zero_copy = True
    with pytest.raises(ValueError, match="zero_copy=false"):
        neat.nodes.camera_input(profile)
    profile.zero_copy = None
    with pytest.raises(ValueError, match="zero_copy=false"):
        neat.nodes.camera_input(profile)


def test_legacy_zero_copy_policy():
    config = neat.CameraInputOptions()
    assert config.zero_copy is None
    config.zero_copy = False
    assert neat.nodes.camera_input(config) is not None
    config.zero_copy = True
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
