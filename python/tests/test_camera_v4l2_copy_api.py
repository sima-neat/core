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
