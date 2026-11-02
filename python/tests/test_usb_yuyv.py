"""Public API checks with no camera or accelerator access."""
import pytest
import pyneat as neat


def test_yuyv_admission_precedes_device_access():
    options = neat.UsbCameraInputOptions()
    options.camera_name = "fixture-not-connected"
    options.width = 641
    with pytest.raises(ValueError, match="even width"):
        neat.usb_camera_info(options, neat.UsbCameraFormat.YUYV)
    options.width = 640
    options.framerate_num = 2**32 - 1
    with pytest.raises(ValueError, match="signed caps"):
        neat.nodes.usb_camera_input(options, neat.UsbCameraFormat.YUYV)


def test_explicit_encoder_policy_and_input_format():
    assert neat.PreprocessColorFormat.YUYV != neat.PreprocessColorFormat.NV16
    options = neat.SimaEncodeOptions()
    strict = neat.nodes.sima_encode(options, neat.EncoderInputPolicy.RequireDirect)
    assert strict is not None
    assert neat.nodes.sima_encode(options) is not None

    options.type = neat.SimaEncodeType.H265
    with pytest.raises(ValueError, match="H.264"):
        neat.nodes.sima_encode(options, neat.EncoderInputPolicy.RequireDirect)
