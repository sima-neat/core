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


def test_explicit_color_overloads():
    color = neat.ImageColorOptions()
    color.matrix = neat.ImageColorMatrix.Bt709
    color.range = neat.ImageYuvRange.Limited
    options = neat.PreprocOptions()
    options.input_img_type = "YUYV"
    assert neat.nodes.preproc(options, color) is not None
    options.input_img_type = "RGB"
    with pytest.raises(ValueError, match="requires YUYV input"):
        neat.nodes.preproc(options, color)

    convert = neat.ColorConvertOptions()
    convert.output_format = neat.ImageFormat.NV12
    convert.color = color
    assert neat.nodes.color_convert(convert, "bt709") is not None
    with pytest.raises(ValueError, match="explicit source colorimetry"):
        neat.nodes.color_convert(convert, "")


def test_strict_video_sender_overload():
    raw = neat.VideoSenderOptions.from_raw(neat.SimaEncodeOptions())
    assert neat.groups.video_sender(raw, neat.EncoderInputPolicy.RequireDirect) is not None
    passthrough = neat.VideoSenderOptions.passthrough(neat.RtspCodec.MJPEG)
    with pytest.raises(ValueError, match="H.264 only"):
        neat.groups.video_sender(passthrough, neat.EncoderInputPolicy.RequireDirect)


def test_capture_info_fields():
    fields = {"camera", "options", "format", "row_stride", "storage_bytes", "color", "colorimetry"}
    assert fields <= set(dir(neat.UsbCameraCaptureInfo))
