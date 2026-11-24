"""Public API checks with no camera or accelerator access."""
import pytest
import pyneat as neat


def test_yuyv_admission_precedes_device_access():
    options = neat.UsbCameraInputOptions()
    options.camera_name = "fixture-not-connected"
    options.format = neat.UsbCameraFormat.YUYV
    options.width = 641
    with pytest.raises(ValueError, match="even width"):
        neat.usb_camera_info(options)
    options.width = 640
    options.framerate_num = 2**32 - 1
    with pytest.raises(ValueError, match="signed caps"):
        neat.nodes.usb_camera_input(options)


def test_explicit_encoder_policy_and_input_format():
    assert neat.PreprocessColorFormat.YUYV != neat.PreprocessColorFormat.NV16
    options = neat.SimaEncodeOptions()
    assert options.input_policy == neat.EncoderInputPolicy.Compatible
    options.input_policy = neat.EncoderInputPolicy.RequireDirect
    strict = neat.nodes.sima_encode(options)
    assert strict is not None
    assert neat.nodes.sima_encode(neat.SimaEncodeOptions()) is not None

    options.type = neat.SimaEncodeType.H265
    with pytest.raises(ValueError, match="H.264"):
        neat.nodes.sima_encode(options)


def test_explicit_color_options():
    color = neat.ImageColorOptions()
    color.matrix = neat.ImageColorMatrix.Bt709
    color.range = neat.ImageYuvRange.Limited
    options = neat.PreprocOptions()
    assert options.input_color is None
    options.input_img_type = "YUYV"
    options.input_color = color
    assert neat.nodes.preproc(options) is not None
    options.input_img_type = "RGB"
    with pytest.raises(ValueError, match="requires YUYV input"):
        neat.nodes.preproc(options)

    convert = neat.ColorConvertOptions()
    convert.output_format = neat.ImageFormat.NV12
    convert.color = color
    convert.input_colorimetry = "bt709"
    assert neat.nodes.color_convert(convert) is not None
    convert.input_colorimetry = ""
    assert neat.nodes.color_convert(convert) is not None


def test_strict_video_sender_options():
    encode = neat.SimaEncodeOptions()
    encode.input_policy = neat.EncoderInputPolicy.RequireDirect
    raw = neat.VideoSenderOptions.from_raw(encode)
    assert neat.groups.video_sender(raw) is not None
    encode.type = neat.SimaEncodeType.MJPEG
    mjpeg = neat.VideoSenderOptions.from_raw(encode)
    with pytest.raises(ValueError, match="H.264 only"):
        neat.groups.video_sender(mjpeg)


def test_capture_info_fields():
    fields = {"format", "row_stride", "storage_bytes", "color", "colorimetry", "width", "height",
              "framerate_num", "framerate_den"}
    assert fields <= set(dir(neat.UsbCameraInfo))
    assert neat.UsbCameraInputOptions().format == neat.UsbCameraFormat.MJPEG
