"""Typed image frontends share Preproc without widening its existing options ABI."""
import pytest
import pyneat


def test_image_frontend_options_and_factories():
    resize = pyneat.ResizeOptions()
    resize.width, resize.height = 320, 240
    resize.max_input_width, resize.max_input_height = 1920, 1080
    resize.fit = pyneat.ResizeFit.Contain
    resize.yuv_padding = [16, 128, 128]
    assert pyneat.nodes.resize(resize) is not None
    convert = pyneat.ColorConvertOptions()
    convert.output_format = pyneat.ImageFormat.NV12
    convert.surface = pyneat.ImageSurfaceLayout.EncoderNv12
    convert.color.range = pyneat.ImageYuvRange.Limited
    assert pyneat.nodes.color_convert(convert) is not None
    norm = pyneat.NormalizationOptions()
    norm.mean = [0.5, 0.25, 0.125]
    norm.stddev = [0.5, 0.25, 1.0]
    assert pyneat.nodes.normalization(norm) is not None
    assert pyneat.advanced.format_tag_name(pyneat.Format.NV16) == "NV16"
    assert pyneat.advanced.is_raw_video_format(pyneat.Format.NV16)


def test_invalid_image_frontend_options_fail_before_preparation():
    with pytest.raises(Exception, match="positive output"):
        pyneat.nodes.resize(pyneat.ResizeOptions())
    norm = pyneat.NormalizationOptions()
    norm.stddev = [1.0, 0.0, 1.0]
    with pytest.raises(Exception, match="standard deviations"):
        pyneat.nodes.normalization(norm)
