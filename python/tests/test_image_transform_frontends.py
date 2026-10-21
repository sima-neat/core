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


@pytest.mark.parametrize("height", [5, 6])
def test_nv16_group_output_specs_are_planar(height):
    # Public Python metadata forwards to the same C++ producers. This does not
    # request native NV16 decoding or add a Model preprocessing format.
    image = pyneat.ImageInputGroupOptions()
    image.output_caps.enable = True
    image.output_caps.format = pyneat.Format.NV16
    image.output_caps.width, image.output_caps.height = 8, height
    spec = pyneat.groups.image_input_output_spec(image)
    assert (spec.format, spec.layout, spec.dtype) == ("NV16", "Planar", "UInt8")
    assert spec.byte_size == 16 * height

    for options, project in (
        (pyneat.VideoInputGroupOptions(), pyneat.groups.video_input_output_spec),
        (pyneat.HttpMjpegDecodedInputOptions(), pyneat.groups.http_mjpeg_decoded_output_spec),
        (pyneat.RtspDecodedInputOptions(), pyneat.groups.rtsp_decoded_output_spec),
    ):
        options.output_caps.enable = True
        options.output_caps.format = pyneat.Format.NV16
        options.output_caps.width, options.output_caps.height = 8, height
        projected = project(options)
        assert (projected.format, projected.layout, projected.dtype) == ("NV16", "Planar", "UInt8")
        assert (projected.width, projected.height) == (8, height)
