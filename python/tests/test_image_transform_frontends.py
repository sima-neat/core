"""Typed image frontends and fixed Preproc options use the shared prepared path."""
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
        assert projected.byte_size == 16 * height


@pytest.mark.parametrize("group", ["http_mjpeg", "rtsp"])
def test_decoded_group_byte_estimates_follow_spatial_overrides(group):
    if group == "http_mjpeg":
        options = pyneat.HttpMjpegDecodedInputOptions()
        project = pyneat.groups.http_mjpeg_decoded_output_spec
    else:
        options = pyneat.RtspDecodedInputOptions()
        project = pyneat.groups.rtsp_decoded_output_spec
    options.dec_width, options.dec_height = 8, 6
    options.source_fps = 30
    assert project(options).byte_size == 72
    options.output_caps.enable = True
    for format_, width, height, expected in (
        (pyneat.Format.NV16, -1, -1, 96),
        (pyneat.Format.Auto, 16, -1, 144),
        (pyneat.Format.Auto, -1, 12, 144),
        (pyneat.Format.RGB, 8, 6, 144),
        (pyneat.Format.BGR, 8, 6, 144),
        (pyneat.Format.GRAY8, 8, 6, 48),
        (pyneat.Format.I420, 8, 6, 72),
        (pyneat.Format.NV16, -1, 5, 80),
        (pyneat.Format.Auto, -1, 5, 0),
        (pyneat.Format.YUYV, 8, 6, 0),
    ):
        options.output_caps.format = format_
        options.output_caps.width, options.output_caps.height = width, height
        assert project(options).byte_size == expected
    for explicit_spatial_caps in (False, True):
        options.output_caps.format = pyneat.Format.NV12 if explicit_spatial_caps else pyneat.Format.Auto
        options.output_caps.width = 8 if explicit_spatial_caps else -1
        options.output_caps.height = 6 if explicit_spatial_caps else -1
        options.output_caps.fps = 24
        options.output_caps.memory = pyneat.CapsMemory.SystemMemory
        projected = project(options)
        assert (projected.byte_size, projected.width, projected.height) == (72, 8, 6)
        assert (projected.fps_num, projected.memory) == (24, "SystemMemory")
    options.out_format = pyneat.Format.YUYV
    options.decoder_raw_output = False
    options.output_caps.format = pyneat.Format.Auto
    assert project(options).byte_size == 0


def test_fixed_preproc_options_are_additive_and_round_trip():
    options = pyneat.ModelOptions()
    p = options.preprocess
    assert p.rescale_factor is None and p.quantization_step is None
    assert p.floating_conversion == pyneat.PreprocFloatingConversion.Legacy
    assert p.color_conversion_placement == pyneat.PreprocColorConversionPlacement.AfterResize
    p.rescale_factor = 1.0 / 255.0
    p.resize.interpolation = pyneat.ResizeInterpolation.BilinearAntialiased
    p.normalize.enable = pyneat.AutoFlag.On
    p.normalize.mean = [0.485, 0.456, 0.406]
    p.normalize.stddev = [0.229, 0.224, 0.225]
    p.floating_conversion = pyneat.PreprocFloatingConversion.NearestEven
    assert options.preprocess.rescale_factor == pytest.approx(1.0 / 255.0)
    assert options.preprocess.floating_conversion == pyneat.PreprocFloatingConversion.NearestEven
    standalone = pyneat.PreprocOptions()
    standalone.rescale_factor = 1.0
    standalone.quantization_step = 0.25
    standalone.color_conversion_placement = pyneat.PreprocColorConversionPlacement.BeforeResize
    assert standalone.quantization_step == 0.25
    standalone.rescale_factor = None
    standalone.quantization_step = None
    assert standalone.rescale_factor is None and standalone.quantization_step is None
    p.rescale_factor = None
    p.quantization_step = 0.25
    p.quantization_step = None
    assert p.rescale_factor is None and p.quantization_step is None
    norm = pyneat.NormalizationOptions()
    norm.rescale_factor = 1.0 / 255.0
    norm.floating_conversion = pyneat.PreprocFloatingConversion.NearestEven
    norm.stddev = [1.0, -0.25, 1.0]
    assert pyneat.nodes.normalization(norm) is not None
    norm.rescale_factor = None
    assert norm.rescale_factor is None
    resize = pyneat.ResizeOptions()
    resize.width = resize.height = 8
    resize.interpolation = pyneat.ResizeInterpolation.BilinearAntialiased
    assert pyneat.nodes.resize(resize) is not None


@pytest.mark.parametrize("name", ["Nearest", "Bilinear", "Bicubic", "BoxAverage", "BilinearAntialiased"])
def test_model_and_standalone_resize_share_interpolation_enum(name):
    model = pyneat.ModelOptions()
    assert model.preprocess.resize.interpolation is None
    assert model.preprocess.resize.scaling_type == "BILINEAR"
    sampler = getattr(pyneat.ResizeInterpolation, name)
    model.preprocess.resize.interpolation = sampler
    standalone = pyneat.ResizeOptions()
    standalone.interpolation = sampler
    assert model.preprocess.resize.interpolation == standalone.interpolation
    model.preprocess.resize.interpolation = None
    assert model.preprocess.resize.interpolation is None
    model.preprocess.resize.scaling_type = "NEAREST_NEIGHBOUR"
    assert model.preprocess.resize.scaling_type == "NEAREST_NEIGHBOUR"
    with pytest.raises(TypeError):
        model.preprocess.resize.interpolation = "BILINEAR_ANTIALIASED"


def test_center_crop_typed_options_and_optional_reset():
    model = pyneat.ModelOptions()
    assert model.preprocess.resize.shorter_side is None
    model.preprocess.resize.mode = pyneat.ResizeMode.Crop
    model.preprocess.resize.shorter_side = 256
    assert model.preprocess.resize.shorter_side == 256
    model.preprocess.resize.shorter_side = None
    assert model.preprocess.resize.shorter_side is None
    raw = pyneat.PreprocOptions()
    assert raw.resize_mode is None and raw.shorter_side is None
    raw.resize_mode = pyneat.ResizeMode.Crop
    raw.shorter_side = 256
    assert raw.resize_mode == pyneat.ResizeMode.Crop and raw.shorter_side == 256
    raw.resize_mode = None
    raw.shorter_side = None
    assert raw.resize_mode is None and raw.shorter_side is None
    standalone = pyneat.ResizeOptions()
    standalone.width = standalone.height = 224
    standalone.fit = pyneat.ResizeFit.Cover
    standalone.shorter_side = 256
    assert pyneat.nodes.resize(standalone) is not None
    standalone.shorter_side = None
    assert pyneat.nodes.resize(standalone) is not None
    standalone.shorter_side = 256
    standalone.fit = pyneat.ResizeFit.Contain
    with pytest.raises(Exception, match="shorter_side"):
        pyneat.nodes.resize(standalone)
    with pytest.raises(TypeError):
        model.preprocess.resize.shorter_side = "256"
