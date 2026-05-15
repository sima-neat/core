
import numpy as np
import pytest

import model_fixture_helpers as model_fixtures
import pyneat

PREPROC_OPTION_FIELDS = (
    "input_shape",
    "output_shape",
    "slice_shape",
    "set_input_shape",
    "set_output_shape",
    "set_slice_shape",
    "has_input_shape",
    "has_output_shape",
    "has_slice_shape",
    "scaled_width",
    "scaled_height",
    "batch_size",
    "normalize",
    "aspect_ratio",
    "tessellate",
    "dynamic_input_dims",
    "input_offset",
    "input_stride",
    "output_stride",
    "q_zp",
    "q_scale",
    "channel_mean",
    "channel_stddev",
    "input_img_type",
    "output_img_type",
    "output_dtype",
    "scaling_type",
    "padding_type",
    "graph_name",
    "node_name",
    "element_name",
    "cpu",
    "next_cpu",
    "debug",
    "upstream_name",
    "graph_input_name",
    "num_buffers",
    "num_buffers_model",
    "num_buffers_locked",
)

QUANT_TESS_OPTION_FIELDS = (
    "config_path",
    "config_json",
    "element_name",
    "num_buffers",
    "num_buffers_model",
    "num_buffers_locked",
)

DETESS_DEQUANT_OPTION_FIELDS = (
    "config_path",
    "config_json",
    "upstream_name",
    "element_name",
    "num_buffers",
    "num_buffers_model",
    "num_buffers_locked",
)

UDP_OUTPUT_OPTION_FIELDS = (
    "host",
    "port",
    "sync",
    "async_",
)

H264_PARSE_OPTION_FIELDS = (
    "config_interval",
    "alignment",
    "stream_format",
    "enforce_caps",
)

UDP_H264_OUTPUT_GROUP_OPTION_FIELDS = (
    "h264_caps",
    "payload_type",
    "config_interval",
    "udp_host",
    "udp_port",
    "udp_sync",
    "udp_async",
)

VIDEO_SENDER_UDP_OPTION_FIELDS = (
    "host",
    "port",
    "sync",
    "async_",
)

VIDEO_SENDER_RTP_OPTION_FIELDS = (
    "payload_type",
    "config_interval",
)

VIDEO_SENDER_ENCODER_OPTION_FIELDS = (
    "bitrate_kbps",
    "profile",
    "level",
)

VIDEO_SENDER_OPTION_FIELDS = (
    "udp",
    "rtp",
    "encoder",
)

METADATA_SENDER_OPTION_FIELDS = (
    "host",
    "channel",
    "metadata_port_base",
)

UDP_OUTPUT_NODE_GROUP_OPTION_FIELDS = (
    "h264_caps",
    "payload_type",
    "config_interval",
    "enable_timings",
    "host",
    "video_port_base",
    "udp_sync",
    "udp_async",
)

METADATA_SENDER_GROUP_OPTION_FIELDS = (
    "host",
    "metadata_port_base",
)


def _strict_resnet50_mpk_path():
  return model_fixtures.strict_model_tar_path("SIMA_RESNET50_TAR")


def _strict_yolo_mpk_path():
  return model_fixtures.strict_model_tar_path("SIMA_YOLO_TAR")


def _assert_not_type_error(call):
  try:
    call()
  except Exception as exc:
    assert not isinstance(exc, TypeError), str(exc)


def test_session_pythonic_add_and_describe():
  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.nodes.output())

  text = session.describe_backend()
  assert isinstance(text, str)
  assert text


def test_model_option_structs_are_mutable():
  opt = pyneat.ModelOptions()
  opt.preprocess.input_max_width = 1920
  opt.preprocess.input_max_height = 1080
  opt.boxdecode_original_width = 1280
  opt.boxdecode_original_height = 720

  assert opt.preprocess.input_max_width == 1920
  assert opt.preprocess.input_max_height == 1080
  assert opt.boxdecode_original_width == 1280
  assert opt.boxdecode_original_height == 720


def test_input_stage_option_structs_expose_expected_fields():
  pre = pyneat.PreprocOptions()
  quant_tess = pyneat.QuantTessOptions()

  for field in PREPROC_OPTION_FIELDS:
    assert hasattr(pre, field), field
  for field in QUANT_TESS_OPTION_FIELDS:
    assert hasattr(quant_tess, field), field


def test_postprocess_stage_option_structs_expose_expected_fields():
  detess = pyneat.DetessDequantOptions()

  for field in DETESS_DEQUANT_OPTION_FIELDS:
    assert hasattr(detess, field), field


def test_output_stage_option_structs_expose_expected_fields():
  udp = pyneat.UdpOutputOptions()
  parse = pyneat.H264ParseOptions()
  group = pyneat.UdpH264OutputGroupOptions()
  video_raw = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(1920, 1080, 30)
  video_udp = pyneat.VideoSenderUdpOptions()
  video_rtp = pyneat.VideoSenderRtpOptions()
  video_encoder = pyneat.VideoSenderEncoderOptions()
  metadata_sender = pyneat.MetadataSenderOptions()
  udp_group = pyneat.UdpOutputNodeGroupOptions()
  metadata_group = pyneat.MetadataSenderGroupOptions()

  for field in UDP_OUTPUT_OPTION_FIELDS:
    assert hasattr(udp, field), field
  assert hasattr(udp, "async"), "async"

  for field in H264_PARSE_OPTION_FIELDS:
    assert hasattr(parse, field), field

  for field in UDP_H264_OUTPUT_GROUP_OPTION_FIELDS:
    assert hasattr(group, field), field

  for field in VIDEO_SENDER_OPTION_FIELDS:
    assert hasattr(video_raw, field), field

  for field in VIDEO_SENDER_UDP_OPTION_FIELDS:
    assert hasattr(video_udp, field), field

  for field in VIDEO_SENDER_RTP_OPTION_FIELDS:
    assert hasattr(video_rtp, field), field

  for field in VIDEO_SENDER_ENCODER_OPTION_FIELDS:
    assert hasattr(video_encoder, field), field

  for field in METADATA_SENDER_OPTION_FIELDS:
    assert hasattr(metadata_sender, field), field

  for field in UDP_OUTPUT_NODE_GROUP_OPTION_FIELDS:
    assert hasattr(udp_group, field), field

  for field in METADATA_SENDER_GROUP_OPTION_FIELDS:
    assert hasattr(metadata_group, field), field

  assert hasattr(pyneat, "H264ParseAlignment")
  assert hasattr(pyneat, "H264ParseStreamFormat")
  assert hasattr(pyneat, "VideoSenderOptions")
  assert hasattr(pyneat, "MetadataSender")
  assert hasattr(pyneat, "MetadataSenderGroup")


def test_input_stage_option_struct_constructors_accept_expected_args():
  mpk_path = _strict_resnet50_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))

  _assert_not_type_error(lambda: pyneat.PreprocOptions())
  _assert_not_type_error(lambda: pyneat.PreprocOptions(model))
  _assert_not_type_error(lambda: pyneat.QuantTessOptions())
  _assert_not_type_error(lambda: pyneat.QuantTessOptions(model))


def test_postprocess_stage_option_struct_constructors_accept_expected_args(tmp_path):
  mpk_path = _strict_yolo_mpk_path()
  model = pyneat.Model(str(mpk_path))

  _assert_not_type_error(lambda: pyneat.DetessDequantOptions())
  _assert_not_type_error(lambda: pyneat.DetessDequantOptions(model))


def test_output_stage_option_struct_constructors_accept_expected_args():
  _assert_not_type_error(lambda: pyneat.UdpOutputOptions())
  _assert_not_type_error(lambda: pyneat.H264ParseOptions())
  _assert_not_type_error(lambda: pyneat.UdpH264OutputGroupOptions())
  _assert_not_type_error(lambda: pyneat.VideoSenderUdpOptions())
  _assert_not_type_error(lambda: pyneat.VideoSenderRtpOptions())
  _assert_not_type_error(lambda: pyneat.VideoSenderEncoderOptions())
  _assert_not_type_error(lambda: pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(1920, 1080, 30))
  _assert_not_type_error(lambda: pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded())
  _assert_not_type_error(lambda: pyneat.MetadataSenderOptions())
  _assert_not_type_error(lambda: pyneat.UdpOutputNodeGroupOptions())
  _assert_not_type_error(lambda: pyneat.MetadataSenderGroupOptions())


def test_input_stage_option_structs_are_mutable():
  pre = pyneat.PreprocOptions()
  pre.input_shape = [240, 320, 3]
  pre.output_shape = [640, 640, 3]
  pre.slice_shape = [32, 128, 3]

  quant_tess = pyneat.QuantTessOptions()
  quant_tess.element_name = "qt"
  quant_tess.config_json = {"node_name": "quanttess"}

  assert pre.input_shape == [240, 320, 3]
  assert pre.output_shape == [640, 640, 3]
  assert pre.slice_shape == [32, 128, 3]
  assert pre.has_input_shape() is True
  assert pre.has_output_shape() is True
  assert pre.has_slice_shape() is True

  assert quant_tess.element_name == "qt"
  assert quant_tess.config_json["node_name"] == "quanttess"
  quant_tess.config_json = None
  assert quant_tess.config_json is None


def test_postprocess_stage_option_structs_are_mutable():
  detess = pyneat.DetessDequantOptions()
  detess.config_path = "/tmp/detess.json"
  detess.config_json = {"node_name": "detessdequant_0"}
  detess.upstream_name = "mla_0"
  detess.element_name = "detessdequant_0"
  detess.num_buffers = 4
  detess.num_buffers_model = 4
  detess.num_buffers_locked = True

  assert detess.config_path == "/tmp/detess.json"
  assert detess.config_json["node_name"] == "detessdequant_0"
  detess.config_json = None
  assert detess.config_json is None
  assert detess.upstream_name == "mla_0"
  assert detess.element_name == "detessdequant_0"
  assert detess.num_buffers == 4
  assert detess.num_buffers_model == 4
  assert detess.num_buffers_locked is True


def test_output_stage_option_structs_are_mutable():
  udp = pyneat.UdpOutputOptions()
  udp.host = "10.0.0.5"
  udp.port = 5500
  udp.sync = True
  udp.async_ = False

  parse = pyneat.H264ParseOptions()
  parse.config_interval = 2
  parse.alignment = pyneat.H264ParseAlignment.AU
  parse.stream_format = pyneat.H264ParseStreamFormat.ByteStream
  parse.enforce_caps = True

  group = pyneat.UdpH264OutputGroupOptions()
  group.h264_caps = 'video/x-h264,profile="high"'
  group.payload_type = 97
  group.config_interval = 2
  group.udp_host = "127.0.0.1"
  group.udp_port = 5600
  group.udp_sync = False
  group.udp_async = False

  video_raw = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(1920, 1080, 30)
  video_raw.udp.host = "127.0.0.1"
  video_raw.udp.port = 9000
  video_raw.udp.sync = True
  video_raw.udp.async_ = False
  video_raw.rtp.payload_type = 99
  video_raw.rtp.config_interval = 4
  video_raw.encoder.bitrate_kbps = 2500
  video_raw.encoder.profile = "main"
  video_raw.encoder.level = "4.1"

  video_encoded = pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()

  metadata_sender = pyneat.MetadataSenderOptions()
  metadata_sender.host = "127.0.0.1"
  metadata_sender.channel = 2
  metadata_sender.metadata_port_base = 9100

  udp_group = pyneat.UdpOutputNodeGroupOptions()
  udp_group.h264_caps = "video/x-h264"
  udp_group.payload_type = 98
  udp_group.config_interval = 3
  udp_group.enable_timings = True
  udp_group.host = "127.0.0.1"
  udp_group.video_port_base = 9200
  udp_group.udp_sync = False
  udp_group.udp_async = False

  metadata_group = pyneat.MetadataSenderGroupOptions()
  metadata_group.host = "127.0.0.1"
  metadata_group.metadata_port_base = 9300

  assert udp.host == "10.0.0.5"
  assert udp.port == 5500
  assert udp.sync is True
  assert udp.async_ is False
  assert getattr(udp, "async") is False

  assert parse.config_interval == 2
  assert parse.alignment == pyneat.H264ParseAlignment.AU
  assert parse.stream_format == pyneat.H264ParseStreamFormat.ByteStream
  assert parse.enforce_caps is True

  assert group.h264_caps == 'video/x-h264,profile="high"'
  assert group.payload_type == 97
  assert group.config_interval == 2
  assert group.udp_host == "127.0.0.1"
  assert group.udp_port == 5600
  assert group.udp_sync is False
  assert group.udp_async is False

  assert video_raw.is_raw_input() is True
  assert video_raw.is_encoded_input() is False
  assert video_raw.width == 1920
  assert video_raw.height == 1080
  assert video_raw.fps == 30
  assert video_raw.udp.host == "127.0.0.1"
  assert video_raw.udp.port == 9000
  assert video_raw.udp.sync is True
  assert video_raw.udp.async_ is False
  assert getattr(video_raw.udp, "async") is False
  assert video_raw.rtp.payload_type == 99
  assert video_raw.rtp.config_interval == 4
  assert video_raw.encoder.bitrate_kbps == 2500
  assert video_raw.encoder.profile == "main"
  assert video_raw.encoder.level == "4.1"
  assert video_encoded.is_encoded_input() is True
  assert video_encoded.is_raw_input() is False

  assert metadata_sender.host == "127.0.0.1"
  assert metadata_sender.channel == 2
  assert metadata_sender.metadata_port_base == 9100

  assert udp_group.h264_caps == "video/x-h264"
  assert udp_group.payload_type == 98
  assert udp_group.config_interval == 3
  assert udp_group.enable_timings is True
  assert udp_group.host == "127.0.0.1"
  assert udp_group.video_port_base == 9200
  assert udp_group.udp_sync is False
  assert udp_group.udp_async is False

  assert metadata_group.host == "127.0.0.1"
  assert metadata_group.metadata_port_base == 9300


def test_input_stage_node_factories_present_and_accept_expected_args():
  assert hasattr(pyneat.nodes, "preproc")
  assert hasattr(pyneat.nodes, "quant_tess")

  _assert_not_type_error(lambda: pyneat.nodes.preproc())
  _assert_not_type_error(lambda: pyneat.nodes.preproc(pyneat.PreprocOptions()))
  _assert_not_type_error(lambda: pyneat.nodes.quant_tess())
  _assert_not_type_error(lambda: pyneat.nodes.quant_tess(pyneat.QuantTessOptions()))


def test_postprocess_stage_node_factories_present_and_accept_expected_args(tmp_path):
  mpk_path = _strict_yolo_mpk_path()
  model = pyneat.Model(str(mpk_path))

  assert hasattr(pyneat.nodes, "detess_dequant")
  assert hasattr(pyneat.nodes, "sima_box_decode")

  _assert_not_type_error(lambda: pyneat.nodes.detess_dequant())
  _assert_not_type_error(lambda: pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions()))
  _assert_not_type_error(lambda: pyneat.nodes.sima_box_decode(model))
  _assert_not_type_error(
      lambda: pyneat.nodes.sima_box_decode(
          model,
          decode_type=pyneat.BoxDecodeType.YoloV8,
          original_width=640,
          original_height=640,
          detection_threshold=0.25,
          nms_iou_threshold=0.55,
          top_k=120,
      )
  )


def test_output_stage_node_and_group_factories_present_and_accept_expected_args():
  assert hasattr(pyneat.nodes, "udp_output")
  assert hasattr(pyneat.nodes, "h264_encode_sima")
  assert hasattr(pyneat.nodes, "h264_parse")
  assert hasattr(pyneat.nodes, "h264_packetize")
  assert hasattr(pyneat.groups, "udp_h264_output_group")
  assert hasattr(pyneat.groups, "video_sender")

  _assert_not_type_error(lambda: pyneat.nodes.udp_output())
  _assert_not_type_error(lambda: pyneat.nodes.udp_output(pyneat.UdpOutputOptions()))
  _assert_not_type_error(lambda: pyneat.nodes.h264_encode_sima(1280, 720, 30))
  _assert_not_type_error(
      lambda: pyneat.nodes.h264_encode_sima(
          1280, 720, 30, bitrate_kbps=2500, profile="main", level="4.1"
      )
  )
  _assert_not_type_error(lambda: pyneat.nodes.h264_parse())
  _assert_not_type_error(lambda: pyneat.nodes.h264_parse(2))
  _assert_not_type_error(lambda: pyneat.nodes.h264_parse(pyneat.H264ParseOptions()))
  _assert_not_type_error(lambda: pyneat.nodes.h264_packetize())
  _assert_not_type_error(lambda: pyneat.nodes.h264_packetize(payload_type=98, config_interval=3))
  _assert_not_type_error(
      lambda: pyneat.groups.udp_h264_output_group(pyneat.UdpH264OutputGroupOptions())
  )
  assert isinstance(
      pyneat.groups.udp_h264_output_group(pyneat.UdpH264OutputGroupOptions()), pyneat.NodeGroup
  )
  _assert_not_type_error(
      lambda: pyneat.groups.video_sender(
          pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()
      )
  )


def test_explicit_rtsp_decode_node_factories_present_and_accept_expected_args():
  assert hasattr(pyneat.nodes, "queue")
  assert hasattr(pyneat.nodes, "rtsp_input")
  assert hasattr(pyneat.nodes, "h264_depacketize")
  assert hasattr(pyneat.nodes, "h264_decode")

  _assert_not_type_error(lambda: pyneat.nodes.queue())
  _assert_not_type_error(lambda: pyneat.nodes.rtsp_input("rtsp://127.0.0.1:8554/src"))
  _assert_not_type_error(
      lambda: pyneat.nodes.rtsp_input(
          "rtsp://127.0.0.1:8554/src",
          latency_ms=120,
          tcp=False,
          drop_on_latency=True,
          buffer_mode="none",
      )
  )
  _assert_not_type_error(lambda: pyneat.nodes.h264_depacketize())
  _assert_not_type_error(
      lambda: pyneat.nodes.h264_depacketize(
          payload_type=96,
          h264_parse_config_interval=1,
          h264_fps=15,
          h264_width=1280,
          h264_height=720,
          enforce_h264_caps=True,
      )
  )
  _assert_not_type_error(lambda: pyneat.nodes.h264_decode())
  _assert_not_type_error(
      lambda: pyneat.nodes.h264_decode(
          sima_allocator_type=2,
          out_format="NV12",
          decoder_name="",
          raw_output=True,
          next_element="",
          dec_width=-1,
          dec_height=-1,
          dec_fps=-1,
          num_buffers=7,
      )
  )


def test_mla_group_helper_present_and_accepts_model():
  mpk_path = _strict_resnet50_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))

  assert hasattr(pyneat.groups, "mla")
  _assert_not_type_error(lambda: pyneat.groups.mla(model))
  assert isinstance(pyneat.groups.mla(model), pyneat.NodeGroup)


def _resnet_model_with_preproc(*, normalize: pyneat.AutoFlag = pyneat.AutoFlag.On):
  opt = pyneat.ModelOptions()
  # Force a real Preproc route through explicit planner intent.  When normalize
  # is disabled, resize=On keeps the route in the Preproc family without relying
  # on legacy per-stage JSON files.
  opt.preprocess.normalize.enable = normalize
  if normalize == pyneat.AutoFlag.Off:
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
  return pyneat.Model(str(_strict_resnet50_mpk_path()), opt)


def _yolo_model_with_boxdecode():
  opt = pyneat.ModelOptions()
  opt.decode_type = pyneat.BoxDecodeType.YoloV8
  return pyneat.Model(str(_strict_yolo_mpk_path()), opt)


def test_session_describe_backend_includes_preproc_stage():
  model = _resnet_model_with_preproc()
  pre = pyneat.PreprocOptions(model)

  session = pyneat.Session()
  session.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  session.add(pyneat.nodes.preproc(pre))
  session.add(pyneat.nodes.output())

  text = session.describe_backend().lower()
  assert "preproc" in text
  assert pre.normalize is True


def test_session_describe_backend_includes_quant_tess_stage():
  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.nodes.quant_tess())
  session.add(pyneat.nodes.output())

  text = session.describe_backend().lower()
  assert "quanttess" in text or "quant_tess" in text


def test_session_describe_backend_includes_detess_dequant_stage(tmp_path):
  mpk_path = _strict_yolo_mpk_path()
  model = pyneat.Model(str(mpk_path))

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.mla(model))
  session.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
  session.add(pyneat.nodes.output())

  text = session.describe_backend().lower()
  assert "detessdequant" in text


def test_session_describe_backend_includes_sima_box_decode_stage(tmp_path):
  model = _yolo_model_with_boxdecode()

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.mla(model))
  session.add(pyneat.nodes.sima_box_decode(model, decode_type=pyneat.BoxDecodeType.YoloV8))
  session.add(pyneat.nodes.output())

  text = session.describe_backend().lower()
  assert "boxdecode" in text


def test_model_preproc_normalize_explicit_on_resolves_preproc_semantics(tmp_path):
  model = _resnet_model_with_preproc(normalize=pyneat.AutoFlag.On)
  pre = pyneat.PreprocOptions(model)

  session = pyneat.Session()
  session.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  session.add(pyneat.nodes.preproc(pre))
  session.add(pyneat.nodes.output())

  backend = session.describe_backend().lower()
  assert "preproc" in backend
  assert pre.normalize is True


def test_model_preproc_normalize_false_overrides_model_pack_value(tmp_path):
  model = _resnet_model_with_preproc(normalize=pyneat.AutoFlag.Off)
  pre = pyneat.PreprocOptions(model)

  session = pyneat.Session()
  session.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  session.add(pyneat.nodes.preproc(pre))
  session.add(pyneat.nodes.output())

  backend = session.describe_backend().lower()
  assert "preproc" in backend
  assert pre.normalize is False


def test_sima_box_decode_without_runtime_dims_uses_model_pack_defaults(tmp_path):
  model = _yolo_model_with_boxdecode()

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.mla(model))
  session.add(pyneat.nodes.sima_box_decode(model, decode_type=pyneat.BoxDecodeType.YoloV8))
  session.add(pyneat.nodes.output())

  backend = session.describe_backend().lower()
  assert "boxdecode" in backend
  assert "detection-threshold=" not in backend
  assert "nms-iou-threshold=" not in backend
  assert "topk=" not in backend


def test_sima_box_decode_runtime_dims_override_backend_config(tmp_path):
  model = _yolo_model_with_boxdecode()

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.mla(model))
  session.add(
      pyneat.nodes.sima_box_decode(
          model,
          decode_type=pyneat.BoxDecodeType.YoloV8,
          original_width=640,
          original_height=360,
          detection_threshold=0.25,
          nms_iou_threshold=0.55,
          top_k=120,
      )
  )
  session.add(pyneat.nodes.output())

  backend = session.describe_backend().lower()
  assert "boxdecode" in backend
  assert "original-width=640" in backend
  assert "original-height=360" in backend


def test_session_describe_backend_includes_explicit_h264_udp_output_chain():
  parse = pyneat.H264ParseOptions()
  parse.config_interval = 2
  parse.enforce_caps = True
  parse.alignment = pyneat.H264ParseAlignment.AU
  parse.stream_format = pyneat.H264ParseStreamFormat.ByteStream

  udp = pyneat.UdpOutputOptions()
  udp.host = "10.0.0.5"
  udp.port = 5500
  udp.sync = True
  udp.async_ = False

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(
      pyneat.nodes.h264_encode_sima(
          1280, 720, 30, bitrate_kbps=2500, profile="main", level="4.1"
      )
  )
  session.add(pyneat.nodes.h264_parse(parse))
  session.add(pyneat.nodes.h264_packetize(payload_type=98, config_interval=2))
  session.add(pyneat.nodes.udp_output(udp))

  text = session.describe_backend().lower()
  assert "neatencoder" in text
  assert "h264parse" in text
  assert "alignment=(string)au" in text
  assert "stream-format=(string)byte-stream" in text
  assert "rtph264pay" in text
  assert "pt=98" in text
  assert "udpsink" in text
  assert "host=10.0.0.5" in text
  assert "port=5500" in text


def test_session_describe_backend_includes_udp_h264_output_group():
  opt = pyneat.UdpH264OutputGroupOptions()
  opt.h264_caps = 'video/x-h264,profile="high"'
  opt.payload_type = 97
  opt.config_interval = 2
  opt.udp_host = "127.0.0.1"
  opt.udp_port = 5600
  opt.udp_sync = False
  opt.udp_async = False

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.udp_h264_output_group(opt))

  text = session.describe_backend().lower()
  assert "h264parse" in text
  assert "capsfilter" in text
  assert 'profile=\\"high\\"' in text
  assert "rtph264pay" in text
  assert "pt=97" in text
  assert "udpsink" in text
  assert "port=5600" in text


def test_session_describe_backend_includes_video_sender_group():
  opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(1280, 720, 30)
  opt.udp.host = "127.0.0.1"
  opt.udp.port = 5700
  opt.encoder.bitrate_kbps = 2500

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.video_sender(opt))

  text = session.describe_backend().lower()
  assert "videoconvert" in text
  assert "neatencoder" in text
  assert "rtph264pay" in text
  assert "udpsink" in text
  assert "port=5700" in text


def test_model_surface_fixtures_are_real_strict_model_tars():
  for mpk_path in (_strict_resnet50_mpk_path(), _strict_yolo_mpk_path()):
    assert mpk_path.name.endswith(".tar.gz")
    assert model_fixtures.has_strict_mpk_json(mpk_path)


def test_postprocess_stage_api_parity_guards_supported_call_surface(tmp_path):
  detess_path = _strict_yolo_mpk_path()
  box_path = _strict_yolo_mpk_path()
  detess_model = pyneat.Model(str(detess_path))
  box_model = pyneat.Model(str(box_path))

  _assert_not_type_error(lambda: pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(detess_model)))
  _assert_not_type_error(lambda: pyneat.nodes.sima_box_decode(box_model, pyneat.BoxDecodeType.YoloV8, 640, 640, 0.25, 0.55, 120))

  with pytest.raises(TypeError):
    pyneat.nodes.detess_dequant(detess_model)

  with pytest.raises(TypeError):
    pyneat.nodes.sima_box_decode(box_model, threshold=0.25)


def test_output_stage_api_parity_guards_supported_call_surface():
  udp = pyneat.UdpOutputOptions()
  parse = pyneat.H264ParseOptions()
  group = pyneat.UdpH264OutputGroupOptions()

  for name in ("udp_output", "h264_encode_sima", "h264_parse", "h264_packetize"):
    assert hasattr(pyneat.nodes, name), name
  assert hasattr(pyneat.groups, "udp_h264_output_group")
  assert hasattr(pyneat.groups, "video_sender")

  assert hasattr(udp, "async_")
  assert hasattr(udp, "async")
  assert udp.host == "127.0.0.1"
  assert udp.port == 5000
  assert udp.sync is False
  assert udp.async_ is False

  assert parse.config_interval == 1
  assert parse.alignment == pyneat.H264ParseAlignment.Auto
  assert parse.stream_format == pyneat.H264ParseStreamFormat.Auto
  assert parse.enforce_caps is False

  assert group.h264_caps == ""
  assert group.payload_type == 96
  assert group.config_interval == 1
  assert group.udp_host == "127.0.0.1"
  assert group.udp_port == 5000
  assert group.udp_sync is False
  assert group.udp_async is False

  _assert_not_type_error(lambda: pyneat.nodes.input())
  _assert_not_type_error(lambda: pyneat.nodes.output())
  _assert_not_type_error(lambda: pyneat.nodes.video_convert())
  _assert_not_type_error(lambda: pyneat.nodes.h264_parse(1))
  _assert_not_type_error(lambda: pyneat.nodes.h264_parse(parse))
  _assert_not_type_error(lambda: pyneat.nodes.h264_packetize(96, 1))
  _assert_not_type_error(lambda: pyneat.nodes.udp_output(udp))
  _assert_not_type_error(lambda: pyneat.groups.udp_h264_output_group(group))
  _assert_not_type_error(
      lambda: pyneat.groups.video_sender(
          pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()
      )
  )


def test_error_code_constants_present():
  assert isinstance(pyneat.ERROR_PIPELINE_SHAPE, str)
  assert isinstance(pyneat.ERROR_RUNTIME_PULL, str)


def test_runtime_overload_methods_present():
  assert hasattr(pyneat.Run, "push")
  assert hasattr(pyneat.Run, "try_push")
  assert hasattr(pyneat.Run, "run")
  assert hasattr(pyneat.ModelRunner, "push")
  assert hasattr(pyneat.ModelRunner, "run")
  assert hasattr(pyneat.Model, "run")


def test_session_build_accepts_numpy_without_type_error():
  session = pyneat.Session()
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: session.build(arr))
  _assert_not_type_error(lambda: session.build(arr, copy=True))
  _assert_not_type_error(
      lambda: session.build(arr, layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB)
  )


def test_inputstream_stats_extended_fields_present():
  s = pyneat.InputStreamStats()
  assert hasattr(s, "alloc_grows")
  assert hasattr(s, "growth_blocked")
  assert hasattr(s, "renegotiation_blocked")



def test_native_build_overload_marker_present():
  import pyneat._pyneat_core as core

  assert bool(getattr(core, "_HAS_NATIVE_BUILD_OBJECT_OVERLOADS", False))



def test_model_build_accepts_numpy_without_type_error():
  mpk_path = _strict_resnet50_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: model.build(arr))
  _assert_not_type_error(lambda: model.build(arr, copy=True))



def test_session_build_accepts_torch_without_type_error():
  try:
    import torch
  except Exception:
    return

  session = pyneat.Session()
  # Use an intentionally invalid rank to keep this as API-overload coverage
  # only: this should fail fast in conversion/validation, but never with TypeError.
  tensor = torch.zeros((8, 8), dtype=torch.uint8)
  _assert_not_type_error(
      lambda: session.build(
          tensor, layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB
      )
  )


def test_model_build_accepts_torch_without_type_error():
  try:
    import torch
  except Exception:
    return

  mpk_path = _strict_resnet50_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))
  # Same fast-path API-overload validation strategy as Session test.
  tensor = torch.zeros((8, 8), dtype=torch.uint8)

  _assert_not_type_error(lambda: model.build(tensor))


def test_model_build_requires_explicit_image_semantic_for_image_tensor():
  mpk_path = _strict_resnet50_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Image
  opt.preprocess.enable = pyneat.AutoFlag.On
  opt.preprocess.resize.enable = pyneat.AutoFlag.On
  opt.preprocess.normalize.enable = pyneat.AutoFlag.On
  model = pyneat.Model(str(mpk_path), opt)
  tensor = pyneat.Tensor.from_numpy(np.zeros((8, 8, 3), dtype=np.uint8), copy=True)

  with pytest.raises(ValueError, match="requires explicit image format"):
    model.build(tensor)


def test_tensor_from_numpy_byte_stream_marks_opaque_transport():
  spec = pyneat.ByteStreamSpec()
  spec.format = pyneat.ByteFormat.Raw
  spec.description = "opaque raw bytes"
  assert spec.format == pyneat.ByteFormat.Raw
  assert spec.description == "opaque raw bytes"

  tensor = pyneat.Tensor.from_numpy(
      np.arange(16, dtype=np.int8),
      copy=True,
      byte_format=pyneat.ByteFormat.Raw,
      memory=pyneat.TensorMemory.CPU,
  )

  assert tensor.semantic.byte_stream is not None
  assert tensor.semantic.byte_stream.format == pyneat.ByteFormat.Raw
  assert tensor.layout == pyneat.TensorLayout.Unknown
  ok, err = tensor.validate()
  assert ok, err
  assert "ByteStream" in tensor.debug_string()


def test_byte_stream_is_incompatible_with_image_semantic():
  with pytest.raises(RuntimeError, match="byte_format tensors cannot also specify image_format"):
    pyneat.Tensor.from_numpy(
        np.zeros((2, 2, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        byte_format=pyneat.ByteFormat.Raw,
        memory=pyneat.TensorMemory.CPU,
    )


def test_model_run_accepts_chw_torch_without_layout_or_image_format():
  try:
    import torch
  except Exception:
    return

  mpk_path = _strict_resnet50_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))
  tensor = torch.zeros((3, 8, 8), dtype=torch.uint8)
  _assert_not_type_error(lambda: model.run(tensor, timeout_ms=1))


def test_model_run_build_reject_layout_and_image_format_kwargs():
  mpk_path = _strict_resnet50_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  try:
    model.build(arr, layout=pyneat.TensorLayout.HWC)
  except TypeError:
    pass
  else:
    raise AssertionError("expected model.build(..., layout=...) to fail with TypeError")

  try:
    model.run(arr, image_format=pyneat.PixelFormat.RGB, timeout_ms=1)
  except TypeError:
    pass
  else:
    raise AssertionError("expected model.run(..., image_format=...) to fail with TypeError")


def test_session_video_push_uses_input_format_without_image_semantic():
  session = pyneat.Session()
  opt = pyneat.InputOptions()
  opt.media_type = "video/x-raw"
  opt.format = "RGB"
  opt.use_simaai_pool = False
  session.add(pyneat.nodes.input(opt))
  session.add(pyneat.nodes.output())

  frame = np.zeros((16, 16, 3), dtype=np.uint8)
  run = session.build(frame)
  try:
    assert run.push(frame)
    out = run.pull(1000)
    assert out is not None
  finally:
    run.close_input()
    run.close()


def test_session_error_in_python_exposes_structured_fields():
  session = pyneat.Session()

  try:
    session.build(np.zeros((8, 8, 3), dtype=np.uint8))
  except pyneat.SessionError as exc:
    text = str(exc)
    assert text
    assert text != "["
    assert isinstance(exc.error_code, str)
    assert exc.error_code
    assert isinstance(exc.repro_note, str)
    assert isinstance(exc.report_json, str)
    assert exc.report_json
    assert "error_code" in exc.report_json
  else:
    raise AssertionError("expected SessionError for empty pipeline")


def test_graph_pipeline_node_accepts_nodegroup_without_type_error():
  group = pyneat.NodeGroup([pyneat.nodes.video_convert()])

  _assert_not_type_error(lambda: pyneat.graph.nodes.pipeline_node(group, "group"))


def test_graph_pipeline_node_preserves_existing_node_overload():
  node = pyneat.graph.nodes.pipeline_node(pyneat.nodes.video_convert(), "convert")
  graph = pyneat.graph.Graph()
  graph.add(node)

  text = pyneat.graph.to_text(graph)
  assert "convert" in text
  assert "in: in" in text
  assert "out: out" in text


def test_graph_pipeline_node_wraps_push_style_nodegroup_in_graph_text():
  group = pyneat.NodeGroup([pyneat.nodes.video_convert()])
  graph = pyneat.graph.Graph()
  graph.add(pyneat.graph.nodes.pipeline_node(group, "push-group"))

  text = pyneat.graph.to_text(graph)
  assert "push-group" in text
  assert "in: in" in text
  assert "out: out" in text


def test_graph_pipeline_node_wraps_source_like_nodegroup_without_input_port():
  opt = pyneat.RtspDecodedInputOptions()
  opt.url = "rtsp://example.com/live"

  group = pyneat.groups.rtsp_decoded_input(opt)
  graph = pyneat.graph.Graph()
  graph.add(pyneat.graph.nodes.pipeline_node(group, "rtsp-source"))

  text = pyneat.graph.to_text(graph)
  assert "rtsp-source" in text
  assert "in: <none>" in text
  assert "out: out" in text
