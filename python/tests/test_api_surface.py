
import importlib
import warnings

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
    "host",
    "channel",
    "video_port_base",
    "sync",
    "async_",
    "async",
    "rtp",
    "encoder",
    "is_raw_input",
    "is_encoded_input",
    "width",
    "height",
    "fps",
    "video_port",
)

METADATA_SENDER_OPTION_FIELDS = (
    "host",
    "channel",
    "metadata_port_base",
)


def _strict_resnet50_model_path():
  return model_fixtures.strict_model_tar_path("SIMA_RESNET50_TAR")


def _strict_yolo_model_path():
  return model_fixtures.strict_model_tar_path("SIMA_YOLO_TAR")


def _assert_not_type_error(call):
  try:
    call()
  except Exception as exc:
    assert not isinstance(exc, TypeError), str(exc)


def test_graph_only_public_surface():
  assert hasattr(pyneat, "Graph")
  assert hasattr(pyneat, "GraphOptions")
  assert hasattr(pyneat, "GraphLinkOptions")
  assert hasattr(pyneat, "GraphLinkPolicy")
  assert hasattr(pyneat, "GraphReport")
  assert hasattr(pyneat, "NeatError")
  assert hasattr(pyneat, "ModelRouteOptions")
  assert hasattr(pyneat, "graphs")
  assert hasattr(pyneat.graphs, "branch")
  assert hasattr(pyneat.graphs, "combine")

  for removed_name in (
      "graph",
      "_graph",
      "Session",
      "SessionError",
      "SessionReport",
      "SessionOptions",
      "ModelSessionOptions",
  ):
    assert not hasattr(pyneat, removed_name)
    with pytest.raises(AttributeError, match="removed|renamed"):
      getattr(pyneat, removed_name)
  with pytest.raises(ModuleNotFoundError):
    importlib.import_module("pyneat._graph")

  assert not hasattr(pyneat.Graph, "add_group")


def test_graph_pythonic_add_and_describe():
  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input())
  graph.add(pyneat.nodes.output())

  text = graph.describe_backend()
  assert isinstance(text, str)
  assert text


def test_graph_pythonic_add_graph_and_connect_alias():
  source = pyneat.Graph()
  source.custom_with_role(
      "videotestsrc num-buffers=1 is-live=false ! "
      "video/x-raw,format=RGB,width=16,height=16,framerate=1/1",
      pyneat.InputRole.Source,
  )
  sink = pyneat.Graph()
  sink.add(pyneat.nodes.output())

  spliced = pyneat.Graph()
  assert spliced.add(source) is spliced
  assert spliced.add(sink) is spliced
  assert "videotestsrc" in spliced.describe_backend()

  connected = pyneat.Graph()
  assert connected.connect(source, sink) is connected


def test_graph_link_options_surface():
  opt = pyneat.GraphLinkOptions()
  assert opt.policy == pyneat.GraphLinkPolicy.Default
  opt.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
  opt.queue_depth = 7
  opt.stream_id = "camera0"
  assert opt.policy == pyneat.GraphLinkPolicy.RealtimeLatestByStream
  assert opt.queue_depth == 7
  assert opt.stream_id == "camera0"

  source = pyneat.Graph()
  source.custom_with_role(
      "videotestsrc num-buffers=1 is-live=false ! "
      "video/x-raw,format=RGB,width=16,height=16,framerate=1/1",
      pyneat.InputRole.Source,
  )
  sink = pyneat.Graph()
  sink.add(pyneat.nodes.output())

  connected = pyneat.Graph()
  assert connected.connect(source, sink, opt) is connected


def test_named_graph_endpoint_api_surface():
  source = pyneat.Graph("image")
  assert source.name == "image"
  assert source.set_name("camera") is source
  assert source.name == "camera"
  source.add(pyneat.nodes.input("image"))

  sink = pyneat.Graph("ignored")
  sink.add(pyneat.nodes.output("classes"))

  connected = pyneat.Graph()
  assert connected.connect(source, sink) is connected

  run = pyneat.Run()
  assert run.input_names() == []
  assert run.output_names() == []


def test_run_export_api_surface():
  export_opt = pyneat.RunExportOptions()
  export_opt.label = "surface"
  export_opt.include_metrics = False
  export_opt.include_power = False
  export_opt.include_node_metrics = True
  export_opt.include_plugin_metrics = True
  export_opt.include_empty_node_metrics = False
  export_opt.indent = 0
  export_opt.metadata = [("suite", "api")]
  assert export_opt.label == "surface"
  assert export_opt.metadata == [("suite", "api")]
  assert export_opt.include_empty_node_metrics is False

  auto_opt = pyneat.RunAutoExportOptions()
  auto_opt.path = "/tmp/pyneat_graph_run_surface.json"
  auto_opt.label = "auto"
  auto_opt.include_metrics = True
  auto_opt.include_power = False
  auto_opt.include_node_metrics = True
  auto_opt.include_plugin_metrics = False
  auto_opt.include_empty_node_metrics = True
  auto_opt.indent = 2

  run_opt = pyneat.RunOptions()
  run_opt.run_export = auto_opt
  run_opt.enable_board_power()
  assert not hasattr(run_opt, "enable_metrics")
  assert run_opt.power_monitor.enabled is True
  run_opt.disable_power_monitor()
  assert run_opt.power_monitor.enabled is False
  run_opt.power_monitor = pyneat.modalix_som_power_monitor_options(250)
  assert run_opt.power_monitor.enabled is True
  assert run_opt.run_export.path.endswith("pyneat_graph_run_surface.json")
  assert run_opt.run_export.label == "auto"
  assert run_opt.run_export.include_plugin_metrics is False

  assert not hasattr(pyneat, "RuntimeMetricsOptions")
  assert not hasattr(pyneat, "RuntimeMetrics")
  assert not hasattr(pyneat, "GraphMetricsReport")
  assert not hasattr(pyneat, "build_graph_metrics_report_run_lifetime")

  measure_report = pyneat.MeasureReport()
  measure_report.elapsed_s = 1.0
  counters = pyneat.MeasureCounters()
  counters.outputs_pulled = 2
  measure_report.counters = counters
  measure_report.throughput_batches_per_s = 2.0
  plugin = pyneat.MeasurePluginLatency()
  plugin.backend = "A65"
  plugin.phase = "Run"
  plugin.kernel_name = "identity"
  plugin.runtime_node_id = 0
  plugin.calls = 1
  measure_report.plugin_latency = [plugin]
  assert measure_report.plugin_latency[0].runtime_node_id == 0
  assert hasattr(measure_report, "to_json")
  assert "sima.neat.measure_report" in measure_report.to_json(0)

  run = pyneat.Run()
  with pytest.raises(Exception, match="Run has no runtime core"):
    pyneat.run_to_json(run, export_opt)
  with pytest.raises(Exception, match="Run has no runtime core"):
    pyneat.run_to_json(run, measure_report, export_opt)


def test_public_graph_connect_no_runtime_port_overload():
  source = pyneat.Graph("image")
  source.add(pyneat.nodes.input("image"))

  sink = pyneat.Graph("classes")
  sink.add(pyneat.nodes.output("classes"))

  app = pyneat.Graph()
  app.connect(source, sink)
  assert "endpoint image -> classes" in app.describe()

  with pytest.raises(TypeError):
    app.connect(source, sink, "out", "in")
  assert not hasattr(pyneat.Graph, "connect_port")


def test_model_graph_fragment_and_direct_graph_add():
  model = pyneat.Model(str(_strict_yolo_model_path()))

  fragment = model.graph()
  backend = fragment.describe_backend().lower()
  assert "processmla" in backend
  assert "appsrc" not in backend
  assert "appsink" not in backend

  graph = pyneat.Graph()
  assert graph.add(model) is graph
  direct_backend = graph.describe_backend().lower()
  assert "processmla" in direct_backend
  assert "appsrc" not in direct_backend
  assert "appsink" not in direct_backend


def test_model_option_structs_are_mutable():
  opt = pyneat.ModelOptions()
  opt.preprocess.input_max_width = 1920
  opt.preprocess.input_max_height = 1080
  opt.boxdecode_original_width = 1280
  opt.boxdecode_original_height = 720
  opt.boxdecode_resize_mode = pyneat.ResizeMode.Letterbox

  assert opt.preprocess.input_max_width == 1920
  assert opt.preprocess.input_max_height == 1080
  assert opt.boxdecode_original_width == 1280
  assert opt.boxdecode_original_height == 720
  assert opt.boxdecode_resize_mode == pyneat.ResizeMode.Letterbox


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
  video_rtp = pyneat.VideoSenderRtpOptions()
  video_encoder = pyneat.VideoSenderEncoderOptions()
  video_sender = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(640, 480, 30)
  metadata_sender = pyneat.MetadataSenderOptions()

  for field in UDP_OUTPUT_OPTION_FIELDS:
    assert hasattr(udp, field), field
  assert hasattr(udp, "async"), "async"

  for field in H264_PARSE_OPTION_FIELDS:
    assert hasattr(parse, field), field

  for field in UDP_H264_OUTPUT_GROUP_OPTION_FIELDS:
    assert hasattr(group, field), field

  for field in VIDEO_SENDER_RTP_OPTION_FIELDS:
    assert hasattr(video_rtp, field), field

  for field in VIDEO_SENDER_ENCODER_OPTION_FIELDS:
    assert hasattr(video_encoder, field), field

  for field in VIDEO_SENDER_OPTION_FIELDS:
    assert hasattr(video_sender, field), field

  for field in METADATA_SENDER_OPTION_FIELDS:
    assert hasattr(metadata_sender, field), field

  assert hasattr(pyneat, "H264ParseAlignment")
  assert hasattr(pyneat, "H264ParseStreamFormat")
  assert hasattr(pyneat, "VideoSenderRtpOptions")
  assert hasattr(pyneat, "VideoSenderEncoderOptions")
  assert hasattr(pyneat, "VideoSenderOptions")
  assert hasattr(pyneat, "MetadataSenderOptions")
  assert hasattr(pyneat, "MetadataSender")
  assert hasattr(pyneat.groups, "video_sender")

  for removed_name in (
      "OptiViewObject",
      "OptiViewChannelOptions",
      "UdpOutputGraphOptions",
      "OptiViewOutputGraphOptions",
      "OptiViewJsonInput",
      "OptiViewJsonResult",
      "OptiViewJsonOutput",
      "OptiViewOutputGraph",
      "OptiViewMakeJson",
  ):
    assert not hasattr(pyneat, removed_name)


def test_input_stage_option_struct_constructors_accept_expected_args():
  model_path = _strict_resnet50_model_path()
  assert model_path.exists(), f"missing fixture: {model_path}"

  model = pyneat.Model(str(model_path))

  _assert_not_type_error(lambda: pyneat.PreprocOptions())
  _assert_not_type_error(lambda: pyneat.PreprocOptions(model))
  _assert_not_type_error(lambda: pyneat.QuantTessOptions())
  _assert_not_type_error(lambda: pyneat.QuantTessOptions(model))


def test_postprocess_stage_option_struct_constructors_accept_expected_args(tmp_path):
  model_path = _strict_yolo_model_path()
  model = pyneat.Model(str(model_path))

  _assert_not_type_error(lambda: pyneat.DetessDequantOptions())
  _assert_not_type_error(lambda: pyneat.DetessDequantOptions(model))


def test_output_stage_option_struct_constructors_accept_expected_args():
  _assert_not_type_error(lambda: pyneat.UdpOutputOptions())
  _assert_not_type_error(lambda: pyneat.H264ParseOptions())
  _assert_not_type_error(lambda: pyneat.UdpH264OutputGroupOptions())
  _assert_not_type_error(lambda: pyneat.VideoSenderRtpOptions())
  _assert_not_type_error(lambda: pyneat.VideoSenderEncoderOptions())
  _assert_not_type_error(lambda: pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(640, 480, 30))
  _assert_not_type_error(lambda: pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded())
  _assert_not_type_error(lambda: pyneat.MetadataSenderOptions())
  _assert_not_type_error(lambda: pyneat.MetadataSender(pyneat.MetadataSenderOptions()))


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

  video_rtp = pyneat.VideoSenderRtpOptions()
  video_rtp.payload_type = 98
  video_rtp.config_interval = 3

  video_encoder = pyneat.VideoSenderEncoderOptions()
  video_encoder.bitrate_kbps = 2500
  video_encoder.profile = "main"
  video_encoder.level = "4.1"

  video_sender = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(640, 480, 30)
  video_sender.host = "127.0.0.1"
  video_sender.channel = 2
  video_sender.video_port_base = 9200
  video_sender.sync = False
  video_sender.async_ = False
  video_sender.rtp = video_rtp
  video_sender.encoder = video_encoder

  encoded_sender = pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()

  metadata_sender = pyneat.MetadataSenderOptions()
  metadata_sender.host = "127.0.0.1"
  metadata_sender.channel = 2
  metadata_sender.metadata_port_base = 9300

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

  assert video_rtp.payload_type == 98
  assert video_rtp.config_interval == 3

  assert video_encoder.bitrate_kbps == 2500
  assert video_encoder.profile == "main"
  assert video_encoder.level == "4.1"

  assert video_sender.is_raw_input() is True
  assert video_sender.is_encoded_input() is False
  assert video_sender.width == 640
  assert video_sender.height == 480
  assert video_sender.fps == 30
  assert video_sender.host == "127.0.0.1"
  assert video_sender.channel == 2
  assert video_sender.video_port_base == 9200
  assert video_sender.video_port == 9202
  assert video_sender.sync is False
  assert video_sender.async_ is False
  assert getattr(video_sender, "async") is False
  assert video_sender.rtp.payload_type == 98
  assert video_sender.encoder.profile == "main"

  assert encoded_sender.is_raw_input() is False
  assert encoded_sender.is_encoded_input() is True
  assert isinstance(pyneat.groups.video_sender(video_sender), pyneat.Graph)
  assert isinstance(pyneat.groups.video_sender(encoded_sender), pyneat.Graph)

  assert metadata_sender.host == "127.0.0.1"
  assert metadata_sender.channel == 2
  assert metadata_sender.metadata_port_base == 9300


def test_input_stage_node_factories_present_and_accept_expected_args():
  assert hasattr(pyneat.nodes, "preproc")
  assert hasattr(pyneat.nodes, "quant_tess")

  _assert_not_type_error(lambda: pyneat.nodes.preproc())
  _assert_not_type_error(lambda: pyneat.nodes.preproc(pyneat.PreprocOptions()))
  _assert_not_type_error(lambda: pyneat.nodes.quant_tess())
  _assert_not_type_error(lambda: pyneat.nodes.quant_tess(pyneat.QuantTessOptions()))


def test_postprocess_stage_node_factories_present_and_accept_expected_args(tmp_path):
  model_path = _strict_yolo_model_path()
  model = pyneat.Model(str(model_path))

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
      pyneat.groups.udp_h264_output_group(pyneat.UdpH264OutputGroupOptions()), pyneat.Graph
  )


def test_explicit_rtsp_decode_node_factories_present_and_accept_expected_args():
  assert hasattr(pyneat.nodes, "queue")
  assert hasattr(pyneat.nodes, "rtsp_input")
  assert hasattr(pyneat.nodes, "h264_depacketize")
  assert hasattr(pyneat.nodes, "h264_decode")
  assert hasattr(pyneat.nodes, "sima_decode")

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
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
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
  assert any(
      issubclass(warning.category, DeprecationWarning)
      and "pyneat.nodes.h264_decode is deprecated" in str(warning.message)
      for warning in caught
  )
  native_decode = pyneat.SimaDecodeOptions()
  assert native_decode.type == pyneat.SimaDecodeType.H264
  assert native_decode.out_format == pyneat.Format.NV12
  assert native_decode.raw_output is True
  _assert_not_type_error(lambda: pyneat.nodes.sima_decode())
  _assert_not_type_error(lambda: pyneat.nodes.sima_decode(native_decode))
  native_decode.type = pyneat.SimaDecodeType.JPEG
  native_decode.raw_output = False
  native_decode.dec_width = 640
  native_decode.dec_height = 480
  native_decode.dec_fps = 30
  _assert_not_type_error(lambda: pyneat.nodes.sima_decode(native_decode))
  native_decode.type = pyneat.SimaDecodeType.MJPEG
  _assert_not_type_error(lambda: pyneat.nodes.sima_decode(native_decode))


def test_jpeg_framing_nodes_are_exposed():
  http = pyneat.HttpSourceOptions()
  assert http.location == ""
  assert http.timeout_seconds == 15
  assert http.retries == 3
  assert http.is_live is False
  assert http.do_timestamp is False
  assert http.ssl_strict is True
  http.location = "http://example.local/mjpeg"
  http.is_live = True
  http.do_timestamp = True
  http.user_agent = "NeatTest"
  http.ssl_strict = False
  _assert_not_type_error(lambda: pyneat.nodes.http_source(http))

  demux = pyneat.MultipartJpegDemuxOptions()
  assert demux.boundary == ""
  assert demux.single_stream is False
  demux.boundary = "frame"
  demux.single_stream = True
  _assert_not_type_error(lambda: pyneat.nodes.multipart_jpeg_demux())
  _assert_not_type_error(lambda: pyneat.nodes.multipart_jpeg_demux(demux))

  fixup = pyneat.EncodedCapsFixupOptions()
  assert fixup.media_type == ""
  assert fixup.fallback_fps == -1
  assert fixup.use_rtsp_sdp_fps is False
  fixup.media_type = "image/jpeg"
  fixup.fallback_fps = 30
  _assert_not_type_error(lambda: pyneat.nodes.encoded_caps_fixup(fixup))

  parser = pyneat.JpegParseOptions()
  assert parser.disable_passthrough is True
  parser.disable_passthrough = False
  _assert_not_type_error(lambda: pyneat.nodes.jpeg_parse())
  _assert_not_type_error(lambda: pyneat.nodes.jpeg_parse(parser))


def test_rtsp_encoded_and_decoded_groups_are_exposed():
  assert pyneat.RtspCodec.H264.name == "H264"
  assert pyneat.RtspCodec.MJPEG.name == "MJPEG"
  _assert_not_type_error(lambda: pyneat.nodes.rtp_jpeg_depacketize())
  _assert_not_type_error(lambda: pyneat.nodes.rtp_jpeg_depacketize(26))

  encoded = pyneat.RtspEncodedInputOptions()
  assert encoded.url == ""
  assert encoded.codec == pyneat.RtspCodec.H264
  assert encoded.latency_ms == 200
  assert encoded.tcp is True
  assert encoded.drop_on_latency is False
  assert encoded.buffer_mode == ""
  assert encoded.insert_queue is True
  assert encoded.sync_mode is False
  assert encoded.h264_payload_type == 96
  assert encoded.mjpeg_payload_type == 26
  assert encoded.auto_caps_from_stream is True
  assert encoded.source_fps == -1

  encoded.url = "rtsp://example.local/mjpeg"
  encoded.codec = pyneat.RtspCodec.MJPEG
  encoded.source_fps = 120
  group = pyneat.groups.rtsp_encoded_input(encoded)
  assert isinstance(group, pyneat.Graph)
  backend = group.describe_backend().lower()
  assert "rtspsrc" in backend
  assert "rtpjpegdepay" in backend
  assert "jpegparse" in backend
  assert "encoded_capsfix" in backend
  assert "rtph264depay" not in backend

  encoded_spec = pyneat.groups.rtsp_encoded_output_spec(encoded)
  assert encoded_spec.payload_type == pyneat.PayloadType.Encoded
  assert encoded_spec.media_type == "image/jpeg"
  assert encoded_spec.format == "JPEG"

  decoded = pyneat.RtspDecodedInputOptions()
  assert decoded.codec == pyneat.RtspCodec.H264
  assert decoded.drop_on_latency is False
  assert decoded.buffer_mode == ""
  assert decoded.payload_type == 96
  assert decoded.mjpeg_payload_type == 26
  assert decoded.dec_width == -1
  assert decoded.dec_height == -1
  assert decoded.dec_fps == -1
  assert decoded.num_buffers == -1
  assert decoded.source_fps == -1
  assert decoded.use_videorate is False
  assert decoded.video_rate_fps == -1
  assert decoded.output_caps.memory == pyneat.CapsMemory.Any

  decoded.url = "rtsp://example.local/mjpeg"
  decoded.codec = pyneat.RtspCodec.MJPEG
  decoded.dec_width = 640
  decoded.dec_height = 480
  decoded.source_fps = 30
  decoded.use_videorate = True
  decoded.video_rate_fps = 15
  decoded.output_caps.width = 640
  decoded.output_caps.height = 480
  decoded.output_caps.fps = 15
  decoded_group = pyneat.groups.rtsp_decoded_input(decoded)
  assert isinstance(decoded_group, pyneat.Graph)
  try:
    decoded_backend = decoded_group.describe_backend().lower()
  except RuntimeError as exc:
    message = str(exc).lower()
    if (
        "required gstreamer element not found: neatdecoder" not in message
        and "required neat factory is missing" not in message
    ):
      raise
    pytest.skip("native decoder backend is not available in this environment")
  assert "rtpjpegdepay" in decoded_backend
  assert "jpegparse" in decoded_backend
  assert "neatdecoder" in decoded_backend
  assert "dec-type=mjpeg" in decoded_backend
  assert "dec-fps=30" in decoded_backend
  assert "videorate" in decoded_backend
  assert "framerate=15/1" in decoded_backend

  decoded_spec = pyneat.groups.rtsp_decoded_output_spec(decoded)
  assert decoded_spec.media_type == "video/x-raw"
  assert decoded_spec.format == "NV12"
  assert decoded_spec.width == 640
  assert decoded_spec.height == 480
  assert decoded_spec.memory == "SimaAI"


def test_http_mjpeg_decoded_input_group_is_exposed():
  opt = pyneat.HttpMjpegDecodedInputOptions()
  assert opt.url == ""
  assert opt.timeout_seconds == 15
  assert opt.retries == 3
  assert opt.is_live is True
  assert opt.do_timestamp is True
  assert opt.ssl_strict is True
  assert opt.multipart_boundary == ""
  assert opt.multipart_single_stream is False
  assert opt.insert_queue is True
  assert opt.sync_mode is False
  assert opt.sima_allocator_type == 2
  assert opt.out_format == pyneat.Format.NV12
  assert opt.decoder_raw_output is True
  assert opt.dec_width == -1
  assert opt.dec_height == -1
  assert opt.dec_fps == -1
  assert opt.num_buffers == -1
  assert opt.source_fps == -1
  assert opt.use_videorate is False
  assert opt.video_rate_fps == -1
  assert opt.output_caps.memory == pyneat.CapsMemory.Any

  opt.url = "http://example.local/mjpeg"
  opt.timeout_seconds = 9
  opt.retries = -1
  opt.user_agent = "NeatTest"
  opt.ssl_strict = False
  opt.multipart_boundary = "frame"
  opt.multipart_single_stream = True
  opt.decoder_name = "mjpeg_decoder"
  opt.dec_width = 640
  opt.dec_height = 480
  opt.source_fps = 30
  opt.use_videorate = True
  opt.video_rate_fps = 15
  opt.output_caps.enable = True
  opt.output_caps.fps = 15
  opt.num_buffers = 8
  group = pyneat.groups.http_mjpeg_decoded_input(opt)
  assert isinstance(group, pyneat.Graph)
  try:
    backend = group.describe_backend().lower()
  except RuntimeError as exc:
    if "required gstreamer element not found: neatdecoder" not in str(exc).lower():
      raise
    pytest.skip("neatdecoder is not available in this environment")
  assert "souphttpsrc" in backend
  assert "ssl-strict=false" in backend
  assert "multipartdemux" in backend
  assert "jpegparse" in backend
  assert "neatdecoder" in backend
  assert "dec-type=mjpeg" in backend
  assert "dec-fps=30" in backend
  assert "videorate" in backend
  assert "framerate=15/1" in backend

  spec = pyneat.groups.http_mjpeg_decoded_output_spec(opt)
  assert spec.format == "NV12"
  assert spec.fps_num == 15

  opt.output_caps.enable = False
  opt.output_caps.width = 320
  opt.output_caps.height = 240
  opt.output_caps.fps = 15
  opt.use_videorate = False
  opt.video_rate_fps = -1
  disabled_caps_spec = pyneat.groups.http_mjpeg_decoded_output_spec(opt)
  assert disabled_caps_spec.width == 640
  assert disabled_caps_spec.height == 480
  assert disabled_caps_spec.fps_num == 30
  assert disabled_caps_spec.memory == "SimaAI"

  opt.output_caps.enable = True
  caps_spec = pyneat.groups.http_mjpeg_decoded_output_spec(opt)
  assert caps_spec.width == 320
  assert caps_spec.height == 240
  assert caps_spec.fps_num == 15
  assert caps_spec.memory == "SimaAI"

  opt.output_caps.memory = pyneat.CapsMemory.SystemMemory
  system_caps_spec = pyneat.groups.http_mjpeg_decoded_output_spec(opt)
  assert system_caps_spec.memory == "SystemMemory"


def test_mla_group_helper_present_and_accepts_model():
  model_path = _strict_resnet50_model_path()
  assert model_path.exists(), f"missing fixture: {model_path}"

  model = pyneat.Model(str(model_path))

  assert hasattr(pyneat.groups, "mla")
  _assert_not_type_error(lambda: pyneat.groups.mla(model))
  assert isinstance(model.preprocess(), pyneat.Graph)
  assert isinstance(model.inference(), pyneat.Graph)
  assert isinstance(model.postprocess(), pyneat.Graph)
  assert isinstance(pyneat.groups.mla(model), pyneat.Graph)


def _resnet_model_with_preproc(*, normalize: pyneat.AutoFlag = pyneat.AutoFlag.On):
  opt = pyneat.ModelOptions()
  # Force a real Preproc route through explicit planner intent.  When normalize
  # is disabled, resize=On keeps the route in the Preproc family without relying
  # on legacy per-stage JSON files.
  opt.preprocess.normalize.enable = normalize
  if normalize == pyneat.AutoFlag.Off:
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
  return pyneat.Model(str(_strict_resnet50_model_path()), opt)


def _yolo_model_with_boxdecode():
  opt = pyneat.ModelOptions()
  opt.decode_type = pyneat.BoxDecodeType.YoloV8
  return pyneat.Model(str(_strict_yolo_model_path()), opt)


def test_graph_describe_backend_includes_preproc_stage():
  model = _resnet_model_with_preproc()
  pre = pyneat.PreprocOptions(model)

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  graph.add(pyneat.nodes.preproc(pre))
  graph.add(pyneat.nodes.output())

  text = graph.describe_backend().lower()
  assert "preproc" in text
  assert pre.normalize is True


def test_graph_describe_backend_includes_quant_tess_stage():
  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input())
  graph.add(pyneat.nodes.quant_tess())
  graph.add(pyneat.nodes.output())

  text = graph.describe_backend().lower()
  assert "quanttess" in text or "quant_tess" in text


def test_graph_describe_backend_includes_detess_dequant_stage(tmp_path):
  model_path = _strict_yolo_model_path()
  model = pyneat.Model(str(model_path))

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input())
  graph.add(pyneat.groups.mla(model))
  graph.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
  graph.add(pyneat.nodes.output())

  text = graph.describe_backend().lower()
  assert "detessdequant" in text


def test_graph_describe_backend_includes_sima_box_decode_stage(tmp_path):
  model = _yolo_model_with_boxdecode()

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input())
  graph.add(pyneat.groups.mla(model))
  graph.add(pyneat.nodes.sima_box_decode(model, decode_type=pyneat.BoxDecodeType.YoloV8))
  graph.add(pyneat.nodes.output())

  text = graph.describe_backend().lower()
  assert "boxdecode" in text


def test_model_preproc_normalize_explicit_on_resolves_preproc_semantics(tmp_path):
  model = _resnet_model_with_preproc(normalize=pyneat.AutoFlag.On)
  pre = pyneat.PreprocOptions(model)

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  graph.add(pyneat.nodes.preproc(pre))
  graph.add(pyneat.nodes.output())

  backend = graph.describe_backend().lower()
  assert "preproc" in backend
  assert pre.normalize is True


def test_model_preproc_normalize_false_overrides_model_pack_value(tmp_path):
  model = _resnet_model_with_preproc(normalize=pyneat.AutoFlag.Off)
  pre = pyneat.PreprocOptions(model)

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  graph.add(pyneat.nodes.preproc(pre))
  graph.add(pyneat.nodes.output())

  backend = graph.describe_backend().lower()
  assert "preproc" in backend
  assert pre.normalize is False


def test_sima_box_decode_without_runtime_dims_uses_model_pack_defaults(tmp_path):
  model = _yolo_model_with_boxdecode()

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input())
  graph.add(pyneat.groups.mla(model))
  graph.add(pyneat.nodes.sima_box_decode(model, decode_type=pyneat.BoxDecodeType.YoloV8))
  graph.add(pyneat.nodes.output())

  backend = graph.describe_backend().lower()
  assert "boxdecode" in backend
  assert "detection-threshold=" not in backend
  assert "nms-iou-threshold=" not in backend
  assert "topk=" not in backend


def test_sima_box_decode_runtime_dims_override_backend_config(tmp_path):
  model = _yolo_model_with_boxdecode()

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input())
  graph.add(pyneat.groups.mla(model))
  graph.add(
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
  graph.add(pyneat.nodes.output())

  backend = graph.describe_backend().lower()
  assert "boxdecode" in backend
  assert "original-width=640" in backend
  assert "original-height=360" in backend


def test_graph_describe_backend_includes_explicit_h264_udp_output_chain():
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

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input())
  graph.add(
      pyneat.nodes.h264_encode_sima(
          1280, 720, 30, bitrate_kbps=2500, profile="main", level="4.1"
      )
  )
  graph.add(pyneat.nodes.h264_parse(parse))
  graph.add(pyneat.nodes.h264_packetize(payload_type=98, config_interval=2))
  graph.add(pyneat.nodes.udp_output(udp))

  text = graph.describe_backend().lower()
  assert "neatencoder" in text
  assert "h264parse" in text
  assert "alignment=(string)au" in text
  assert "stream-format=(string)byte-stream" in text
  assert "rtph264pay" in text
  assert "pt=98" in text
  assert "udpsink" in text
  assert "host=10.0.0.5" in text
  assert "port=5500" in text


def test_graph_describe_backend_includes_udp_h264_output_group():
  opt = pyneat.UdpH264OutputGroupOptions()
  opt.h264_caps = 'video/x-h264,profile="high"'
  opt.payload_type = 97
  opt.config_interval = 2
  opt.udp_host = "127.0.0.1"
  opt.udp_port = 5600
  opt.udp_sync = False
  opt.udp_async = False

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input())
  graph.add(pyneat.groups.udp_h264_output_group(opt))

  text = graph.describe_backend().lower()
  assert "h264parse" in text
  assert "capsfilter" in text
  assert 'profile=\\"high\\"' in text
  assert "rtph264pay" in text
  assert "pt=97" in text
  assert "udpsink" in text
  assert "port=5600" in text


def test_model_surface_fixtures_are_real_strict_model_tars():
  for model_path in (_strict_resnet50_model_path(), _strict_yolo_model_path()):
    assert model_path.name.endswith(".tar.gz")
    assert model_fixtures.has_strict_mpk_json(model_path)


def test_postprocess_stage_api_parity_guards_supported_call_surface(tmp_path):
  detess_path = _strict_yolo_model_path()
  box_path = _strict_yolo_model_path()
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


def test_error_code_constants_present():
  assert isinstance(pyneat.ERROR_PIPELINE_SHAPE, str)
  assert isinstance(pyneat.ERROR_RUNTIME_PULL, str)


def test_memory_and_image_type_aliases_present():
  assert pyneat.Memory is pyneat.TensorMemory
  assert pyneat.ImageType is pyneat.PixelFormat


def test_runtime_overload_methods_present():
  assert hasattr(pyneat.Run, "push")
  assert hasattr(pyneat.Run, "try_push")
  assert hasattr(pyneat.Run, "run")
  assert hasattr(pyneat.Graph, "build")
  assert hasattr(pyneat.Graph, "run")
  assert hasattr(pyneat.ModelRunner, "push")
  assert hasattr(pyneat.ModelRunner, "run")
  assert hasattr(pyneat.Model, "build")
  assert hasattr(pyneat.Model, "run")


def test_runtime_tensor_sample_aliases_are_not_public():
  for cls in (pyneat.Run, pyneat.Graph, pyneat.ModelRunner, pyneat.Model):
    assert not hasattr(cls, "run_tensors"), cls
    assert not hasattr(cls, "run_samples"), cls

  for cls in (pyneat.Graph, pyneat.Model):
    assert not hasattr(cls, "build_tensors"), cls
    assert not hasattr(cls, "build_samples"), cls


def test_graph_build_accepts_numpy_without_type_error():
  graph = pyneat.Graph()
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: graph.build([arr]))
  _assert_not_type_error(lambda: graph.build([arr], copy=True))
  _assert_not_type_error(
      lambda: graph.build([arr], layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB)
  )


def test_graph_run_accepts_numpy_without_type_error():
  graph = pyneat.Graph()
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: graph.run([arr]))
  _assert_not_type_error(lambda: graph.run([arr], copy=True))
  _assert_not_type_error(
      lambda: graph.run([arr], layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB)
  )


def test_measure_input_stats_extended_fields_present():
  assert not hasattr(pyneat, "RuntimeInputStreamMetrics")
  s = pyneat.MeasureInputStats()
  assert hasattr(s, "alloc_grows")
  assert hasattr(s, "growth_blocked")
  assert hasattr(s, "renegotiation_blocked")



def test_native_build_overload_marker_present():
  import pyneat._pyneat_core as core

  assert bool(getattr(core, "_HAS_NATIVE_BUILD_OBJECT_OVERLOADS", False))



def test_model_build_accepts_numpy_without_type_error():
  model_path = _strict_resnet50_model_path()
  assert model_path.exists(), f"missing fixture: {model_path}"

  model = pyneat.Model(str(model_path))
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: model.build([arr]))
  _assert_not_type_error(lambda: model.build([arr], copy=True))



def test_graph_build_accepts_torch_without_type_error():
  try:
    import torch
  except Exception:
    return

  graph = pyneat.Graph()
  # Use an intentionally invalid rank to keep this as API-overload coverage
  # only: this should fail fast in conversion/validation, but never with TypeError.
  tensor = torch.zeros((8, 8), dtype=torch.uint8)
  _assert_not_type_error(
      lambda: graph.build(
          [tensor], layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB
      )
  )


def test_model_build_accepts_torch_without_type_error():
  try:
    import torch
  except Exception:
    return

  model_path = _strict_resnet50_model_path()
  assert model_path.exists(), f"missing fixture: {model_path}"

  model = pyneat.Model(str(model_path))
  # Same fast-path API-overload validation strategy as Graph test.
  tensor = torch.zeros((8, 8), dtype=torch.uint8)

  _assert_not_type_error(lambda: model.build([tensor]))


def test_model_build_requires_explicit_image_semantic_for_image_tensor():
  model_path = _strict_resnet50_model_path()
  assert model_path.exists(), f"missing fixture: {model_path}"

  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Image
  opt.preprocess.enable = pyneat.AutoFlag.On
  opt.preprocess.resize.enable = pyneat.AutoFlag.On
  opt.preprocess.normalize.enable = pyneat.AutoFlag.On
  model = pyneat.Model(str(model_path), opt)
  tensor = pyneat.Tensor.from_numpy(np.zeros((8, 8, 3), dtype=np.uint8), copy=True)

  # Model input-contract violations now surface as pyneat.NeatError (was std::invalid_argument
  # auto-translated to ValueError). The message is preserved as the NeatError repro_note, so the
  # substring match still holds, and the structured error_code is populated.
  with pytest.raises(pyneat.NeatError, match="requires explicit image format") as excinfo:
    model.build([tensor])
  assert excinfo.value.error_code == "misconfig.input_shape"


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

  model_path = _strict_resnet50_model_path()
  assert model_path.exists(), f"missing fixture: {model_path}"

  model = pyneat.Model(str(model_path))
  tensor = torch.zeros((3, 8, 8), dtype=torch.uint8)
  _assert_not_type_error(lambda: model.run([tensor], timeout_ms=1))


def test_model_run_build_reject_layout_and_image_format_kwargs():
  model_path = _strict_resnet50_model_path()
  assert model_path.exists(), f"missing fixture: {model_path}"

  model = pyneat.Model(str(model_path))
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  try:
    model.build([arr], layout=pyneat.TensorLayout.HWC)
  except TypeError:
    pass
  else:
    raise AssertionError("expected model.build(..., layout=...) to fail with TypeError")

  try:
    model.run([arr], image_format=pyneat.PixelFormat.RGB, timeout_ms=1)
  except TypeError:
    pass
  else:
    raise AssertionError("expected model.run(..., image_format=...) to fail with TypeError")


def test_graph_video_push_uses_input_format_without_image_semantic():
  graph = pyneat.Graph()
  opt = pyneat.InputOptions()
  opt.payload_type = pyneat.PayloadType.Image
  opt.format = pyneat.Format.RGB
  opt.memory_policy = pyneat.InputMemoryPolicy.SystemMemory
  graph.add(pyneat.nodes.input(opt))
  graph.add(pyneat.nodes.output())

  frame = np.zeros((16, 16, 3), dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(frame, copy=True, memory=pyneat.TensorMemory.CPU)
  run = graph.build([tensor])
  try:
    assert run.push([tensor])
    out = run.pull(1000)
    assert out is not None
  finally:
    run.close_input()
    run.close()


def test_input_options_use_simaai_pool_is_deprecated_compatibility_property():
  opt = pyneat.InputOptions()
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    opt.use_simaai_pool = False
  assert opt.use_simaai_pool is False
  assert any("use_simaai_pool is deprecated" in str(w.message) for w in caught)


def test_graph_error_in_python_exposes_structured_fields():
  graph = pyneat.Graph()
  tensor = pyneat.Tensor.from_numpy(
      np.zeros((8, 8, 3), dtype=np.uint8),
      copy=True,
      memory=pyneat.TensorMemory.CPU,
  )

  try:
    graph.build([tensor])
  except pyneat.NeatError as exc:
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
    raise AssertionError("expected NeatError for empty pipeline")


def test_low_level_runtime_graph_removed_from_python_surface():
  opt = pyneat.RtspDecodedInputOptions()
  opt.url = "rtsp://example.com/live"

  fragment = pyneat.groups.rtsp_decoded_input(opt)
  assert isinstance(fragment, pyneat.Graph)
  with pytest.raises(AttributeError, match="removed"):
    getattr(pyneat, "_graph")


def test_graph_build_rejects_legacy_mode_argument():
  graph = pyneat.Graph()
  tensor = pyneat.Tensor.from_numpy(
      np.zeros((8, 8, 3), dtype=np.uint8),
      copy=True,
      image_format=pyneat.PixelFormat.RGB,
  )

  with pytest.raises(TypeError):
    graph.build([tensor], mode="Sync")
  with pytest.raises(TypeError):
    graph.build([tensor], "Sync", pyneat.RunOptions())


def test_runner_legacy_measurement_methods_are_not_public():
  run = pyneat.Run()
  assert not hasattr(run, "stats")
  assert not hasattr(run, "input_stats")
  assert not hasattr(run, "measurement_summary")
  assert not hasattr(run, "metrics_report")
  assert not hasattr(run, "metrics")
  assert not hasattr(run, "measure")
  assert not hasattr(run, "report")
  assert not hasattr(run, "diag_snapshot")
  assert not hasattr(run, "power_summary")
  assert not hasattr(run, "diagnostics_summary")
  assert not hasattr(pyneat, "RunMode")
  assert not hasattr(pyneat, "InputStreamStats")
  assert not hasattr(pyneat, "RunStats")
  assert not hasattr(pyneat, "RunDiagSnapshot")
  assert not hasattr(pyneat, "RunReportOptions")


def test_model_runner_measurement_surface_matches_run():
  assert hasattr(pyneat.ModelRunner, "start_measurement")
  assert hasattr(pyneat.ModelRunner, "close_input")
  assert not hasattr(pyneat.ModelRunner, "metrics")
  assert not hasattr(pyneat.ModelRunner, "measure")
  assert not hasattr(pyneat.ModelRunner, "warmup")
  assert not hasattr(pyneat.ModelRunner, "stats")
  assert not hasattr(pyneat.ModelRunner, "input_stats")
  assert not hasattr(pyneat.ModelRunner, "measurement_summary")
  assert not hasattr(pyneat.ModelRunner, "metrics_report")
  assert not hasattr(pyneat.ModelRunner, "diag_snapshot")
  assert not hasattr(pyneat.ModelRunner, "report")


def test_measurement_bool_overload_surface():
  _assert_not_type_error(lambda: pyneat.Run().start_measurement(False))
  _assert_not_type_error(lambda: pyneat.Run().start_measurement(True))
  _assert_not_type_error(lambda: pyneat.ModelRunner().start_measurement(False))
  _assert_not_type_error(lambda: pyneat.ModelRunner().start_measurement(True))
