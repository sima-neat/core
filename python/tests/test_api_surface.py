import io
import json
import re
import tarfile

import numpy as np
import pytest

import pyneat

PREPROC_OPTION_FIELDS = (
    "input_width",
    "input_height",
    "output_width",
    "output_height",
    "scaled_width",
    "scaled_height",
    "input_channels",
    "output_channels",
    "batch_size",
    "normalize",
    "aspect_ratio",
    "tessellate",
    "dynamic_input_dims",
    "tile_width",
    "tile_height",
    "tile_channels",
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
    "output_memory_order",
    "num_buffers",
    "num_buffers_model",
    "num_buffers_locked",
    "config_path",
    "config_dir",
    "keep_config",
    "config_json",
)

QUANT_TESS_OPTION_FIELDS = (
    "config_path",
    "config_dir",
    "keep_config",
    "config_json",
    "element_name",
    "num_buffers",
    "num_buffers_model",
    "num_buffers_locked",
)

DETESS_DEQUANT_OPTION_FIELDS = (
    "config_path",
    "config_dir",
    "keep_config",
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


def _assert_not_type_error(call):
  try:
    call()
  except Exception as exc:
    assert not isinstance(exc, TypeError), str(exc)


def _write_tar_text(tar, name, text):
  data = text.encode("utf-8")
  info = tarfile.TarInfo(name=name)
  info.size = len(data)
  tar.addfile(info, io.BytesIO(data))


def _write_tar_bytes(tar, name, data):
  info = tarfile.TarInfo(name=name)
  info.size = len(data)
  tar.addfile(info, io.BytesIO(data))


def _write_mpk_fixture(tmp_path, name, files):
  tar_path = tmp_path / f"{name}.tar.gz"
  with tarfile.open(tar_path, "w:gz") as tar:
    for path, contents in files.items():
      _write_tar_text(tar, path, contents)
    _write_tar_bytes(tar, "share/placeholder.elf", bytes((0x7F, 0x45, 0x4C, 0x46, 0x02, 0x01, 0x01)))
  return tar_path


def _postprocess_fixture_path(tmp_path, name, *, include_boxdecode=False, include_detess=False):
  sequence = [
      {
          "sequence_id": 1,
          "name": "preproc_0",
          "pluginId": "processcvu",
          "configPath": "0_preproc.json",
          "processor": "CVU",
          "kernel": "preproc",
          "input": "decoder",
      },
      {
          "sequence_id": 2,
          "name": "mla_0",
          "pluginId": "processmla",
          "configPath": "0_process_mla.json",
          "processor": "MLA",
          "kernel": "infer",
          "input": "preproc_0",
      },
  ]
  files = {
      "etc/pipeline_sequence.json": json.dumps({"pipelines": [{"sequence": sequence}]}, indent=2),
      "etc/0_preproc.json": json.dumps(
          {
              "node_name": "preproc_0",
              "input_width": 1280,
              "input_height": 720,
              "input_img_type": "RGB",
              "output_width": 640,
              "output_height": 640,
              "output_img_type": "RGB",
          },
          indent=2,
      ),
      "etc/0_process_mla.json": json.dumps(
          {
              "node_name": "mla_0",
              "input_buffers": [{"name": "preproc_0"}],
              "data_type": ["INT8"],
              "output_width": [80],
              "output_height": [80],
              "output_depth": [6],
          },
          indent=2,
      ),
  }

  if include_boxdecode:
    sequence.append(
        {
            "sequence_id": 3,
            "name": "boxdecode_0",
            "pluginId": "processcvu",
            "configPath": "0_boxdecode.json",
            "processor": "CVU",
            "kernel": "boxdecode",
            "input": "mla_0",
        }
    )
    files["etc/0_boxdecode.json"] = json.dumps(
        {
            "node_name": "boxdecode_0",
            "input_buffers": [{"name": "mla_0"}],
            "decode_type": "yolov8",
            "original_width": 320,
            "original_height": 240,
            "detection_threshold": 0.15,
            "nms_iou_threshold": 0.45,
            "topk": 24,
        },
        indent=2,
    )

  if include_detess:
    sequence.append(
        {
            "sequence_id": 3,
            "name": "detessdequant_0",
            "pluginId": "processcvu",
            "configPath": "0_postproc.json",
            "processor": "CVU",
            "kernel": "detessdequant",
            "input": "mla_0",
        }
    )
    files["etc/0_postproc.json"] = json.dumps(
        {
            "node_name": "detessdequant_0",
            "input_buffers": [{"name": "mla_0"}],
            "memory": {"cpu": "CVU", "next_cpu": "CVU"},
            "simaai__params": {"cpu": "CVU", "next_cpu": "CVU"},
        },
        indent=2,
    )

  return _write_mpk_fixture(tmp_path, name, files)


def _preproc_fixture_path(tmp_path, name, *, normalize=True):
  files = {
      "etc/pipeline_sequence.json": json.dumps(
          {
              "pipelines": [
                  {
                      "sequence": [
                          {
                              "sequence_id": 1,
                              "name": "preproc_0",
                              "pluginId": "processcvu",
                              "configPath": "0_preproc.json",
                              "processor": "CVU",
                              "kernel": "preproc",
                              "input": "decoder",
                          },
                          {
                              "sequence_id": 2,
                              "name": "mla_0",
                              "pluginId": "processmla",
                              "configPath": "0_process_mla.json",
                              "processor": "MLA",
                              "kernel": "infer",
                              "input": "preproc_0",
                          },
                      ]
                  }
              ]
          },
          indent=2,
      ),
      "etc/0_preproc.json": json.dumps(
          {
              "node_name": "preproc_0",
              "input_width": 1280,
              "input_height": 720,
              "input_img_type": "RGB",
              "output_width": 640,
              "output_height": 640,
              "output_img_type": "RGB",
              "normalize": normalize,
          },
          indent=2,
      ),
      "etc/0_process_mla.json": json.dumps(
          {
              "node_name": "mla_0",
              "input_buffers": [{"name": "preproc_0"}],
              "data_type": ["INT8"],
              "output_width": [80],
              "output_height": [80],
              "output_depth": [6],
          },
          indent=2,
      ),
  }
  return _write_mpk_fixture(tmp_path, name, files)


def test_session_pythonic_add_and_describe():
  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.nodes.output())

  text = session.describe_backend()
  assert isinstance(text, str)
  assert text


def test_model_option_structs_are_mutable():
  opt = pyneat.ModelOptions()
  opt.media_type = "video/x-raw"
  opt.format = "RGB"
  opt.input_max_width = 1920
  opt.input_max_height = 1080

  assert opt.media_type == "video/x-raw"
  assert opt.format == "RGB"
  assert opt.input_max_width == 1920
  assert opt.input_max_height == 1080


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

  for field in UDP_OUTPUT_OPTION_FIELDS:
    assert hasattr(udp, field), field
  assert hasattr(udp, "async"), "async"

  for field in H264_PARSE_OPTION_FIELDS:
    assert hasattr(parse, field), field

  for field in UDP_H264_OUTPUT_GROUP_OPTION_FIELDS:
    assert hasattr(group, field), field

  assert hasattr(pyneat, "H264ParseAlignment")
  assert hasattr(pyneat, "H264ParseStreamFormat")


def test_input_stage_option_struct_constructors_accept_expected_args():
  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))

  _assert_not_type_error(lambda: pyneat.PreprocOptions())
  _assert_not_type_error(lambda: pyneat.PreprocOptions(model))
  _assert_not_type_error(lambda: pyneat.QuantTessOptions())
  _assert_not_type_error(lambda: pyneat.QuantTessOptions(model))


def test_postprocess_stage_option_struct_constructors_accept_expected_args(tmp_path):
  mpk_path = _postprocess_fixture_path(tmp_path, "detess_valid", include_detess=True)
  model = pyneat.Model(str(mpk_path))

  _assert_not_type_error(lambda: pyneat.DetessDequantOptions())
  _assert_not_type_error(lambda: pyneat.DetessDequantOptions(model))


def test_output_stage_option_struct_constructors_accept_expected_args():
  _assert_not_type_error(lambda: pyneat.UdpOutputOptions())
  _assert_not_type_error(lambda: pyneat.H264ParseOptions())
  _assert_not_type_error(lambda: pyneat.UdpH264OutputGroupOptions())


def test_input_stage_option_structs_are_mutable():
  pre = pyneat.PreprocOptions()
  pre.input_width = 320
  pre.input_height = 240
  pre.output_memory_order = ["SystemMemory", "SimaAI"]
  pre.keep_config = True
  pre.config_json = {"graph_name": "preproc", "input_width": 320}

  quant_tess = pyneat.QuantTessOptions()
  quant_tess.element_name = "qt"
  quant_tess.keep_config = True
  quant_tess.config_json = {"node_name": "quanttess"}

  assert pre.input_width == 320
  assert pre.input_height == 240
  assert pre.output_memory_order == ["SystemMemory", "SimaAI"]
  assert pre.keep_config is True
  assert pre.config_json["graph_name"] == "preproc"
  pre.config_json = None
  assert pre.config_json is None

  assert quant_tess.element_name == "qt"
  assert quant_tess.keep_config is True
  assert quant_tess.config_json["node_name"] == "quanttess"
  quant_tess.config_json = None
  assert quant_tess.config_json is None


def test_postprocess_stage_option_structs_are_mutable():
  detess = pyneat.DetessDequantOptions()
  detess.config_path = "/tmp/detess.json"
  detess.config_dir = "/tmp"
  detess.keep_config = True
  detess.config_json = {"node_name": "detessdequant_0"}
  detess.upstream_name = "mla_0"
  detess.element_name = "detessdequant_0"
  detess.num_buffers = 4
  detess.num_buffers_model = 4
  detess.num_buffers_locked = True

  assert detess.config_path == "/tmp/detess.json"
  assert detess.config_dir == "/tmp"
  assert detess.keep_config is True
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


def test_input_stage_node_factories_present_and_accept_expected_args():
  assert hasattr(pyneat.nodes, "preproc")
  assert hasattr(pyneat.nodes, "quant_tess")

  _assert_not_type_error(lambda: pyneat.nodes.preproc())
  _assert_not_type_error(lambda: pyneat.nodes.preproc(pyneat.PreprocOptions()))
  _assert_not_type_error(lambda: pyneat.nodes.quant_tess())
  _assert_not_type_error(lambda: pyneat.nodes.quant_tess(pyneat.QuantTessOptions()))


def test_postprocess_stage_node_factories_present_and_accept_expected_args(tmp_path):
  mpk_path = _postprocess_fixture_path(tmp_path, "boxdecode_valid", include_boxdecode=True)
  model = pyneat.Model(str(mpk_path))

  assert hasattr(pyneat.nodes, "detess_dequant")
  assert hasattr(pyneat.nodes, "sima_box_decode")

  _assert_not_type_error(lambda: pyneat.nodes.detess_dequant())
  _assert_not_type_error(lambda: pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions()))
  _assert_not_type_error(lambda: pyneat.nodes.sima_box_decode(model))
  _assert_not_type_error(
      lambda: pyneat.nodes.sima_box_decode(
          model,
          decode_type="yolov8",
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
      pyneat.groups.udp_h264_output_group(pyneat.UdpH264OutputGroupOptions()), pyneat.NodeGroup
  )


def test_mla_group_helper_present_and_accepts_model():
  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))

  assert hasattr(pyneat.groups, "mla")
  _assert_not_type_error(lambda: pyneat.groups.mla(model))
  assert isinstance(pyneat.groups.mla(model), pyneat.NodeGroup)


def test_session_describe_backend_includes_preproc_stage():
  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.nodes.preproc())
  session.add(pyneat.nodes.output())

  text = session.describe_backend().lower()
  assert "preproc" in text


def test_session_describe_backend_includes_quant_tess_stage():
  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.nodes.quant_tess())
  session.add(pyneat.nodes.output())

  text = session.describe_backend().lower()
  assert "quanttess" in text or "quant_tess" in text


def test_session_describe_backend_includes_detess_dequant_stage(tmp_path):
  mpk_path = _postprocess_fixture_path(tmp_path, "detess_backend", include_detess=True)
  model = pyneat.Model(str(mpk_path))

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.mla(model))
  session.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
  session.add(pyneat.nodes.output())

  text = session.describe_backend().lower()
  assert "detessdequant" in text


def test_session_describe_backend_includes_sima_box_decode_stage(tmp_path):
  mpk_path = _postprocess_fixture_path(tmp_path, "boxdecode_backend", include_boxdecode=True)
  model = pyneat.Model(str(mpk_path))

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.mla(model))
  session.add(pyneat.nodes.sima_box_decode(model))
  session.add(pyneat.nodes.output())

  text = session.describe_backend().lower()
  assert "boxdecode" in text


def _boxdecode_backend_config_path(text: str) -> str:
  for line in text.splitlines():
    if "boxdecode" not in line.lower():
      continue
    match = re.search(r'config="([^"]+)"', line)
    if match:
      return match.group(1)
  raise AssertionError("failed to locate boxdecode config path in backend description")


def _preproc_backend_config_path(text: str) -> str:
  for line in text.splitlines():
    if "preproc" not in line.lower():
      continue
    match = re.search(r'config="?([^" ]+)"?', line)
    if match:
      return match.group(1)
  raise AssertionError("failed to locate preproc config path in backend description")


def test_model_preproc_normalize_unset_preserves_model_pack_value(tmp_path):
  mpk_path = _preproc_fixture_path(tmp_path, "preproc_normalize_default_true", normalize=True)
  model = pyneat.Model(str(mpk_path))

  session = pyneat.Session()
  session.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  session.add(model.preprocess())
  session.add(pyneat.nodes.output())

  backend = session.describe_backend()
  cfg_path = _preproc_backend_config_path(backend)
  with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

  assert cfg["normalize"] is True


def test_model_preproc_normalize_false_overrides_model_pack_value(tmp_path):
  mpk_path = _preproc_fixture_path(tmp_path, "preproc_normalize_override_false", normalize=True)
  opt = pyneat.ModelOptions()
  opt.preproc.normalize = False
  model = pyneat.Model(str(mpk_path), opt)

  session = pyneat.Session()
  session.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  session.add(model.preprocess())
  session.add(pyneat.nodes.output())

  backend = session.describe_backend()
  cfg_path = _preproc_backend_config_path(backend)
  with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

  assert cfg["normalize"] is False


def test_sima_box_decode_without_runtime_dims_uses_model_pack_defaults(tmp_path):
  mpk_path = _postprocess_fixture_path(tmp_path, "boxdecode_defaults", include_boxdecode=True)
  model = pyneat.Model(str(mpk_path))

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.mla(model))
  session.add(pyneat.nodes.sima_box_decode(model))
  session.add(pyneat.nodes.output())

  backend = session.describe_backend()
  cfg_path = _boxdecode_backend_config_path(backend)
  with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

  assert "detection-threshold=" not in backend.lower()
  assert "nms-iou-threshold=" not in backend.lower()
  assert "topk=" not in backend.lower()
  assert cfg["original_width"] == 320
  assert cfg["original_height"] == 240
  assert cfg["detection_threshold"] == pytest.approx(0.15)
  assert cfg["nms_iou_threshold"] == pytest.approx(0.45)
  assert cfg["topk"] == 24


def test_sima_box_decode_runtime_dims_override_backend_config(tmp_path):
  mpk_path = _postprocess_fixture_path(tmp_path, "boxdecode_override", include_boxdecode=True)
  model = pyneat.Model(str(mpk_path))

  session = pyneat.Session()
  session.add(pyneat.nodes.input())
  session.add(pyneat.groups.mla(model))
  session.add(
      pyneat.nodes.sima_box_decode(
          model,
          decode_type="yolov8",
          original_width=640,
          original_height=360,
          detection_threshold=0.25,
          nms_iou_threshold=0.55,
          top_k=120,
      )
  )
  session.add(pyneat.nodes.output())

  backend = session.describe_backend()
  cfg_path = _boxdecode_backend_config_path(backend)
  with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

  assert "detection-threshold=0.25" in backend.lower()
  assert "nms-iou-threshold=0.55" in backend.lower()
  assert "topk=120" in backend.lower()
  assert cfg["original_width"] == 640
  assert cfg["original_height"] == 360
  # Runtime threshold/top-k overrides are emitted as backend properties, while
  # the rewritten config keeps the packaged postprocess defaults.
  assert cfg["detection_threshold"] == pytest.approx(0.15)
  assert cfg["nms_iou_threshold"] == pytest.approx(0.45)
  assert cfg["topk"] == 24


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


def test_postprocess_stage_missing_model_config_reports_clear_error():
  model = pyneat.Model(str(_basic_valid_mpk_path()))

  with pytest.raises(RuntimeError, match="boxdecode"):
    pyneat.nodes.sima_box_decode(model)

  with pytest.raises(RuntimeError, match="DetessDequant: config not found"):
    pyneat.DetessDequantOptions(model)


def test_postprocess_stage_api_parity_guards_supported_call_surface(tmp_path):
  detess_path = _postprocess_fixture_path(tmp_path, "detess_signature", include_detess=True)
  box_path = _postprocess_fixture_path(tmp_path, "box_signature", include_boxdecode=True)
  detess_model = pyneat.Model(str(detess_path))
  box_model = pyneat.Model(str(box_path))

  _assert_not_type_error(lambda: pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(detess_model)))
  _assert_not_type_error(lambda: pyneat.nodes.sima_box_decode(box_model, "yolov8", 640, 640, 0.25, 0.55, 120))

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


def test_session_build_ambiguous_3d_layout_requires_explicit_layout():
  session = pyneat.Session()
  opt = pyneat.InputOptions()
  opt.media_type = "application/vnd.simaai.tensor"
  opt.format = "FP32"
  session.add(pyneat.nodes.input(opt))
  session.add(pyneat.nodes.output())

  arr = np.zeros((3, 8, 3), dtype=np.float32)

  try:
    session.build(arr)
  except Exception as exc:
    assert "layout" in str(exc).lower()
  else:
    raise AssertionError("expected ambiguous 3D layout to fail without explicit layout")


def test_native_build_overload_marker_present():
  import pyneat._pyneat_core as core

  assert bool(getattr(core, "_HAS_NATIVE_BUILD_OBJECT_OVERLOADS", False))


def _basic_valid_mpk_path():
  from pathlib import Path

  root = Path(__file__).resolve().parents[2]
  fixture_root = root / "tests" / "assets" / "mpk" / "valid"
  mpk = fixture_root / "basic_valid.mpk"
  if mpk.exists():
    return mpk
  return fixture_root / "basic_valid.tar"


def test_model_build_accepts_numpy_without_type_error():
  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: model.build(arr))
  _assert_not_type_error(lambda: model.build(arr, copy=True))


def test_model_build_ambiguous_3d_layout_reports_layout_error():
  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))
  arr = np.zeros((3, 8, 3), dtype=np.float32)

  try:
    model.build(arr)
  except Exception as exc:
    assert "layout" in str(exc).lower()
  else:
    raise AssertionError("expected ambiguous 3D layout to fail without explicit layout")


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

  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))
  # Same fast-path API-overload validation strategy as Session test.
  tensor = torch.zeros((8, 8), dtype=torch.uint8)

  _assert_not_type_error(lambda: model.build(tensor))


def test_model_run_accepts_chw_torch_without_layout_or_image_format():
  try:
    import torch
  except Exception:
    return

  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))
  tensor = torch.zeros((3, 8, 8), dtype=torch.uint8)
  _assert_not_type_error(lambda: model.run(tensor, timeout_ms=1))


def test_model_run_build_reject_layout_and_image_format_kwargs():
  mpk_path = _basic_valid_mpk_path()
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
  opt = pyneat.InputOptions()
  opt.media_type = "video/x-raw"
  opt.format = "RGB"
  opt.use_simaai_pool = False
  session.add(pyneat.nodes.input(opt))
  session.add(pyneat.nodes.output())

  bad = np.zeros((8, 8, 3), dtype=np.float32)
  try:
    session.build(bad)
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
    raise AssertionError("expected SessionError for invalid video tensor dtype")


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
