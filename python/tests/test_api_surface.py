import numpy as np

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


def test_input_stage_option_struct_constructors_accept_expected_args():
  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pyneat.Model(str(mpk_path))

  _assert_not_type_error(lambda: pyneat.PreprocOptions())
  _assert_not_type_error(lambda: pyneat.PreprocOptions(model))
  _assert_not_type_error(lambda: pyneat.QuantTessOptions())
  _assert_not_type_error(lambda: pyneat.QuantTessOptions(model))


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


def test_input_stage_node_factories_present_and_accept_expected_args():
  assert hasattr(pyneat.nodes, "preproc")
  assert hasattr(pyneat.nodes, "quant_tess")

  _assert_not_type_error(lambda: pyneat.nodes.preproc())
  _assert_not_type_error(lambda: pyneat.nodes.preproc(pyneat.PreprocOptions()))
  _assert_not_type_error(lambda: pyneat.nodes.quant_tess())
  _assert_not_type_error(lambda: pyneat.nodes.quant_tess(pyneat.QuantTessOptions()))


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
