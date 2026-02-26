import numpy as np

import pyneat as pn


def _assert_not_type_error(call):
  try:
    call()
  except Exception as exc:
    assert not isinstance(exc, TypeError), str(exc)


def test_session_pythonic_add_and_describe():
  session = pn.Session()
  session.add(pn.nodes.input())
  session.add(pn.nodes.output())

  text = session.describe_backend()
  assert isinstance(text, str)
  assert text


def test_model_option_structs_are_mutable():
  opt = pn.ModelOptions()
  opt.media_type = "video/x-raw"
  opt.format = "RGB"
  opt.input_max_width = 1920
  opt.input_max_height = 1080

  assert opt.media_type == "video/x-raw"
  assert opt.format == "RGB"
  assert opt.input_max_width == 1920
  assert opt.input_max_height == 1080


def test_error_code_constants_present():
  assert isinstance(pn.ERROR_PIPELINE_SHAPE, str)
  assert isinstance(pn.ERROR_RUNTIME_PULL, str)


def test_runtime_overload_methods_present():
  assert hasattr(pn.Run, "push")
  assert hasattr(pn.Run, "try_push")
  assert hasattr(pn.Run, "run")
  assert hasattr(pn.ModelRunner, "push")
  assert hasattr(pn.ModelRunner, "run")
  assert hasattr(pn.Model, "run")


def test_session_build_accepts_numpy_without_type_error():
  session = pn.Session()
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: session.build(arr))
  _assert_not_type_error(lambda: session.build(arr, copy=True))
  _assert_not_type_error(
      lambda: session.build(arr, layout=pn.TensorLayout.HWC, image_format=pn.PixelFormat.RGB)
  )


def test_inputstream_stats_extended_fields_present():
  s = pn.InputStreamStats()
  assert hasattr(s, "alloc_grows")
  assert hasattr(s, "growth_blocked")
  assert hasattr(s, "renegotiation_blocked")


def test_session_build_ambiguous_3d_layout_requires_explicit_layout():
  session = pn.Session()
  opt = pn.InputOptions()
  opt.media_type = "application/vnd.simaai.tensor"
  opt.format = "FP32"
  session.add(pn.nodes.input(opt))
  session.add(pn.nodes.output())

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

  model = pn.Model(str(mpk_path))
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: model.build(arr))
  _assert_not_type_error(lambda: model.build(arr, copy=True))


def test_model_build_ambiguous_3d_layout_reports_layout_error():
  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pn.Model(str(mpk_path))
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

  session = pn.Session()
  # Use an intentionally invalid rank to keep this as API-overload coverage
  # only: this should fail fast in conversion/validation, but never with TypeError.
  tensor = torch.zeros((8, 8), dtype=torch.uint8)
  _assert_not_type_error(
      lambda: session.build(tensor, layout=pn.TensorLayout.HWC, image_format=pn.PixelFormat.RGB)
  )


def test_model_build_accepts_torch_without_type_error():
  try:
    import torch
  except Exception:
    return

  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pn.Model(str(mpk_path))
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

  model = pn.Model(str(mpk_path))
  tensor = torch.zeros((3, 8, 8), dtype=torch.uint8)
  _assert_not_type_error(lambda: model.run(tensor, timeout_ms=1))


def test_model_run_build_reject_layout_and_image_format_kwargs():
  mpk_path = _basic_valid_mpk_path()
  assert mpk_path.exists(), f"missing fixture: {mpk_path}"

  model = pn.Model(str(mpk_path))
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  try:
    model.build(arr, layout=pn.TensorLayout.HWC)
  except TypeError:
    pass
  else:
    raise AssertionError("expected model.build(..., layout=...) to fail with TypeError")

  try:
    model.run(arr, image_format=pn.PixelFormat.RGB, timeout_ms=1)
  except TypeError:
    pass
  else:
    raise AssertionError("expected model.run(..., image_format=...) to fail with TypeError")


def test_session_video_push_uses_input_format_without_image_semantic():
  session = pn.Session()
  opt = pn.InputOptions()
  opt.media_type = "video/x-raw"
  opt.format = "RGB"
  opt.use_simaai_pool = False
  session.add(pn.nodes.input(opt))
  session.add(pn.nodes.output())

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
  session = pn.Session()
  opt = pn.InputOptions()
  opt.media_type = "video/x-raw"
  opt.format = "RGB"
  opt.use_simaai_pool = False
  session.add(pn.nodes.input(opt))
  session.add(pn.nodes.output())

  bad = np.zeros((8, 8, 3), dtype=np.float32)
  try:
    session.build(bad)
  except pn.SessionError as exc:
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
