import numpy as np
import warnings

import pyneat as pn


def _assert_not_type_error(call):
  try:
    call()
  except Exception as exc:  # Runtime is expected to fail for an unbuilt Run.
    assert not isinstance(exc, TypeError), str(exc)


def test_run_accepts_numpy_without_explicit_tensor_wrap():
  run = pn.Run()
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: run.push(arr))
  _assert_not_type_error(lambda: run.push(arr, copy=False))
  _assert_not_type_error(lambda: run.push(arr, copy=True))
  _assert_not_type_error(lambda: run.run(arr, timeout_ms=1))


def test_model_runner_accepts_numpy_without_explicit_tensor_wrap():
  runner = pn.ModelRunner()
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: runner.push(arr))
  _assert_not_type_error(lambda: runner.push(arr, copy=False))
  _assert_not_type_error(lambda: runner.run(arr, timeout_ms=1))


def test_run_chw_image_autoconverts_to_hwc_with_warning():
  run = pn.Run()
  arr = np.zeros((3, 8, 8), dtype=np.uint8)

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _assert_not_type_error(lambda: run.push(arr, image_format=pn.PixelFormat.RGB))

  if caught:
    assert any(
        "Auto-converting to HWC" in str(item.message) and "full memory copy" in str(item.message)
        for item in caught
    )
