import os

import numpy as np
import pytest

import pyneatpcie as pcie


def test_options_surface():
  conn = pcie.ConnectionOptions(card_host="10.0.0.2", card_id=0, user="sima", queue=0)
  assert conn.card_host == "10.0.0.2"
  assert conn.card_id == 0
  assert conn.user == "sima"
  assert conn.queue == 0

  opt = pcie.ModelOptions()
  opt.preprocess.kind = pcie.InputKind.Image
  opt.preprocess.color_convert.input_format = pcie.ColorFormat.BGR
  opt.preprocess.resize.enable = pcie.AutoFlag.On
  opt.decode_type = pcie.BoxDecodeType.YoloV8

  assert opt.preprocess.kind == pcie.InputKind.Image
  assert opt.preprocess.color_convert.input_format == pcie.ColorFormat.BGR
  assert opt.decode_type == pcie.BoxDecodeType.YoloV8
  assert not hasattr(pcie, "SimaPCIeHost")


def test_tensor_numpy_round_trip():
  image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape((4, 5, 3))
  tensor = pcie.Tensor.from_numpy(
      image,
      image_format=pcie.PixelFormat.BGR,
      route_name="input_image",
  )

  assert tensor.dtype == pcie.TensorDType.UInt8
  assert tensor.layout == pcie.TensorLayout.HWC
  assert tensor.shape == [4, 5, 3]
  assert tensor.strides_bytes == [15, 3, 1]
  assert tensor.image_format == pcie.PixelFormat.BGR
  assert tensor.route.name == "input_image"
  assert tensor.size_bytes == image.nbytes
  np.testing.assert_array_equal(tensor.to_numpy(), image)


def test_tensor_from_numpy_default_is_zero_copy():
  image = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
  tensor = pcie.Tensor.from_numpy(image, route_name="view")
  image[0, 0] = 42.0

  assert tensor.dtype == pcie.TensorDType.Float32
  assert tensor.route.name == "view"
  assert tensor.shape == [2, 3]
  assert tensor.strides_bytes == [12, 4]
  assert tensor.size_bytes == image.nbytes
  np.testing.assert_array_equal(tensor.to_numpy(), image)


def test_tensor_from_numpy_copy_true_copies():
  image = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
  tensor = pcie.Tensor.from_numpy(image, copy=True, route_name="copy")
  image[:] = -1

  assert tensor.dtype == pcie.TensorDType.Float32
  assert tensor.route.name == "copy"
  np.testing.assert_array_equal(
      tensor.to_numpy(),
      np.arange(2 * 3, dtype=np.float32).reshape((2, 3)),
  )


def test_tensor_from_numpy_default_rejects_noncontiguous():
  image = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
  noncontiguous = image[:, ::2]
  try:
    pcie.Tensor.from_numpy(noncontiguous)
    assert False, "from_numpy(copy=False) should reject non-contiguous arrays"
  except RuntimeError:
    pass

  tensor = pcie.Tensor.from_numpy(noncontiguous, copy=True)
  np.testing.assert_array_equal(tensor.to_numpy(), np.ascontiguousarray(noncontiguous))


def test_tensor_from_bytes():
  tensor = pcie.Tensor.from_bytes(
      b"\x00\x01\x02\x03",
      pcie.TensorDType.UInt8,
      [2, 2],
      route_name="bytes",
  )

  assert tensor.shape == [2, 2]
  assert tensor.route.name == "bytes"
  np.testing.assert_array_equal(tensor.to_numpy(), np.array([[0, 1], [2, 3]], dtype=np.uint8))


def test_model_requires_build_before_io():
  # A full model archive is needed by the constructor. The metadata-path test covers that
  # when SIMAPCIE_YOLOV8_MODEL is available.
  model_path = os.environ.get("SIMAPCIE_YOLOV8_MODEL", "")
  if not model_path:
    pytest.skip("SIMAPCIE_YOLOV8_MODEL is not set")

  model = pcie.Model(model_path)
  with pytest.raises(RuntimeError, match="PCIe model is not built; call model.build"):
    model.push(pcie.Tensor())
  with pytest.raises(RuntimeError, match="PCIe model is not built; call model.build"):
    model.pull(1)

  with pcie.Model(model_path) as scoped:
    assert not scoped.running()
