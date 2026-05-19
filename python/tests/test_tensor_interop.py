import numpy as np

import pyneat as pn


def test_tensor_from_numpy_cpu_zero_copy_roundtrip():
  arr = np.arange(12, dtype=np.float32).reshape(3, 4)
  tensor = pn.Tensor.from_numpy(arr, copy=False, memory=pn.TensorMemory.CPU)

  out = tensor.to_numpy(copy=False)
  assert out.shape == (3, 4)
  assert out.dtype == np.float32
  np.testing.assert_array_equal(out, arr)

  arr[0, 0] = 123.0
  out2 = tensor.to_numpy(copy=False)
  assert float(out2[0, 0]) == 123.0


def test_tensor_from_numpy_copy_isolation():
  arr = np.arange(6, dtype=np.int32).reshape(2, 3)
  tensor = pn.Tensor.from_numpy(arr, copy=True)

  assert tensor.device.type == pn.DeviceType.SIMA_CVU

  arr[0, 0] = -999
  out = tensor.to_numpy(copy=False)
  assert int(out[0, 0]) != -999


def test_tensor_from_dlpack_generic():
  arr = np.arange(8, dtype=np.uint8).reshape(2, 4)
  tensor = pn.Tensor.from_dlpack(arr, copy=False, memory=pn.TensorMemory.CPU)
  out = tensor.to_numpy(copy=False)
  np.testing.assert_array_equal(out, arr)
