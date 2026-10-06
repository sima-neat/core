import gc
import pytest
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


@pytest.mark.parametrize("memory", [pn.TensorMemory.CPU, pn.TensorMemory.A65,
                                   pn.TensorMemory.EV74, pn.TensorMemory.MLA])
def test_tensor_copy_placement_isolates_strided_source(memory):
  source = np.arange(120, dtype=np.float32).reshape(10, 12)
  view = source[1:9:2, 2:12:2]
  expected = view.copy()
  tensor = pn.Tensor.from_numpy(view, copy=True, memory=memory)
  source.fill(-99)
  del view, source
  gc.collect()
  np.testing.assert_array_equal(tensor.to_numpy(copy=False), expected)
  assert not tensor.read_only


@pytest.mark.parametrize("memory", [pn.TensorMemory.CPU, pn.TensorMemory.EV74,
                                   pn.TensorMemory.MLA])
def test_numpy_export_outlives_tensor_and_source(memory):
  source = np.arange(64, dtype=np.float32).reshape(8, 8)
  expected = source.copy()
  tensor = pn.Tensor.from_numpy(source, copy=True, memory=memory)
  exported = tensor.to_numpy(copy=False)
  del tensor, source
  gc.collect()
  np.testing.assert_array_equal(exported, expected)


def test_concurrent_device_import_and_export_keep_owners():
  from concurrent.futures import ThreadPoolExecutor

  def roundtrip(seed):
    for offset in range(16):
      source = np.arange(2048, dtype=np.float32) + seed + offset
      expected = source.copy()
      tensor = pn.Tensor.from_numpy(source, copy=True, memory=pn.TensorMemory.EV74)
      source.fill(-1)
      exported = tensor.to_numpy(copy=False)
      del tensor, source
      np.testing.assert_array_equal(exported, expected)

  with ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(roundtrip, range(4)))
