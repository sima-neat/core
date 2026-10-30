import gc
import pytest
import numpy as np

import pyneat as pn


def test_output_placement_options():
  options = pn.OutputOptions.latest()
  assert options.placement == pn.OutputPlacement.Auto
  options.placement = pn.OutputPlacement.EV74Compatible
  route = pn.ModelRouteOptions()
  assert route.output.placement == pn.OutputPlacement.Auto
  route.output = options
  assert route.output.placement == pn.OutputPlacement.EV74Compatible
  assert route.output.max_buffers == 1 and route.output.drop
  route.output.max_buffers = 7
  route.output.placement = pn.OutputPlacement.Auto
  assert route.output.max_buffers == 7
  assert route.output.placement == pn.OutputPlacement.Auto
  assert options.max_buffers == 1
  assert options.placement == pn.OutputPlacement.EV74Compatible


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


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("copy", [False, True])
def test_rank_zero_dlpack_copy_and_lifetime(dtype, copy):
  source = np.array(37, dtype=dtype)
  tensor = pn.Tensor.from_dlpack(source, copy=copy, memory=pn.TensorMemory.CPU)
  cloned = tensor.clone()
  expected = source.tobytes()
  assert bytes(tensor.copy_dense_bytes_tight()) == expected
  assert bytes(tensor.copy_payload_bytes()) == expected
  assert bytes(cloned.copy_dense_bytes_tight()) == expected
  source[...] = 91
  assert tensor.to_numpy(copy=False).shape == ()
  assert tensor.to_numpy(copy=False).item() == (37 if copy else 91)
  exported = cloned.to_numpy(copy=False)
  del tensor, cloned, source
  gc.collect()
  assert exported.shape == ()
  assert exported.dtype == dtype
  assert exported.item() == 37


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


def test_explicit_tiled_storage_rejects_dense_exports():
  stored = np.arange(32, dtype=np.uint8).reshape(1, 1, 2, 16)
  tensor = pn.Tensor.from_numpy(stored, copy=False, memory=pn.TensorMemory.CPU)
  tess = pn.TessSpec()
  tess.format = "MLA_TILED_UINT8_PAD16"
  tess.slice_shape = [1, 1, 1, 16]
  tess.axes = [0, 1, 2, 3]
  tess.storage_bytes = stored.nbytes
  semantic = tensor.semantic
  semantic.tess = tess
  tensor.semantic = semantic

  assert tensor.has_tiled_storage()
  assert not tensor.is_contiguous()
  for export in (tensor.__dlpack__, tensor.contiguous,
                 tensor.copy_dense_bytes_tight, lambda: tensor.to_numpy(copy=False)):
    with pytest.raises(RuntimeError, match="tiled storage"):
      export()
  clone = tensor.clone()
  assert clone.has_tiled_storage()
  assert clone.semantic.tess.axes == tess.axes
  assert bytes(clone.copy_payload_bytes()) == stored.tobytes()

  # Historical dtype-format annotations alone are not physical tile metadata.
  tess.storage_bytes = 0
  semantic.tess = tess
  tensor.semantic = semantic
  assert not tensor.has_tiled_storage()
  np.testing.assert_array_equal(tensor.to_numpy(copy=False), stored)
