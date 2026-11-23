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


@pytest.mark.parametrize("copy", [False, True])
def test_bfloat16_numpy_expansion(copy):
  words = np.array([0x3f80, 0xc020, 0x8000, 0x7f80, 0x7fc1, 0x0001], dtype=np.uint16)
  source = np.zeros((3, 4), dtype=np.uint16)
  view = source[:, ::2]
  view[:] = words.reshape(3, 2)
  tensor = pn.Tensor.from_numpy(view, copy=False, memory=pn.TensorMemory.CPU)
  tensor.dtype = pn.TensorDType.BFloat16
  result = tensor.to_numpy(copy=copy)
  assert result.dtype == np.float32
  assert result.shape == view.shape
  source.fill(0)
  np.testing.assert_array_equal(result.view(np.uint32), (words.astype(np.uint32) << 16).reshape(3, 2))


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


# --- Dense geometry that does not fit the storage -------------------------------------------
#
# The tensor wraps the first 16 bytes of a larger NumPy buffer, so a read past the 16-byte
# storage stays inside memory that is valid for the test.


def _cpu_tensor_over_prefix():
  backing = np.arange(1024, dtype=np.uint8) % 251
  tensor = pn.Tensor.from_numpy(backing[:16].reshape(4, 4), copy=False, memory=pn.TensorMemory.CPU)
  assert tensor.storage.size_bytes == 16
  return backing, tensor


def _grow_shape_keep_strides(t):
  t.shape = [8, 8]


def _wide_row_stride(t):
  t.strides_bytes = [100, 1]


def _offset_far_past_end(t):
  t.byte_offset = 1000


def _stride_rank_mismatch(t):
  t.shape = [2, 2, 4]


def _negative_dimension(t):
  t.shape = [4, -4]


_BAD_GEOMETRY = [
    _grow_shape_keep_strides, _wide_row_stride, _offset_far_past_end, _stride_rank_mismatch,
    _negative_dimension
]
_BAD_GEOMETRY_IDS = [f.__name__.lstrip("_") for f in _BAD_GEOMETRY]


@pytest.mark.parametrize("edit", _BAD_GEOMETRY, ids=_BAD_GEOMETRY_IDS)
def test_validate_rejects_dense_geometry_outside_storage(edit):
  _, tensor = _cpu_tensor_over_prefix()
  ok, message = tensor.validate()
  assert ok, message
  edit(tensor)
  ok, message = tensor.validate()
  assert not ok
  assert message


@pytest.mark.parametrize("edit", _BAD_GEOMETRY, ids=_BAD_GEOMETRY_IDS)
def test_to_numpy_refuses_dense_geometry_outside_storage(edit):
  _, tensor = _cpu_tensor_over_prefix()
  edit(tensor)
  with pytest.raises(RuntimeError):
    tensor.to_numpy(copy=False)
  with pytest.raises(RuntimeError):
    tensor.to_numpy(copy=True)
  with pytest.raises(RuntimeError):
    np.from_dlpack(tensor)


@pytest.mark.parametrize("copy", [False, True])
def test_to_numpy_exports_sub_view_inside_storage(copy):
  backing, tensor = _cpu_tensor_over_prefix()
  tensor.shape = [2, 4]
  tensor.byte_offset = 8
  assert tensor.validate()[0]
  np.testing.assert_array_equal(tensor.to_numpy(copy=copy), backing[8:16].reshape(2, 4))


@pytest.mark.parametrize("copy", [False, True])
def test_to_numpy_exports_strided_view_with_gaps(copy):
  view = np.arange(24, dtype=np.int16).reshape(3, 8)[:, ::2]
  tensor = pn.Tensor.from_numpy(view, copy=False, memory=pn.TensorMemory.CPU)
  assert tensor.validate()[0]
  np.testing.assert_array_equal(tensor.to_numpy(copy=copy), view)


def test_zero_element_tensor_validates_and_exports():
  _, tensor = _cpu_tensor_over_prefix()
  tensor.shape = [0, 6]
  tensor.strides_bytes = []
  ok, message = tensor.validate()
  assert ok, message
  assert tensor.to_numpy(copy=False).shape == (0, 6)


def test_rank_zero_scalar_validates_and_exports():
  backing = np.arange(8, dtype=np.uint8)
  tensor = pn.Tensor.from_numpy(backing.view(np.int32).reshape(2), copy=False,
                                memory=pn.TensorMemory.CPU)
  tensor.shape = []
  tensor.strides_bytes = []
  tensor.byte_offset = 4
  ok, message = tensor.validate()
  assert ok, message
  assert tensor.to_numpy(copy=False).item() == backing.view(np.int32)[1]
