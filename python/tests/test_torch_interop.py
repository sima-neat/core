import pytest

import pyneat as pn


def test_tensor_torch_roundtrip_cpu():
  torch = pytest.importorskip("torch")

  src = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)
  tensor = pn.Tensor.from_torch(src, copy=False, memory=pn.TensorMemory.CPU)
  out = tensor.to_torch(copy=False)

  assert out.shape == src.shape
  assert out.dtype == src.dtype
  assert torch.equal(out, src)
