"""Phase 1 ergonomics slice (S10): Pythonic Sample sequence protocol + Tensor.from_list."""

from __future__ import annotations

import pytest

import pyneat


def test_sample_sequence_protocol():
  bundle = pyneat.Sample()
  # Empty sample: falsy and zero-length.
  assert len(bundle) == 0
  assert not bundle

  first = pyneat.Sample()
  first.payload_tag = "boxes"
  second = pyneat.Sample()
  second.payload_tag = "scores"
  bundle.append(first)
  bundle.append(second)

  assert len(bundle) == 2
  assert bool(bundle)
  assert bundle[0].payload_tag == "boxes"
  assert bundle[1].payload_tag == "scores"
  assert bundle[-1].payload_tag == "scores"  # negative indexing
  assert [child.payload_tag for child in bundle] == ["boxes", "scores"]  # __iter__


def test_sample_getitem_is_bounds_checked():
  bundle = pyneat.Sample()
  bundle.append(pyneat.Sample())
  # Raw C++ operator[] returns *this for any index; the binding must raise instead.
  with pytest.raises(IndexError):
    _ = bundle[5]
  with pytest.raises(IndexError):
    _ = bundle[-5]


def test_sample_front_back_not_exposed():
  # Dropped in favor of s[0] / s[-1] (S10).
  assert not hasattr(pyneat.Sample, "front")
  assert not hasattr(pyneat.Sample, "back")
  assert not hasattr(pyneat.Sample, "reserve")


def test_tensor_from_list_with_shape_and_dtype():
  tensor = pyneat.Tensor.from_list([1.0, 2.0, 3.0, 4.0], shape=(2, 2), dtype="float32")
  out = tensor.to_numpy()
  assert out.shape == (2, 2)
  assert out.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_tensor_from_list_infers_shape():
  out = pyneat.Tensor.from_list([1, 2, 3]).to_numpy()
  assert out.tolist() == [1, 2, 3]


def test_tensor_from_list_is_isolated_from_source():
  data = [1.0, 2.0, 3.0]
  tensor = pyneat.Tensor.from_list(data)
  data[0] = 99.0  # mutating the source list must not affect the tensor
  assert tensor.to_numpy().tolist() == [1.0, 2.0, 3.0]
