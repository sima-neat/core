"""Independent BBOX_OBB_V1 byte fixture; no inference or device required."""
import struct

import numpy as np
import pytest
import pyneat


def payload(count=1, capacity=2):
  raw = bytearray(struct.pack("<IHH6I", 0x3156424f, 1, 32, 32 + 32 * capacity,
                              count, capacity, 32, 32, 0))
  for index in range(capacity):
    raw += (struct.pack("<6fiI", -4.0, 12.0, -32.0, 48.0, 0.4, 0.5, 9 + index, 0)
            if index < count else bytes(32))
  return raw


def tensor(raw, tag="BBOX_OBB_V1"):
  result = pyneat.Tensor.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy(), copy=True,
                                    memory=pyneat.TensorMemory.CPU)
  detection = pyneat.DetectionSpec()
  detection.format = tag
  semantic = result.semantic
  semantic.detection = detection
  result.semantic = semantic
  return result


def test_decode_obb_shapes_and_values():
  result = pyneat.decode_obb([tensor(payload())])[0]
  np.testing.assert_array_equal(result.geometry.to_numpy(),
                                np.array([[-4, 12, -32, 48, 0.4]], dtype=np.float32))
  np.testing.assert_array_equal(result.scores.to_numpy(), np.array([0.5], dtype=np.float32))
  np.testing.assert_array_equal(result.class_ids.to_numpy(), np.array([9], dtype=np.int32))
  assert pyneat.BoxDecodeType.YoloV26Obb is not pyneat.BoxDecodeType.YoloV26
  assert pyneat.BoxDecodeType.YoloV8Obb is not pyneat.BoxDecodeType.YoloV26Obb


def test_decode_obb_empty_and_list_correspondence():
  results = pyneat.decode_obb([tensor(payload(0)), tensor(payload(2))])
  assert results[0].geometry.to_numpy().shape == (0, 5)
  assert results[0].class_ids.to_numpy().shape == (0,)
  assert results[1].geometry.to_numpy().shape == (2, 5)


@pytest.mark.parametrize("offset", [0, 4, 6, 8, 12, 16, 20, 24, 28, 60, 64])
def test_decode_obb_rejects_header_reserved_and_tail(offset):
  raw = payload()
  raw[offset] ^= 255
  with pytest.raises((ValueError, RuntimeError)):
    pyneat.decode_obb_tensor(tensor(raw))


def test_decode_obb_rejects_truncation_and_format_confusion():
  for value in [tensor(payload()[:-1]), tensor(payload() + bytes(1)),
                tensor(payload(), "BBOX"), tensor(payload(), "")]:
    with pytest.raises((ValueError, RuntimeError)):
      pyneat.decode_obb([value])
  # The established axis-aligned parser reports a semantic format mismatch as TypeError.
  with pytest.raises(TypeError, match="BBOX_OBB_V1"):
    pyneat.decode_bbox([tensor(payload())])


@pytest.mark.parametrize("offset,value", [(32, float("nan")), (48, float("inf")),
                                          (52, -0.1), (52, 1.1)])
def test_decode_obb_rejects_nonfinite_and_score_range(offset, value):
  raw = payload()
  struct.pack_into("<f", raw, offset, value)
  with pytest.raises((ValueError, RuntimeError)):
    pyneat.decode_obb([tensor(raw)])


def test_decode_obb_returns_owned_storage_after_carrier_reuse():
  import gc

  backing = np.frombuffer(payload(), dtype=np.uint8).copy()
  source = pyneat.Tensor.from_numpy(backing, copy=False, memory=pyneat.TensorMemory.CPU)
  detection = pyneat.DetectionSpec()
  detection.format = "BBOX_OBB_V1"
  semantic = source.semantic
  semantic.detection = detection
  source.semantic = semantic
  decoded = pyneat.decode_obb_tensor(source)
  geometry = decoded.geometry.to_numpy()
  scores = decoded.scores.to_numpy()
  classes = decoded.class_ids.to_numpy()
  backing[:] = 0
  del source, decoded, backing
  gc.collect()
  np.testing.assert_array_equal(geometry,
                                np.array([[-4, 12, -32, 48, 0.4]], dtype=np.float32))
  np.testing.assert_array_equal(scores, np.array([0.5], dtype=np.float32))
  np.testing.assert_array_equal(classes, np.array([9], dtype=np.int32))
