"""Tests for pyneat.decode_segmentation_pose (TensorList -> list of results).

The wire payload is built here in Python, independently of the C++ parser: a 4-byte
count header, then boxes, masks and keypoints, each region strided by the same slot
count. Every field carries a distinctive value so a wrong region base or stride shows
up as a wrong number rather than a crash.

Needs no model or device — the payload is synthesized, so this runs everywhere.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

import pyneat

BOX_COLUMNS = 6
MASK_W = 160
MASK_H = 160
MASK_BYTES = MASK_W * MASK_H
POSE_POINTS = 17
POSE_COLUMNS = 3
BOX_BYTES = 24   # int32 x, y, w, h; float32 score; int32 class
POSE_BYTES = 204  # 17 * (uint32 x, uint32 y, float32 visible)


def _payload(count: int, capacity: int) -> bytes:
  """Build a BBOX_SEGMENTATION_POSE payload with per-slot identifiable values."""
  out = bytearray(struct.pack("<I", count))

  for slot in range(capacity):
    out += struct.pack("<iiiifi", 100 + slot, 200 + slot, 30, 40, 0.5, 7 + slot)
  assert len(out) == 4 + capacity * BOX_BYTES

  for slot in range(capacity):
    out += bytes([slot + 1]) * MASK_BYTES

  for slot in range(capacity):
    for point in range(POSE_POINTS):
      out += struct.pack("<IIf", 1000 * (slot + 1) + point, 2000 * (slot + 1) + point,
                         0.01 * (point + 1))

  assert len(out) == 4 + capacity * (BOX_BYTES + MASK_BYTES + POSE_BYTES)
  return bytes(out)


def _tensor(payload: bytes) -> pyneat.Tensor:
  """An untagged rank-1 uint8 buffer, which the decoder accepts as this format."""
  return pyneat.Tensor.from_numpy(
      np.frombuffer(payload, dtype=np.uint8).copy(), copy=True, memory=pyneat.TensorMemory.CPU
  )


def _decode(payload: bytes, **kwargs):
  return pyneat.decode_segmentation_pose([_tensor(payload)], **kwargs)[0]


def test_shapes_and_dtypes():
  result = _decode(_payload(2, 2))
  boxes = result.boxes.to_numpy()
  masks = result.masks.to_numpy()
  keypoints = result.keypoints.to_numpy()

  assert boxes.shape == (2, BOX_COLUMNS)
  assert boxes.dtype == np.float32
  assert masks.shape == (2, MASK_H, MASK_W)
  assert masks.dtype == np.uint8
  assert keypoints.shape == (2, POSE_POINTS, POSE_COLUMNS)
  assert keypoints.dtype == np.float32


def test_values_match_the_python_reference():
  result = _decode(_payload(2, 2))
  boxes = result.boxes.to_numpy()
  masks = result.masks.to_numpy()
  keypoints = result.keypoints.to_numpy()

  # The payload carries x, y, w, h; the decoder returns corners, so x2 == x + w.
  assert list(boxes[0]) == [100.0, 200.0, 130.0, 240.0, 0.5, 7.0]
  assert list(boxes[1]) == [101.0, 201.0, 131.0, 241.0, 0.5, 8.0]

  # Slot i's mask is filled with i + 1. Checking both slots separates a wrong base
  # (both wrong) from a wrong stride (only the second wrong).
  assert np.all(masks[0] == 1)
  assert np.all(masks[1] == 2)

  # Keypoints sit past the whole mask region, which is the offset unique to this format.
  assert keypoints[0][0][0] == 1000.0
  assert keypoints[0][0][1] == 2000.0
  assert keypoints[1][0][0] == 2000.0
  assert keypoints[0][16][0] == 1016.0
  assert keypoints[0][3][2] == pytest.approx(0.04, rel=1e-6)


def test_empty_output():
  result = _decode(_payload(0, 8))
  assert result.boxes.to_numpy().shape == (0, BOX_COLUMNS)
  assert result.masks.to_numpy().shape == (0, MASK_H, MASK_W)
  assert result.keypoints.to_numpy().shape == (0, POSE_POINTS, POSE_COLUMNS)


def test_partially_filled_uses_capacity_for_strides():
  # Count 4 of capacity 8: the row count comes from the header, the region strides
  # from the capacity the buffer size implies.
  result = _decode(_payload(4, 8))
  masks = result.masks.to_numpy()
  keypoints = result.keypoints.to_numpy()
  assert masks.shape[0] == 4
  assert np.all(masks[3] == 4)
  assert keypoints[3][0][0] == 4000.0


def test_top_k_caps_rows_without_moving_regions():
  result = _decode(_payload(4, 8), top_k=2)
  masks = result.masks.to_numpy()
  assert masks.shape[0] == 2
  assert np.all(masks[1] == 2)


def test_capacity_limit_clamps_then_rejects_in_strict_mode():
  payload = bytearray(_payload(2, 2))
  payload[0:4] = struct.pack("<I", 9)  # more detections than the buffer can hold
  clamped = _decode(bytes(payload))
  assert clamped.boxes.to_numpy().shape[0] == 2

  with pytest.raises(RuntimeError):
    _decode(bytes(payload), strict=True)


def test_malformed_payload_size_is_rejected():
  truncated = _payload(1, 1)[:-1]
  with pytest.raises(RuntimeError):
    _decode(truncated)


def test_positional_one_to_one():
  tensors = [_tensor(_payload(1, 1)), _tensor(_payload(2, 2))]
  results = pyneat.decode_segmentation_pose(tensors)
  assert len(results) == 2
  assert results[0].boxes.to_numpy().shape[0] == 1
  assert results[1].boxes.to_numpy().shape[0] == 2
