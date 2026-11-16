"""Segment Anything BoxDecode types and their versioned masks payload.

The payload is built here in Python, independently of the C++ parser, following
SegmentationMasksHeaderV1: a 48-byte header, the box records, padding to 16 bytes,
then one uint8 mask per slot at the original image size. Needs no model or device.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

import pyneat

MAGIC = 0x3153414D  # "MAS1"
HEADER_BYTES = 48
BOX_BYTES = 24  # int32 x, y, w, h; float32 score; int32 class


def _masks_v1(boxes, height: int, width: int, capacity: int | None = None) -> bytes:
  capacity = len(boxes) if capacity is None else capacity
  masks_offset = (HEADER_BYTES + capacity * BOX_BYTES + 15) // 16 * 16
  stride = height * width
  total = masks_offset + capacity * stride
  out = bytearray(total)
  struct.pack_into("<IHHIIIIIIIIII", out, 0, MAGIC, 1, HEADER_BYTES, total, len(boxes), capacity,
                   height, width, HEADER_BYTES, BOX_BYTES, masks_offset, stride, 0)
  for slot, (x, y, w, h, score, klass) in enumerate(boxes):
    struct.pack_into("<iiiifi", out, HEADER_BYTES + slot * BOX_BYTES, x, y, w, h, score, klass)
  for slot in range(capacity):
    start = masks_offset + slot * stride
    out[start:start + stride] = bytes([(slot + 1) * 40]) * stride
  return bytes(out)


def _tensor(payload: bytes) -> pyneat.Tensor:
  return pyneat.Tensor.from_numpy(
      np.frombuffer(payload, dtype=np.uint8).copy(), copy=True, memory=pyneat.TensorMemory.CPU
  )


# One decode type per Segment Anything model, and EdgeTAM's with or without its raw outputs,
# with Core's enum values.
SAM_TYPES = {
    "MobileSam": 28,
    "EdgeTamSeg": 29,
    "EfficientSam": 30,
    "EfficientSam3": 31,
    "EdgeTamTrack": 32,
}


def test_sam_decode_types():
  types = [getattr(pyneat.BoxDecodeType, name) for name in SAM_TYPES]
  assert [t.value for t in types] == list(SAM_TYPES.values())
  options = pyneat.ModelOptions()
  for decode_type in types:
    options.decode_type = decode_type
    assert options.decode_type == decode_type
  for retired in ("SamMask", "SamMaskSingle", "SamMaskStability", "SamConcept", "EdgeTam"):
    assert not hasattr(pyneat.BoxDecodeType, retired)
  # EdgeTamTrack returns the raw outputs; no option does.
  assert not hasattr(options, "boxdecode_raw_outputs")


def test_multimask_output_option():
  # Upstream predict(multimask_output=...): True by default. The per-model
  # mapping and its rejections are checked when the BoxDecode stage is built.
  options = pyneat.ModelOptions()
  assert options.multimask_output is True
  options.decode_type = pyneat.BoxDecodeType.EdgeTamSeg
  options.multimask_output = False
  assert options.multimask_output is False


def test_decoded_tensor_comes_first_with_raw_outputs():
  # EdgeTamTrack returns the BoxDecode tensor first, then the raw outputs; decode only that one.
  payload = _masks_v1([(320, 144, 426, 360, 0.8125, 2)], height=72, width=128)
  logits = pyneat.Tensor.from_numpy(np.zeros((1, 256, 256, 4), np.float32), copy=True,
                                    memory=pyneat.TensorMemory.CPU)
  outputs = [_tensor(payload), logits]
  result = pyneat.decode_segmentation(outputs[:1])[0]
  assert result.masks.to_numpy().shape == (1, 72, 128)
  plane = int(result.boxes.to_numpy()[0, 5])
  assert outputs[1].to_numpy()[0, :, :, plane].shape == (256, 256)


def test_sam_concept_keeps_several_masks():
  # An EfficientSAM3 concept image model publishes up to top_k queries, each
  # with a full-size mask.
  boxes = [(10, 20, 30, 40, 0.9, 0), (50, 60, 70, 80, 0.75, 0), (0, 0, 0, 0, 0.6, 0)]
  payload = _masks_v1(boxes, height=6, width=8, capacity=4)
  result = pyneat.decode_segmentation([_tensor(payload)])[0]
  # Scores are float32 on the wire; compare with the same float32 values.
  np.testing.assert_array_equal(result.boxes.to_numpy()[:, 4], np.float32([0.9, 0.75, 0.6]))
  masks = result.masks.to_numpy()
  assert masks.shape == (3, 6, 8)
  assert [int(m[0, 0]) for m in masks] == [40, 80, 120]
  capped = pyneat.decode_segmentation([_tensor(payload)], top_k=2)[0]
  assert capped.masks.to_numpy().shape == (2, 6, 8)


def test_decode_segmentation_reads_full_size_masks():
  payload = _masks_v1([(320, 144, 426, 360, 0.8125, 2)], height=72, width=128)
  result = pyneat.decode_segmentation([_tensor(payload)])[0]
  boxes = result.boxes.to_numpy()
  masks = result.masks.to_numpy()
  assert boxes.shape == (1, 6) and boxes.dtype == np.float32
  np.testing.assert_array_equal(boxes[0], [320, 144, 746, 504, 0.8125, 2])
  assert masks.shape == (1, 72, 128) and masks.dtype == np.uint8
  assert (masks == 40).all()


def test_empty_mask_keeps_score_and_index():
  payload = _masks_v1([(0, 0, 0, 0, 0.5, 1)], height=4, width=6)
  result = pyneat.decode_segmentation([_tensor(payload)])[0]
  np.testing.assert_array_equal(result.boxes.to_numpy()[0], [0, 0, 0, 0, 0.5, 1])
  assert result.masks.to_numpy().shape == (1, 4, 6)
  assert pyneat.decode_bbox([_tensor(payload)])[0].to_numpy().shape == (1, 6)


def test_unused_capacity_and_stride():
  payload = _masks_v1([(1, 2, 3, 4, 0.25, 0)], height=3, width=5, capacity=3)
  masks = pyneat.decode_segmentation([_tensor(payload)])[0].masks.to_numpy()
  assert masks.shape == (1, 3, 5)
  assert (masks == 40).all()


def test_legacy_160_payload_is_unchanged():
  legacy = bytearray(struct.pack("<I", 1))
  legacy += struct.pack("<iiiifi", 5, 6, 7, 8, 0.25, 9)
  legacy += bytes(range(256)) * 100
  result = pyneat.decode_segmentation([_tensor(bytes(legacy))])[0]
  np.testing.assert_array_equal(result.boxes.to_numpy()[0], [5, 6, 12, 14, 0.25, 9])
  masks = result.masks.to_numpy()
  assert masks.shape == (1, 160, 160)
  assert masks.reshape(-1)[255] == 255


def test_malformed_header_is_rejected():
  payload = bytearray(_masks_v1([(1, 2, 3, 4, 0.5, 0)], height=4, width=4))
  struct.pack_into("<H", payload, 4, 2)  # unsupported version
  with pytest.raises((ValueError, RuntimeError)):
    pyneat.decode_segmentation([_tensor(bytes(payload))])
