"""Phase 6 slice (S7): pyneat.detections typed object-detection decode helpers."""

from __future__ import annotations

import pyneat


def test_detections_surface_present():
  det = pyneat.detections
  for name in (
      "Box",
      "BoxDecodeResult",
      "decode_bbox_tensor",
      "parse_bbox_bytes",
      "read_detection_format",
      "format_is_bbox",
      "format_is_pose",
      "format_is_segmentation",
      "format_is_bbox_family",
  ):
    assert hasattr(det, name), name


def test_detections_is_not_a_mirror_of_top_level():
  # S7: do NOT mirror the existing top-level decode_bbox/decode_pose/decode_segmentation.
  assert hasattr(pyneat, "decode_bbox")
  assert not hasattr(pyneat.detections, "decode_bbox")
  assert not hasattr(pyneat.detections, "decode_pose")
  assert not hasattr(pyneat.detections, "decode_segmentation")
  # boxes_to_tensor and tensor-level pose/seg decoders were intentionally dropped.
  assert not hasattr(pyneat.detections, "boxes_to_tensor")


def test_box_roundtrip_and_repr():
  box = pyneat.detections.Box()
  box.x1, box.y1, box.x2, box.y2 = 10.0, 20.0, 110.0, 220.0
  box.score = 0.5  # exactly representable in float32
  box.class_id = 3
  assert box.x2 == 110.0
  assert box.score == 0.5
  assert box.class_id == 3
  assert "class_id=3" in repr(box)


def test_box_decode_result_default():
  result = pyneat.detections.BoxDecodeResult()
  assert list(result.boxes) == []
  assert result.raw == b""


def test_format_predicates():
  det = pyneat.detections
  assert det.format_is_bbox("BBOX") is True
  assert det.format_is_pose("BBOX_POSE") is True
  assert det.format_is_segmentation("BBOX_SEGMENTATION") is True
  assert det.format_is_bbox_family("BBOX_POSE") is True
  assert det.format_is_pose("BBOX") is False
  assert det.format_is_bbox("BBOX_POSE") is False


def test_parse_bbox_bytes_empty_best_effort():
  # Best-effort (non-strict) parse of an empty payload yields no boxes.
  boxes = pyneat.detections.parse_bbox_bytes(b"", img_w=640, img_h=640, expected_topk=0, strict=False)
  assert list(boxes) == []
