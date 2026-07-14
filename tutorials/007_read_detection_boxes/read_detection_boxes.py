#!/usr/bin/env python3
"""Decode YOLO detections through ModelOptions and read the BBOX tensor.

The output tensor is a rank-1 uint8 buffer: a uint32 count header followed by
N 24-byte RawBox records (int32 x, y, w, h; float32 score; int32 class_id).

Usage:
  python3 read_detection_boxes.py --model /path/to/yolo_v8s.tar.gz --image /path/to.jpg
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

try:
  import cv2
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )

def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", type=Path, required=True)
  ap.add_argument("--image", type=Path, required=True)
  args = ap.parse_args(argv[1:])

  bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
  if bgr is None:
    raise RuntimeError(f"failed to load image: {args.image}")
  height, width, channels = bgr.shape

  # STEP configure-decode
  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Image
  opt.preprocess.enable = pyneat.AutoFlag.On
  opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
  opt.preprocess.input_max_width = width
  opt.preprocess.input_max_height = height
  opt.preprocess.input_max_depth = channels
  opt.preprocess.normalize.enable = pyneat.AutoFlag.On
  opt.decode_type = pyneat.BoxDecodeType.YoloV8
  opt.score_threshold = 0.55
  opt.nms_iou_threshold = 0.50
  opt.top_k = 100
  opt.boxdecode_original_width = width
  opt.boxdecode_original_height = height
  # END STEP

  # CORE LOGIC
  # STEP load-model
  model = pyneat.Model(str(args.model), opt)
  # END STEP

  # STEP run-decode
  tensor = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)
  outputs = model.run([tensor], timeout_ms=30000)
  if not outputs:
    raise RuntimeError("model produced no outputs")
  # END STEP
  # END CORE LOGIC

  # STEP read-boxes
  # Two paths for reading the output:
  #   - Runtimes that wire BoxDecode into model.run produce one BBOX uint8 tensor.
  #   - Runtimes that do not return the raw MLA feature-map tensors instead.
  # The shipped README explains the BBOX wire format (uint32 count + N 24-byte RawBox).
  first = outputs[0] if outputs else None
  detection = getattr(first.semantic, "detection", None) if first is not None else None
  if first is not None and detection is not None and getattr(detection, "format", "") == "BBOX":
    buf = bytes(first.to_numpy(copy=False))
    detections = struct.unpack_from("<I", buf, 0)[0] if len(buf) >= 4 else 0
    print(f"detections={detections}")
    for index in range(detections):
      offset = 4 + index * 24
      if offset + 24 > len(buf):
        raise RuntimeError("truncated BBOX output")
      x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", buf, offset)
      print(
        f"box[{index}] class={class_id} score={score:.6f} "
        f"xyxy=[{x},{y},{x + w},{y + h}]"
      )
    if detections == 0:
      raise RuntimeError("image produced no detections above threshold")
  else:
    raise RuntimeError(f"BoxDecode not wired by this runtime; raw_output_heads={len(outputs)}")
  # END STEP
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
