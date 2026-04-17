#!/usr/bin/env python3
"""End-to-end YOLOv8 detection: load the MPK, run on an image, count detections.

Usage:
  python3 yolo_quickstart.py --mpk /path/to/yolo_v8s.tar.gz [--image /path/to.jpg]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either NEAT is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )

import numpy as np
import cv2


def load_image(path: Path | None, size: int) -> np.ndarray:
  if path is None:
    return np.full((size, size, 3), 66, dtype=np.uint8)
  bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if bgr is None:
    raise RuntimeError(f"failed to read image: {path}")
  if bgr.shape[0] != size or bgr.shape[1] != size:
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mpk", type=Path, required=True)
  ap.add_argument("--image", type=Path)
  ap.add_argument("--size", type=int, default=640)
  args = ap.parse_args(argv[1:])

  opt = pyneat.ModelOptions()
  opt.format = "RGB"
  opt.input_max_width = args.size
  opt.input_max_height = args.size
  opt.input_max_depth = 3
  opt.decode_type = "yolov8"
  opt.score_threshold = 0.52
  opt.nms_iou_threshold = 0.50
  opt.top_k = 100
  opt.original_width = args.size
  opt.original_height = args.size

  model = pyneat.Model(str(args.mpk), opt)
  image = load_image(args.image, size=args.size)
  sample = model.run(image, timeout_ms=2000)

  buf = bytes(sample.tensor.to_numpy(copy=False))
  detections = struct.unpack_from("<I", buf, 0)[0] if len(buf) >= 4 else 0
  print(f"detections={detections}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
