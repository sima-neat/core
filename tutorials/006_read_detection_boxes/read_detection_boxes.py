#!/usr/bin/env python3
"""Decode YOLO detections through ModelOptions and read the BBOX tensor.

The output tensor is a rank-1 uint8 buffer: a uint32 count header followed by
N 24-byte RawBox records (int32 x, y, w, h; float32 score; int32 class_id).

Usage:
  python3 postproc_boxdecode.py --mpk /path/to/yolo_v8s.tar.gz [--width 640] [--height 640]
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
      "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )

import numpy as np


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mpk", type=Path, required=True)
  ap.add_argument("--width", type=int, default=640)
  ap.add_argument("--height", type=int, default=640)
  args = ap.parse_args(argv[1:])

  opt = pyneat.ModelOptions()
  opt.format = "RGB"
  opt.input_max_width = args.width
  opt.input_max_height = args.height
  opt.input_max_depth = 3
  opt.decode_type = "yolov8"
  opt.score_threshold = 0.55
  opt.nms_iou_threshold = 0.50
  opt.top_k = 100
  opt.original_width = args.width
  opt.original_height = args.height

  # CORE LOGIC
  model = pyneat.Model(str(args.mpk), opt)

  rgb = np.full((args.height, args.width, 3), 80, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
  sample = model.run(tensor, timeout_ms=2000)
  # END CORE LOGIC

  # Two paths for reading the output:
  #   - Runtimes that wire BoxDecode into model.run produce one BBOX uint8 tensor.
  #   - Runtimes that do not produce a Bundle of raw MLA feature maps instead.
  # The shipped README explains the BBOX wire format (uint32 count + N 24-byte RawBox).
  if sample.tensor is not None:
    buf = bytes(sample.tensor.to_numpy(copy=False))
    detections = struct.unpack_from("<I", buf, 0)[0] if len(buf) >= 4 else 0
    print(f"detections={detections}")
  else:
    heads = len(sample.fields or [])
    print(f"raw_output_heads={heads}  # BoxDecode not wired by this runtime")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
