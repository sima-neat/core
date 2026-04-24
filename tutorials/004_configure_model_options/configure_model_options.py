#!/usr/bin/env python3
"""Configure a YOLO Model with ModelOptions knobs and introspect its specs.

Usage:
  python3 model_options_chapter.py --mpk /path/to/yolo_v8s.tar.gz
"""
from __future__ import annotations

import argparse
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


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mpk", type=Path, required=True)
  args = ap.parse_args(argv[1:])

  opt = pyneat.ModelOptions()
  opt.format = "BGR"
  opt.input_max_width = 640
  opt.input_max_height = 640
  opt.input_max_depth = 3
  opt.decode_type = "yolov8"
  opt.score_threshold = 0.55
  opt.nms_iou_threshold = 0.45
  opt.top_k = 100
  opt.original_width = 640
  opt.original_height = 640
  opt.name_suffix = "_chapter"
  opt.preproc.normalize = True
  opt.preproc.channel_mean = [0.485, 0.456, 0.406]
  opt.preproc.channel_stddev = [0.229, 0.224, 0.225]

  # CORE LOGIC
  model = pyneat.Model(str(args.mpk), opt)
  print(f"input_rank={model.input_spec().rank}")
  print(f"output_rank={model.output_spec().rank}")
  print(f"metadata_keys={len(model.metadata())}")
  # END CORE LOGIC

  rgb = np.full((224, 224, 3), 44, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.BGR)
  sample = model.run(tensor, timeout_ms=2000)
  print(f"output_kind={sample.kind}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
