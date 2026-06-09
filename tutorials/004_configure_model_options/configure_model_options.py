#!/usr/bin/env python3
"""Configure a YOLO Model with ModelOptions knobs and introspect its specs.

Usage:
  python3 configure_model_options.py --model /path/to/yolo_v8s.tar.gz
"""
from __future__ import annotations

import argparse
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
  ap.add_argument("--model", type=Path, required=True)
  args = ap.parse_args(argv[1:])

  opt = pyneat.ModelOptions()
  # STEP set-input-preproc
  opt.preprocess.kind = pyneat.InputKind.Image
  opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
  opt.preprocess.input_max_width = 640
  opt.preprocess.input_max_height = 640
  opt.preprocess.input_max_depth = 3
  opt.preprocess.normalize.enable = pyneat.AutoFlag.On
  opt.preprocess.normalize.mean = [0.485, 0.456, 0.406]
  opt.preprocess.normalize.stddev = [0.229, 0.224, 0.225]
  # END STEP
  # STEP set-postproc
  opt.decode_type = pyneat.BoxDecodeType.YoloV8
  opt.score_threshold = 0.55
  opt.nms_iou_threshold = 0.45
  opt.top_k = 100
  opt.boxdecode_original_width = 640
  opt.boxdecode_original_height = 640
  opt.name_suffix = "_chapter"
  # END STEP

  # CORE LOGIC
  # STEP load-and-inspect
  model = pyneat.Model(str(args.model), opt)
  print(f"input_shape={model.input_spec().shape}")
  print(f"output_shape={model.output_spec().shape}")
  print(f"metadata_keys={len(model.metadata())}")
  # END STEP
  # END CORE LOGIC

  # STEP run-inference
  bgr = np.full((640, 640, 3), 44, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)
  outputs = model.run([tensor], timeout_ms=2000)
  print(f"output_count={len(outputs)}")
  # END STEP
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
