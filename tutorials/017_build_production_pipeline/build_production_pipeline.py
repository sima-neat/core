#!/usr/bin/env python3
"""Production blueprint: Model + ModelRouteOptions + async RunOptions.

Usage:
  python3 build_production_pipeline.py --model /path/to/resnet_50.tar.gz [--iters 4]
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
  ap.add_argument("--iters", type=int, default=4)
  args = ap.parse_args(argv[1:])

  rgb = np.full((224, 224, 3), 123, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

  # CORE LOGIC
  # STEP configure-run-options
  ropt = pyneat.RunOptions()
  ropt.queue_depth = 8
  ropt.overflow_policy = pyneat.OverflowPolicy.Block
  ropt.output_memory = pyneat.OutputMemory.Owned
  # END STEP

  # STEP configure-model
  mopt = pyneat.ModelOptions()
  mopt.input_max_width = int(tensor.shape[1])
  mopt.input_max_height = int(tensor.shape[0])
  mopt.input_max_depth = int(tensor.shape[2])
  mopt.name_suffix = "_prod"
  model = pyneat.Model(str(args.model), mopt)
  # END STEP

  # STEP build-runner
  sopt = pyneat.ModelRouteOptions()
  sopt.include_input = True
  sopt.include_output = True
  sopt.name_suffix = "_prod"

  runner = model.build([tensor], sopt, ropt)
  # END STEP

  # STEP run-loop
  ok = 0
  for _ in range(args.iters):
    if not runner.push([tensor]):
      continue
    if runner.pull(timeout_ms=2000) is not None:
      ok += 1
  runner.close()
  # END STEP
  # END CORE LOGIC

  print(f"iters={args.iters} ok={ok}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
