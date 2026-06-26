#!/usr/bin/env python3
"""Tune queue_depth and overflow_policy on RunOptions, then read MeasureReport.

Usage:
  python3 tune_throughput_and_queues.py [--iters 32] [--queue 4] [--drop {block,latest,incoming}]
"""
from __future__ import annotations

import argparse
import sys

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )

import numpy as np


DROP_MODES = {
    "block": "Block",
    "latest": "KeepLatest",
    "incoming": "DropIncoming",
}


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--iters", type=int, default=32)
  ap.add_argument("--queue", type=int, default=4)
  ap.add_argument("--drop", choices=DROP_MODES.keys(), default="block")
  args = ap.parse_args(argv[1:])

  rgb = np.full((120, 160, 3), 88, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

  inp = pyneat.InputOptions()
  inp.format = pyneat.Format.RGB
  inp.width = 160
  inp.height = 120
  inp.depth = 3
  inp.is_live = True

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(inp))
  graph.add(pyneat.nodes.output())

  # CORE LOGIC
  # STEP configure-run-options
  opt = pyneat.RunOptions()
  opt.queue_depth = args.queue
  opt.overflow_policy = getattr(pyneat.OverflowPolicy, DROP_MODES[args.drop])
  opt.output_memory = pyneat.OutputMemory.Owned
  run = graph.build([tensor], opt)
  # END STEP

  # STEP push-workload
  measure = pyneat.MeasureOptions()
  measure.title = "tutorial 016 throughput"
  scope = run.start_measurement(measure)
  for _ in range(args.iters):
    run.try_push([tensor])
  run.close_input()

  pulled = 0
  while run.pull(timeout_ms=1000) is not None:
    pulled += 1
  measured = scope.stop()
  if measured.counters.inputs_enqueued <= 0 or pulled <= 0:
    raise RuntimeError("throughput run produced no measured inputs/outputs")
  # END STEP
  # END CORE LOGIC
  # STEP read-measurement
  print(f"inputs_enqueued={measured.counters.inputs_enqueued}")
  print(f"inputs_dropped={measured.counters.inputs_dropped}")
  print(f"outputs_pulled={pulled}")
  print(f"avg_latency_ms={measured.end_to_end.avg_ms}")
  print(f"avg_push_us={measured.input.avg_push_us}")
  print(f"renegotiations={measured.input.renegotiations}")
  # END STEP
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
