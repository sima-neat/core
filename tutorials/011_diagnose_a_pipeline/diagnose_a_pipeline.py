#!/usr/bin/env python3
"""Validate a Graph, enable metrics on the Run, then read stats/report/diagnostics.

Usage:
  python3 diagnose_a_pipeline.py
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


def main(argv: list[str]) -> int:
  argparse.ArgumentParser(description=__doc__).parse_args(argv[1:])

  rgb = np.full((96, 128, 3), 77, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

  inp = pyneat.InputOptions()
  inp.format = "RGB"
  inp.width = 128
  inp.height = 96
  inp.depth = 3

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(inp))
  graph.add(pyneat.nodes.output())

  # CORE LOGIC
  # 1. Validate pipeline before building.
  report = graph.validate()
  print(f"validate_error_code={report.error_code}")

  # 2. Build + run with metrics enabled.
  ropt = pyneat.RunOptions()
  ropt.enable_metrics = True
  ropt.output_memory = pyneat.OutputMemory.Owned
  run = graph.build([tensor], pyneat.RunMode.Sync, ropt)
  run.run([tensor], timeout_ms=1000)

  # 3. Read diagnostics.
  stats = run.stats()
  print(f"inputs_enqueued={stats.inputs_enqueued} outputs_pulled={stats.outputs_pulled}")
  print(f"report_size={len(run.report())}")
  print(f"diagnostics_summary={run.diagnostics_summary()}")
  # END CORE LOGIC
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
