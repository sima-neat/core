#!/usr/bin/env python3
"""Validate a Graph, measure a Run workload, then read the report.

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
  inp.format = pyneat.Format.RGB
  inp.width = 128
  inp.height = 96
  inp.depth = 3

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(inp))
  graph.add(pyneat.nodes.output())

  # CORE LOGIC
  # STEP validate-graph
  # Validate pipeline before building.
  report = graph.validate()
  print(f"validate_error_code={report.error_code}")
  # END STEP

  # STEP run-with-measurement
  # Build a reusable runner and measure the caller-owned workload.
  ropt = pyneat.RunOptions()
  ropt.output_memory = pyneat.OutputMemory.Owned
  run = graph.build([tensor], ropt)
  measure = pyneat.MeasureOptions()
  measure.title = "tutorial 011 diagnosis"
  scope = run.start_measurement(measure)
  run.run([tensor], timeout_ms=1000)
  measured = scope.stop()
  # END STEP

  # STEP read-diagnostics
  # Read diagnostics from the measurement report.
  print(f"inputs_enqueued={measured.counters.inputs_enqueued} outputs_pulled={measured.counters.outputs_pulled}")
  print(f"measure_text_size={len(measured.to_text())}")
  # END STEP
  # END CORE LOGIC
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
