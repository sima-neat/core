#!/usr/bin/env python3
"""Benchmark a compiled model with deterministic synthetic inputs.

Usage:
  python3 benchmark_your_model.py --model /path/to/model.tar.gz [--samples 100]
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


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", type=Path, required=True)
  ap.add_argument("--samples", type=int, default=100, help="number of measured synthetic samples")
  args = ap.parse_args(argv[1:])

  # CORE LOGIC
  # STEP load-model
  model = pyneat.Model(str(args.model))
  # END STEP

  # STEP run-benchmark
  report = model.benchmark(args.samples)
  if report.latency_ms <= 0 or report.fps <= 0:
    raise RuntimeError("benchmark produced no measured latency/fps")
  # END STEP

  # STEP read-report
  print(f"report_latency_ms={report.latency_ms}")
  print(f"report_fps={report.fps}")
  print(f"report_avg_power_watts={report.avg_power_watts}")
  print(f"report_energy_joules={report.energy_joules}")
  # END STEP
  # END CORE LOGIC

  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
