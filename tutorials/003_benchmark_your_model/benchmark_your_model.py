#!/usr/bin/env python3
"""Benchmark a compiled model with deterministic synthetic inputs.

Usage:
  python3 benchmark_your_model.py --model /path/to/model.tar.gz [--samples 100]
    [--report results/model/yolo_benchmark.yaml] [--include-layer-stats]
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
  ap.add_argument("--report", type=Path, help="destination report file (.json selects JSON; otherwise YAML)")
  ap.add_argument("--include-layer-stats", action="store_true",
                  help="advanced: request separate MLA layer profiling (requires an idle devkit)")
  args = ap.parse_args(argv[1:])

  # CORE LOGIC
  # STEP load-model
  model = pyneat.Model(str(args.model))
  # END STEP

  # STEP run-benchmark
  options = pyneat.BenchmarkOptions()
  options.num_samples = args.samples
  options.include_layer_stats = args.include_layer_stats
  report = model.benchmark(options)
  if report.latency_ms <= 0 or report.fps <= 0:
    raise RuntimeError("benchmark produced no measured latency/fps")
  # END STEP

  # STEP read-report
  print(f"report_latency_ms={report.latency_ms}")
  print(f"report_fps={report.fps}")
  print(f"report_avg_power_watts={report.avg_power_watts}")
  print(f"report_energy_joules={report.energy_joules}")
  print(f"report_peak_power_watts={report.peak_power_watts}")
  print(f"report_throughput_duration_ms={report.throughput_duration_ms}")
  print(f"report_telemetry_status={report.telemetry.status}")
  # END STEP
  # STEP read-layer-stats
  if options.include_layer_stats:
    print("Layer stats (execution time in microseconds):")
    for layer in report.layer_stats:
      print(f"  stage_index={layer.stage_index} name={layer.name} "
            f"total_execution_time_us={layer.total_execution_time_us}")
  # END STEP

  # STEP save-report
  if args.report is not None:
    print(f"report_path={report.save(args.report)}")
  # END STEP
  # END CORE LOGIC

  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
