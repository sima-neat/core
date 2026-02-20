#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import perf_schema as schema


def make_baseline() -> schema.ScenarioBaseline:
    return schema.parse_scenario_baseline(
        {
            "scenario_id": "runtime_session_async_rgb",
            "model_id": "synthetic",
            "pipeline_id": "async_rgb",
            "run_mode": "async",
            "iterations": 200,
            "metrics_thresholds": {
                "throughput_min": 100.0,
                "p50_max": 10.0,
                "p95_max": 20.0,
                "startup_max": 100.0,
                "rss_peak_kb_max": 100000.0,
                "input_drop_count_max": 0.0,
                "output_drop_count_max": 0.0,
                "regression_tolerance_percent": 10.0,
            },
        }
    )


class PerfComparatorTest(unittest.TestCase):
    def test_tolerance_boundary_passes(self) -> None:
        baseline = make_baseline()
        metrics = {
            "throughput": 90.0,  # exactly floor with 10% tolerance
            "p50": 11.0,  # exactly ceiling
            "p95": 22.0,  # exactly ceiling
            "startup": 110.0,  # exactly ceiling
            "rss_peak_kb": 110000.0,  # exactly ceiling
            "input_drop_count": 0.0,
            "output_drop_count": 0.0,
        }
        self.assertEqual(schema.compare_metrics(metrics, baseline), [])

    def test_tolerance_breach_fails(self) -> None:
        baseline = make_baseline()
        metrics = {
            "throughput": 89.999,
            "p50": 11.001,
            "p95": 22.001,
            "startup": 110.001,
            "rss_peak_kb": 110000.001,
            "input_drop_count": 1.0,
            "output_drop_count": 0.0,
        }
        failures = schema.compare_metrics(metrics, baseline)
        self.assertIn(schema.ReasonCode.REGRESSION_THROUGHPUT, failures)
        self.assertIn(schema.ReasonCode.REGRESSION_P50, failures)
        self.assertIn(schema.ReasonCode.REGRESSION_P95, failures)
        self.assertIn(schema.ReasonCode.REGRESSION_STARTUP, failures)
        self.assertIn(schema.ReasonCode.REGRESSION_RSS, failures)
        self.assertIn(schema.ReasonCode.REGRESSION_DROPS, failures)


if __name__ == "__main__":
    unittest.main()
