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
            "scenario_id": "runtime_session_sync_rgb",
            "model_id": "synthetic",
            "pipeline_id": "sync_rgb",
            "run_mode": "sync",
            "iterations": 200,
            "metrics_thresholds": {
                "throughput_min": 20.0,
                "p50_max": 40.0,
                "p95_max": 60.0,
                "startup_max": 1000.0,
                "rss_peak_kb_max": 200000.0,
                "input_drop_count_max": 0.0,
                "output_drop_count_max": 0.0,
                "regression_tolerance_percent": 10.0,
            },
        }
    )


def ok_metrics() -> dict[str, float]:
    return {
        "throughput": 25.0,
        "p50": 20.0,
        "p95": 40.0,
        "startup": 500.0,
        "rss_peak_kb": 100000.0,
        "input_drop_count": 0.0,
        "output_drop_count": 0.0,
    }


class PerfClassifierTest(unittest.TestCase):
    def test_throughput_regression_maps_reason(self) -> None:
        baseline = make_baseline()
        metrics = ok_metrics()
        metrics["throughput"] = 1.0
        failures = schema.compare_metrics(metrics, baseline)
        self.assertIn(schema.ReasonCode.REGRESSION_THROUGHPUT, failures)

    def test_p95_regression_maps_reason(self) -> None:
        baseline = make_baseline()
        metrics = ok_metrics()
        metrics["p95"] = 999.0
        failures = schema.compare_metrics(metrics, baseline)
        self.assertIn(schema.ReasonCode.REGRESSION_P95, failures)

    def test_drop_regression_maps_reason(self) -> None:
        baseline = make_baseline()
        metrics = ok_metrics()
        metrics["output_drop_count"] = 1.0
        failures = schema.compare_metrics(metrics, baseline)
        self.assertIn(schema.ReasonCode.REGRESSION_DROPS, failures)

    def test_env_reason_mapping(self) -> None:
        self.assertEqual(
            schema.classify_env_failure(1, "sima-cli fail: not found"),
            schema.ReasonCode.ENV_SIMA_CLI_FAIL,
        )
        self.assertEqual(
            schema.classify_env_failure(1, "model download error"),
            schema.ReasonCode.ENV_MODEL_DOWNLOAD_FAIL,
        )
        self.assertEqual(
            schema.classify_env_failure(124, "timeout"),
            schema.ReasonCode.ENV_TIMEOUT,
        )


if __name__ == "__main__":
    unittest.main()
