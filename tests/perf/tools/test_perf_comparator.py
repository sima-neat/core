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


def make_component_baseline() -> schema.ScenarioBaseline:
    data = {
        "scenario_id": "ssd_mobilenet_boxdecode",
        "model_id": "ssd_mobilenet_v2_heads_mpk",
        "pipeline_id": "ssd_boxdecode",
        "run_mode": "sync",
        "iterations": 1000,
        "metrics_thresholds": {
            "throughput_min": 1.0,
            "p50_max": 100.0,
            "p95_max": 100.0,
            "startup_max": 1000.0,
            "rss_peak_kb_max": 1000000.0,
            "input_drop_count_max": 0.0,
            "output_drop_count_max": 0.0,
            # This tolerance must never relax component caps.
            "regression_tolerance_percent": 50.0,
        },
        "component_latency_thresholds": {
            "boxdecode_plugin_exclusive": {
                "required": True,
                "min_samples": 1000,
                "p99_max_ms": 0.9,
                "max_max_ms": 0.999,
            }
        },
    }
    return schema.parse_scenario_baseline(data)


def component_row(**overrides: object) -> schema.ComponentLatency:
    data: dict[str, object] = {
        "backend": "A65",
        "phase": "Exec",
        "kernel_name": "boxdecode_plugin_exclusive",
        "stage_name": "boxdecode",
        "plugin_instance_id": "r1.g2.s0.n3.boxdecode",
        "samples": 1000,
        "reliable": True,
        "percentiles_available": True,
        "avg_ms": 0.4,
        "p50_ms": 0.4,
        "p95_ms": 0.7,
        "p99_ms": 0.9,
        "max_ms": 0.999,
    }
    data.update(overrides)
    return schema.parse_optional_component_latency_payload(
        {"component_latency": {"component": [data]}}
    )["component"][0]


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

    def test_component_absolute_cap_boundary_passes_without_tolerance(self) -> None:
        failures = schema.compare_component_latency(
            {"boxdecode_plugin_exclusive": [component_row()]}, make_component_baseline()
        )
        self.assertEqual(failures, [])

    def test_component_absolute_caps_are_not_tolerance_adjusted(self) -> None:
        failures = schema.compare_component_latency(
            {"boxdecode_plugin_exclusive": [component_row(p99_ms=0.901, max_ms=1.0)]},
            make_component_baseline(),
        )
        self.assertEqual(
            [failure.reason_code for failure in failures],
            [
                schema.ReasonCode.REGRESSION_COMPONENT_P99,
                schema.ReasonCode.REGRESSION_COMPONENT_MAX,
            ],
        )

    def test_component_missing_and_ambiguous_are_harness_errors(self) -> None:
        baseline = make_component_baseline()
        missing = schema.compare_component_latency({}, baseline)
        self.assertEqual(missing[0].reason_code, schema.ReasonCode.HARNESS_COMPONENT_MISSING)
        self.assertEqual(missing[0].failure_class, schema.FailureClass.HARNESS_ERROR)

        row = component_row()
        ambiguous = schema.compare_component_latency(
            {"boxdecode_plugin_exclusive": [row, row]}, baseline
        )
        self.assertEqual(
            ambiguous[0].reason_code, schema.ReasonCode.HARNESS_COMPONENT_AMBIGUOUS
        )

    def test_component_reliability_samples_and_percentiles_fail_closed(self) -> None:
        baseline = make_component_baseline()
        cases = (
            (component_row(reliable=False), schema.ReasonCode.HARNESS_COMPONENT_UNRELIABLE),
            (component_row(samples=999), schema.ReasonCode.HARNESS_COMPONENT_SAMPLE_COUNT),
            (
                component_row(percentiles_available=False),
                schema.ReasonCode.HARNESS_COMPONENT_PERCENTILES_MISSING,
            ),
        )
        for row, expected in cases:
            with self.subTest(expected=expected):
                failures = schema.compare_component_latency(
                    {"boxdecode_plugin_exclusive": [row]}, baseline
                )
                self.assertEqual(failures[0].reason_code, expected)
                self.assertEqual(failures[0].failure_class, schema.FailureClass.HARNESS_ERROR)


if __name__ == "__main__":
    unittest.main()
