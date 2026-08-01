#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import perf_schema as schema


def valid_profile_dict() -> dict[str, object]:
    return {
        "modalix_profile_id": "modalix_default",
        "board_class": "modalix-v1",
        "sdk_version": "2.0.0",
        "compiler": "g++-12",
        "gstreamer_version": "1.20.3",
        "runtime_plugin_bundle_hash": "abc123",
    }


def valid_thresholds_dict() -> dict[str, object]:
    return {
        "throughput_min": 10.0,
        "p50_max": 100.0,
        "p95_max": 200.0,
        "startup_max": 5000.0,
        "rss_peak_kb_max": 500000.0,
        "input_drop_count_max": 0.0,
        "output_drop_count_max": 0.0,
        "regression_tolerance_percent": 10.0,
    }


def valid_scenario_dict(scenario_id: str) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "model_id": "synthetic",
        "pipeline_id": "pass_through",
        "run_mode": "async",
        "iterations": 100,
        "metrics_thresholds": valid_thresholds_dict(),
    }


def valid_component_row() -> dict[str, object]:
    return {
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
        "p99_ms": 0.8,
        "max_ms": 0.95,
    }


class PerfSchemaTest(unittest.TestCase):
    def test_parse_profile_ok(self) -> None:
        profile = schema.parse_profile(valid_profile_dict())
        self.assertEqual(profile.modalix_profile_id, "modalix_default")

    def test_unknown_field_rejected(self) -> None:
        bad = valid_scenario_dict("runtime_session_sync_rgb")
        bad["unexpected"] = 1
        with self.assertRaises(schema.SchemaError):
            schema.parse_scenario_baseline(bad)

    def test_missing_metric_rejected(self) -> None:
        payload = {
            "scenario_id": "runtime_session_sync_rgb",
            "throughput": 12.0,
            "p50": 1.0,
            # p95 missing on purpose
            "startup": 10.0,
            "rss_peak_kb": 10000.0,
            "input_drop_count": 0.0,
            "output_drop_count": 0.0,
        }
        with self.assertRaises(schema.SchemaError):
            schema.parse_metrics_payload(payload)

    def test_optional_power_payload_is_metadata_not_metric(self) -> None:
        payload = {
            "scenario_id": "runtime_session_sync_rgb",
            "throughput": 12.0,
            "p50": 1.0,
            "p95": 2.0,
            "startup": 10.0,
            "rss_peak_kb": 10000.0,
            "input_drop_count": 0.0,
            "output_drop_count": 0.0,
            "power": {"samples": 2, "total_avg_watts": 3.5},
            "measure_report": {"schema": "sima.neat.measure_report", "throughput_batches_per_s": 12.0},
        }
        metrics = schema.parse_metrics_payload(payload)
        self.assertEqual(metrics["throughput"], 12.0)
        self.assertEqual(schema.parse_optional_power_payload(payload)["total_avg_watts"], 3.5)
        self.assertEqual(
            schema.parse_optional_measure_report_payload(payload)["schema"], "sima.neat.measure_report"
        )

    def test_optional_power_payload_must_be_object(self) -> None:
        payload = {
            "throughput": 12.0,
            "p50": 1.0,
            "p95": 2.0,
            "startup": 10.0,
            "rss_peak_kb": 10000.0,
            "input_drop_count": 0.0,
            "output_drop_count": 0.0,
            "power": 3.5,
        }
        with self.assertRaises(schema.SchemaError):
            schema.parse_optional_power_payload(payload)

    def test_component_thresholds_and_exact_rows_parse(self) -> None:
        baseline_data = valid_scenario_dict("ssd_mobilenet_boxdecode")
        baseline_data["component_latency_thresholds"] = {
            "boxdecode_plugin_exclusive": {
                "required": True,
                "min_samples": 1000,
                "p99_max_ms": 0.9,
                "max_max_ms": 0.999,
            }
        }
        baseline = schema.parse_scenario_baseline(baseline_data)
        self.assertEqual(
            baseline.component_latency_thresholds["boxdecode_plugin_exclusive"].min_samples,
            1000,
        )

        components = schema.parse_optional_component_latency_payload(
            {"component_latency": {"boxdecode_plugin_exclusive": [valid_component_row()]}}
        )
        self.assertEqual(components["boxdecode_plugin_exclusive"][0].p99_ms, 0.8)

    def test_component_threshold_requires_an_absolute_cap(self) -> None:
        baseline_data = valid_scenario_dict("ssd_mobilenet_boxdecode")
        baseline_data["component_latency_thresholds"] = {
            "boxdecode": {"required": True, "min_samples": 1000}
        }
        with self.assertRaises(schema.SchemaError):
            schema.parse_scenario_baseline(baseline_data)

    def test_component_rows_are_arrays_and_strictly_typed(self) -> None:
        with self.assertRaises(schema.SchemaError):
            schema.parse_optional_component_latency_payload(
                {"component_latency": {"boxdecode": valid_component_row()}}
            )
        bad_row = valid_component_row()
        bad_row["reliable"] = 1
        with self.assertRaises(schema.SchemaError):
            schema.parse_optional_component_latency_payload(
                {"component_latency": {"boxdecode": [bad_row]}}
            )

    def test_validate_baseline_directory_rejects_malformed_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "profile.json").write_text(
                "{\n"
                '  "modalix_profile_id": "modalix_default",\n'
                '  "board_class": "modalix-v1",\n'
                '  "sdk_version": "2.0.0",\n'
                '  "compiler": "g++-12",\n'
                '  "gstreamer_version": "1.20.3",\n'
                '  "runtime_plugin_bundle_hash": "abc123"\n'
                "}\n",
                encoding="utf-8",
            )
            malformed = valid_scenario_dict("runtime_session_sync_rgb")
            malformed["metrics_thresholds"] = {"throughput_min": 1.0}
            (root / "runtime_session_sync_rgb.json").write_text(
                json.dumps(malformed), encoding="utf-8"
            )

            with self.assertRaises(schema.SchemaError):
                schema.validate_baseline_directory(root)


if __name__ == "__main__":
    unittest.main()
