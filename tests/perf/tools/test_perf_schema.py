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
