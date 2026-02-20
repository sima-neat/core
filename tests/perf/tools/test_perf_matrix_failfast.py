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
import run_perf_matrix


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class PerfMatrixFailfastTest(unittest.TestCase):
    def test_missing_baseline_emits_harness_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_dir = root / "profile"
            results_dir = root / "results"

            write_json(
                profile_dir / "profile.json",
                {
                    "modalix_profile_id": "modalix_default",
                    "board_class": "modalix-v1",
                    "sdk_version": "2.0.0",
                    "compiler": "g++-12",
                    "gstreamer_version": "1.20.3",
                    "runtime_plugin_bundle_hash": "abc123",
                },
            )
            # Only one scenario baseline; the rest should fail fast as missing.
            write_json(
                profile_dir / "mpk_parse_smoke.json",
                {
                    "scenario_id": "mpk_parse_smoke",
                    "model_id": "none",
                    "pipeline_id": "mpk_parse",
                    "run_mode": "parse",
                    "iterations": 100,
                    "metrics_thresholds": {
                        "throughput_min": 1.0,
                        "p50_max": 1.0,
                        "p95_max": 1.0,
                        "startup_max": 1.0,
                        "rss_peak_kb_max": 100000.0,
                        "input_drop_count_max": 0.0,
                        "output_drop_count_max": 0.0,
                        "regression_tolerance_percent": 10.0,
                    },
                },
            )

            profile, _, failed = run_perf_matrix.preflight_baselines(profile_dir, results_dir)
            self.assertTrue(failed)
            self.assertIsNotNone(profile)

            missing_result = schema.load_perf_result(results_dir / "runtime_session_sync_rgb.json")
            self.assertEqual(missing_result.failure_class, schema.FailureClass.HARNESS_ERROR)
            self.assertEqual(missing_result.reason_code, schema.ReasonCode.HARNESS_BASELINE_MISSING)

    def test_malformed_profile_emits_schema_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_dir = root / "profile"
            results_dir = root / "results"

            # Invalid profile: missing required fields.
            write_json(profile_dir / "profile.json", {"modalix_profile_id": "broken"})

            _, _, failed = run_perf_matrix.preflight_baselines(profile_dir, results_dir)
            self.assertTrue(failed)

            result = schema.load_perf_result(results_dir / "runtime_graph_join_bundle.json")
            self.assertEqual(result.failure_class, schema.FailureClass.HARNESS_ERROR)
            self.assertEqual(result.reason_code, schema.ReasonCode.HARNESS_SCHEMA_INVALID)


if __name__ == "__main__":
    unittest.main()
