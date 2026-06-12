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
                profile_dir / "runtime_graph_fanout.json",
                {
                    "scenario_id": "runtime_graph_fanout",
                    "model_id": "none",
                    "pipeline_id": "runtime_graph_fanout",
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

    def test_run_scenario_copies_power_payload_to_run_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            results_dir = root / "results"
            exe = build_dir / "tests" / "fake_power_perf"
            exe.parent.mkdir(parents=True)
            exe.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "print(json.dumps({\n"
                "  'scenario_id': 'runtime_session_sync_rgb',\n"
                "  'iterations': 1,\n"
                "  'run_mode': 'sync',\n"
                "  'throughput': 100.0,\n"
                "  'p50': 1.0,\n"
                "  'p95': 2.0,\n"
                "  'startup': 3.0,\n"
                "  'rss_peak_kb': 4.0,\n"
                "  'input_drop_count': 0.0,\n"
                "  'output_drop_count': 0.0,\n"
                "  'power': {'samples': 2, 'total_avg_watts': 5.5},\n"
                "  'measure_report': {'schema': 'sima.neat.measure_report', 'throughput_batches_per_s': 100.0},\n"
                "}))\n",
                encoding="utf-8",
            )
            exe.chmod(0o755)

            profile = schema.PerfProfile(
                modalix_profile_id="modalix_default",
                board_class="modalix-v1",
                sdk_version="2.0.0",
                compiler="g++",
                gstreamer_version="1",
                runtime_plugin_bundle_hash="abc",
            )
            baseline = schema.ScenarioBaseline(
                scenario_id="runtime_session_sync_rgb",
                model_id="synthetic",
                pipeline_id="pass",
                run_mode="sync",
                iterations=1,
                metrics_thresholds=schema.MetricsThresholds(
                    throughput_min=1.0,
                    p50_max=10.0,
                    p95_max=10.0,
                    startup_max=10.0,
                    rss_peak_kb_max=100.0,
                    input_drop_count_max=0.0,
                    output_drop_count_max=0.0,
                    regression_tolerance_percent=0.0,
                ),
            )
            result = run_perf_matrix.run_scenario(
                repo_root=root,
                build_dir=build_dir,
                results_dir=results_dir,
                profile=profile,
                spec=run_perf_matrix.ScenarioSpec("runtime_session_sync_rgb", "fake_power_perf"),
                baseline=baseline,
                timeout_sec=10,
                iterations_override=1,
            )

            self.assertEqual(result.status, schema.ResultStatus.PASS)
            self.assertEqual(result.run_meta["power"]["total_avg_watts"], 5.5)
            self.assertEqual(result.run_meta["measure_report"]["schema"], "sima.neat.measure_report")


if __name__ == "__main__":
    unittest.main()
