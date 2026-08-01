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

import perf_schema as schema  # noqa: E402
import run_perf_matrix  # noqa: E402


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def component_baseline() -> schema.ScenarioBaseline:
    return schema.parse_scenario_baseline(
        {
            "scenario_id": "ssd_mobilenet_boxdecode",
            "model_id": "ssd",
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
                "regression_tolerance_percent": 10.0,
            },
            "component_latency_thresholds": {
                "boxdecode_plugin_exclusive": {
                    "required": True,
                    "min_samples": 1000,
                    "p99_max_ms": 0.9,
                }
            },
        }
    )


def passing_payload() -> dict[str, object]:
    return {
        "scenario_id": "ssd_mobilenet_boxdecode",
        "iterations": 1000,
        "run_mode": "sync",
        "throughput": 100.0,
        "p50": 10.0,
        "p95": 11.0,
        "startup": 100.0,
        "rss_peak_kb": 10000.0,
        "input_drop_count": 0.0,
        "output_drop_count": 0.0,
    }


def write_fake_perf(executable: Path, payload: dict[str, object]) -> None:
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        f"print(json.dumps({payload!r}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)


def write_fake_exit(executable: Path, return_code: int) -> None:
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"print('fake exit {return_code}', file=sys.stderr)\n"
        f"raise SystemExit({return_code})\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)


class PerfMatrixFailfastTest(unittest.TestCase):
    def test_ssd_scenario_is_weekly_only(self) -> None:
        standard_ids = {spec.scenario_id for spec in run_perf_matrix.STANDARD_SCENARIOS}
        long_ids = {spec.scenario_id for spec in run_perf_matrix.LONG_SCENARIOS}

        self.assertNotIn("ssd_mobilenet_boxdecode", standard_ids)
        self.assertIn("ssd_mobilenet_boxdecode", long_ids)
        self.assertTrue(run_perf_matrix.LONG_SCENARIOS[0].allow_skip)
        self.assertTrue(all(not spec.allow_skip for spec in run_perf_matrix.STANDARD_SCENARIOS))
        self.assertEqual(
            run_perf_matrix.SCENARIOS,
            run_perf_matrix.STANDARD_SCENARIOS + run_perf_matrix.LONG_SCENARIOS,
        )
        self.assertEqual(
            tuple(spec.scenario_id for spec in run_perf_matrix.STANDARD_SCENARIOS),
            schema.STANDARD_SCENARIO_IDS,
        )
        self.assertEqual(
            tuple(spec.scenario_id for spec in run_perf_matrix.LONG_SCENARIOS),
            schema.LONG_SCENARIO_IDS,
        )

    def test_only_skippable_scenario_honors_return_code_77(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable_dir = root / "tests"
            executable = executable_dir / "fake_skip"
            write_fake_exit(executable, run_perf_matrix.SKIP_RETURN_CODE)
            profile = schema.PerfProfile("modalix", "board", "sdk", "cc", "gst", "bundle")
            baseline = component_baseline()

            skipped = run_perf_matrix.run_scenario(
                repo_root=root,
                executable_dir=executable_dir,
                results_dir=root / "results",
                profile=profile,
                spec=run_perf_matrix.ScenarioSpec(
                    "ssd_mobilenet_boxdecode", executable.name, allow_skip=True
                ),
                baseline=baseline,
                timeout_sec=10,
                iterations_override=None,
            )
            self.assertEqual(skipped.status, schema.ResultStatus.SKIP)
            self.assertIsNone(skipped.failure_class)
            self.assertEqual(skipped.run_meta["phase"], "skip")
            self.assertEqual(skipped.run_meta["exit_code"], run_perf_matrix.SKIP_RETURN_CODE)

            strict = run_perf_matrix.run_scenario(
                repo_root=root,
                executable_dir=executable_dir,
                results_dir=root / "results",
                profile=profile,
                spec=run_perf_matrix.ScenarioSpec(
                    "ssd_mobilenet_boxdecode", executable.name
                ),
                baseline=baseline,
                timeout_sec=10,
                iterations_override=None,
            )
            self.assertEqual(strict.status, schema.ResultStatus.FAIL)
            self.assertEqual(strict.failure_class, schema.FailureClass.ENV_BROKEN)

            write_fake_exit(executable, 1)
            non_skip_failure = run_perf_matrix.run_scenario(
                repo_root=root,
                executable_dir=executable_dir,
                results_dir=root / "results",
                profile=profile,
                spec=run_perf_matrix.ScenarioSpec(
                    "ssd_mobilenet_boxdecode", executable.name, allow_skip=True
                ),
                baseline=baseline,
                timeout_sec=10,
                iterations_override=None,
            )
            self.assertEqual(non_skip_failure.status, schema.ResultStatus.FAIL)
            self.assertEqual(non_skip_failure.failure_class, schema.FailureClass.ENV_BROKEN)

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
            executable_dir = root / "prebuilt-tests"
            results_dir = root / "results"
            exe = executable_dir / "fake_power_perf"
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
                executable_dir=executable_dir,
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

    def test_run_scenario_component_p99_is_an_absolute_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "tests" / "fake_component_perf"
            payload = passing_payload()
            payload["component_latency"] = {
                "boxdecode_plugin_exclusive": [
                    {
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
                        "p99_ms": 0.901,
                        "max_ms": 0.95,
                    }
                ]
            }
            write_fake_perf(executable, payload)
            profile = schema.PerfProfile("modalix", "board", "sdk", "cc", "gst", "bundle")
            result = run_perf_matrix.run_scenario(
                repo_root=root,
                executable_dir=executable.parent,
                results_dir=root / "results",
                profile=profile,
                spec=run_perf_matrix.ScenarioSpec(
                    "ssd_mobilenet_boxdecode", executable.name
                ),
                baseline=component_baseline(),
                timeout_sec=10,
                iterations_override=None,
            )
            self.assertEqual(result.failure_class, schema.FailureClass.REGRESSION)
            self.assertEqual(result.reason_code, schema.ReasonCode.REGRESSION_COMPONENT_P99)
            self.assertEqual(
                result.run_meta["component_latency"]["boxdecode_plugin_exclusive"][0]["samples"],
                1000,
            )

    def test_run_scenario_missing_required_component_is_harness_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "tests" / "fake_component_perf"
            write_fake_perf(executable, passing_payload())
            profile = schema.PerfProfile("modalix", "board", "sdk", "cc", "gst", "bundle")
            result = run_perf_matrix.run_scenario(
                repo_root=root,
                executable_dir=executable.parent,
                results_dir=root / "results",
                profile=profile,
                spec=run_perf_matrix.ScenarioSpec(
                    "ssd_mobilenet_boxdecode", executable.name
                ),
                baseline=component_baseline(),
                timeout_sec=10,
                iterations_override=None,
            )
            self.assertEqual(result.failure_class, schema.FailureClass.HARNESS_ERROR)
            self.assertEqual(result.reason_code, schema.ReasonCode.HARNESS_COMPONENT_MISSING)

    def test_scenario_side_component_marker_is_not_misclassified_as_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "tests" / "fake_component_perf"
            executable.parent.mkdir(parents=True)
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "print('PERF_HARNESS_ERROR:HARNESS_COMPONENT_AMBIGUOUS: matches=2', "
                "file=sys.stderr)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            profile = schema.PerfProfile("modalix", "board", "sdk", "cc", "gst", "bundle")
            result = run_perf_matrix.run_scenario(
                repo_root=root,
                executable_dir=executable.parent,
                results_dir=root / "results",
                profile=profile,
                spec=run_perf_matrix.ScenarioSpec(
                    "ssd_mobilenet_boxdecode", executable.name
                ),
                baseline=component_baseline(),
                timeout_sec=10,
                iterations_override=None,
            )
            self.assertEqual(result.failure_class, schema.FailureClass.HARNESS_ERROR)
            self.assertEqual(result.reason_code, schema.ReasonCode.HARNESS_COMPONENT_AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
