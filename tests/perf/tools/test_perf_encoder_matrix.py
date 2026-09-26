#!/usr/bin/env python3
"""Runner control-flow checks with synthetic process results; no encoder/device execution."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import perf_schema as schema
import run_perf_matrix as runner
from test_perf_matrix_failfast import component_baseline


class EncoderMatrixTest(unittest.TestCase):
    def setUp(self):
        self.spec = next(s for s in runner.ENCODER_SCENARIOS if s.scenario_id ==
                         "runtime_encoder_raw_sender_h264_dma")
        self.profile = schema.PerfProfile("profile", "board", "sdk", "cc", "gst", "bundle")
        self.workload = {"warmup_frames": 200, "pattern": "static-blocks-v1",
                         "input_fingerprint": {"sha256": "a" * 64}}
        self.settings = {"num-output-buffers": "4"}
        original = component_baseline()
        self.baseline = replace(
            original, scenario_id=self.spec.scenario_id, run_mode="unpaced",
            component_latency_thresholds={},
            metrics_thresholds=replace(original.metrics_thresholds, throughput_min=100),
            encoder_reference={"workload": self.workload, "native_encoder_settings": self.settings},
        )
        self.calls = []

    def payload(self, fps, paced=False):
        iterations = math.ceil(fps * 60) if paced else round(fps * 10)
        return {
            "scenario_id": self.spec.scenario_id, "run_mode": "paced" if paced else "unpaced",
            "throughput": iterations / (60 if paced else 10), "p50": fps / 10,
            "p95": 20, "startup": 100, "rss_peak_kb": 10000,
            "input_drop_count": 0, "output_drop_count": 0, "failure": "",
            "iterations": iterations, "measured_seconds": 60 if paced else 10,
            "workload": self.workload, "native_encoder_settings": self.settings,
            "counts": {key: iterations + 200 for key in (
                "attempted", "accepted", "completed", "output", "sent_frames", "sent_packets", "received_packets")},
            "completion": {"fps": fps, "p50_ms": 1, "p95_ms": 2, "seconds": 60 if paced else 10},
            "pacing": {"target_fps": fps, "target_frames": iterations, "shortfall_frames": 0,
                       "producer_seconds": 60},
        }

    def run_case(self, alter=None):
        def fake_run(command, **kwargs):
            self.calls.append((command, kwargs))
            index = len(self.calls) - 1
            payload = self.payload(float(command[-1]) * .95, True) if index == 3 else self.payload((120, 100, 110)[index])
            if alter:
                alter(index, payload)
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "diagnostic")
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "run_cmd", side_effect=fake_run):
            result = runner.run_scenario(
                repo_root=Path(tmp), executable_dir=Path(tmp), results_dir=Path(tmp)/"results",
                profile=self.profile, spec=self.spec, baseline=self.baseline,
                timeout_sec=10, iterations_override=1,
            )
            artifacts = Path(result.run_meta["artifacts"])
            self.assertTrue((artifacts/"unpaced-1.stdout.json").exists())
            return result

    def test_median_run_metrics_and_separate_paced_json_are_retained(self):
        result = self.run_case()
        self.assertEqual(result.status, schema.ResultStatus.PASS)
        self.assertEqual(result.metrics["throughput"], 110)
        self.assertEqual(result.metrics["p50"], 11)
        self.assertEqual(result.run_meta["median_run"], 3)
        self.assertEqual(len(result.run_meta["runs"]), 4)
        self.assertIn("completion", result.run_meta["runs"][3]["payload"])
        self.assertEqual(self.calls[-1][0][-2:], ["--median-fps", "110.0"])
        self.assertEqual(self.calls[-1][1]["timeout_sec"], 300)
        self.assertNotIn("SIMA_PERF_ITERS", self.calls[0][1]["env"])

    def test_sender_loss_cannot_pass_or_continue_to_pacing(self):
        def lose(index, payload):
            if index == 1:
                payload["counts"]["received_packets"] -= 1
        result = self.run_case(lose)
        self.assertEqual(result.status, schema.ResultStatus.FAIL)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("sender/receiver", result.run_meta["error"])

    def test_paced_shortfall_fails_even_when_process_returns_zero(self):
        result = self.run_case(lambda index, payload: payload["pacing"].update(shortfall_frames=1) if index == 3 else None)
        self.assertEqual(result.status, schema.ResultStatus.FAIL)
        self.assertEqual(len(self.calls), 4)
        self.assertIn("shortfall", result.run_meta["error"])

    def test_mismatched_reference_does_not_become_a_regression_result(self):
        result = self.run_case(lambda index, payload: payload.update(native_encoder_settings={"num-output-buffers": "8"}))
        self.assertEqual(result.failure_class, schema.FailureClass.HARNESS_ERROR)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("saved Core reference", result.run_meta["error"])

    def test_only_median_compares_to_saved_reference(self):
        self.baseline = replace(self.baseline, metrics_thresholds=replace(
            self.baseline.metrics_thresholds, throughput_min=123,
        ))
        result = self.run_case()
        self.assertEqual(result.reason_code, schema.ReasonCode.REGRESSION_THROUGHPUT)
        self.assertEqual(len(self.calls), 4)

    def test_encoder_references_cannot_omit_identity_or_allow_zero_fps_or_loss(self):
        good = asdict(self.baseline)
        self.assertEqual(schema.parse_scenario_baseline(good), self.baseline)
        for field, value in (("throughput_min", 0), ("regression_tolerance_percent", 11),
                             ("input_drop_count_max", 1), ("output_drop_count_max", 1)):
            bad = asdict(self.baseline)
            bad["metrics_thresholds"][field] = value
            with self.subTest(field=field), self.assertRaises(schema.SchemaError):
                schema.parse_scenario_baseline(bad)
        missing_input = asdict(self.baseline)
        del missing_input["encoder_reference"]["workload"]["input_fingerprint"]
        with self.assertRaises(schema.SchemaError):
            schema.parse_scenario_baseline(missing_input)
        del good["encoder_reference"]
        with self.assertRaises(schema.SchemaError):
            schema.parse_scenario_baseline(good)

    def test_failed_and_timed_out_processes_retain_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(repo_root=Path(tmp), executable_dir=Path(tmp), results_dir=Path(tmp)/"results",
                          profile=self.profile, spec=self.spec, baseline=self.baseline, timeout_sec=10)
            with patch.object(runner, "run_cmd", return_value=subprocess.CompletedProcess([], 77, "partial", "missing native plugin")):
                result = runner.run_encoder_scenario(**kwargs)
            self.assertEqual(result.status, schema.ResultStatus.FAIL)
            self.assertEqual(result.run_meta["runs"][0]["exit_code"], 77)
            with patch.object(runner, "run_cmd", side_effect=subprocess.TimeoutExpired("encoder", 300, output=b"partial")):
                result = runner.run_encoder_scenario(**kwargs)
            self.assertEqual(result.reason_code, schema.ReasonCode.ENV_TIMEOUT)
            self.assertEqual((Path(result.run_meta["artifacts"])/"unpaced-1.stdout.json").read_text(), "partial")


if __name__ == "__main__":
    unittest.main()
