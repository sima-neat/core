#!/usr/bin/env python3
"""Perf matrix orchestrator: strict schema checks + scenario comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import perf_schema as schema  # noqa: E402


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    target: str
    allow_skip: bool = False
    args: tuple[str, ...] = ()


SKIP_RETURN_CODE = 77


STANDARD_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("runtime_session_sync_rgb", "perf_runtime_graph_sync_rgb_test"),
    ScenarioSpec("runtime_session_async_rgb", "perf_runtime_graph_async_rgb_test"),
    ScenarioSpec("runtime_graph_fanout", "perf_runtime_graph_fanout_test"),
    ScenarioSpec("runtime_graph_join_bundle", "perf_runtime_graph_join_bundle_test"),
    ScenarioSpec("runtime_codec_mjpeg_decode", "perf_runtime_codec_mjpeg_decode_test"),
    ScenarioSpec("runtime_codec_h264_decode", "perf_runtime_codec_h264_decode_test"),
    ScenarioSpec("runtime_codec_h265_decode", "perf_runtime_codec_h265_decode_test"),
    ScenarioSpec("runtime_model_archive_load", "perf_runtime_model_archive_load_test"),
)

ENCODER_SCENARIOS: tuple[ScenarioSpec, ...] = tuple(
    ScenarioSpec(scenario_id, "perf_runtime_encoder_test", args=(
        "--path", path, "--codec", codec, "--input", memory,
    ))
    for scenario_id, (path, codec, memory) in zip(schema.ENCODER_SCENARIO_IDS, schema.ENCODER_CASES)
)

LONG_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "ssd_mobilenet_boxdecode",
        "perf_ssd_mobilenet_boxdecode_test",
        allow_skip=True,
    ),
)

SCENARIOS: tuple[ScenarioSpec, ...] = STANDARD_SCENARIOS + LONG_SCENARIOS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scenario_result_path(results_dir: Path, scenario_id: str) -> Path:
    return results_dir / f"{scenario_id}.json"


def build_result(
    *,
    scenario_id: str,
    modalix_profile_id: str,
    status: schema.ResultStatus,
    failure_class: schema.FailureClass | None,
    reason_code: schema.ReasonCode | None,
    metrics: dict[str, float] | None,
    run_meta: dict[str, Any] | None,
) -> schema.PerfResult:
    payload = {
        "scenario_id": scenario_id,
        "modalix_profile_id": modalix_profile_id,
        "status": status.value,
        "failure_class": failure_class.value if failure_class else None,
        "reason_code": reason_code.value if reason_code else None,
        "metrics": metrics if metrics is not None else dict(schema.DEFAULT_EMPTY_METRICS),
        "run_meta": run_meta if run_meta is not None else {},
        "timestamp": utc_now(),
    }
    return schema.parse_perf_result(payload, context=f"generated:{scenario_id}")


def write_result(results_dir: Path, result: schema.PerfResult) -> None:
    schema.write_result(scenario_result_path(results_dir, result.scenario_id), result)


def preflight_baselines(
    profile_dir: Path,
    results_dir: Path,
    scenarios: tuple[ScenarioSpec, ...] = SCENARIOS,
    result_scenarios: tuple[ScenarioSpec, ...] | None = None,
) -> tuple[schema.PerfProfile | None, dict[str, schema.ScenarioBaseline], bool]:
    results_dir.mkdir(parents=True, exist_ok=True)
    report_specs = scenarios if result_scenarios is None else result_scenarios

    try:
        profile, baseline_map = schema.validate_baseline_directory(profile_dir)
    except schema.SchemaError as exc:
        for spec in report_specs:
            result = build_result(
                scenario_id=spec.scenario_id,
                modalix_profile_id="unknown",
                status=schema.ResultStatus.FAIL,
                failure_class=schema.FailureClass.HARNESS_ERROR,
                reason_code=schema.ReasonCode.HARNESS_SCHEMA_INVALID,
                metrics=dict(schema.DEFAULT_EMPTY_METRICS),
                run_meta={"phase": "preflight", "error": str(exc)},
            )
            write_result(results_dir, result)
        return None, {}, True

    expected = {spec.scenario_id for spec in scenarios}
    extras = sorted(set(baseline_map.keys()) - expected)
    if extras:
        for spec in report_specs:
            result = build_result(
                scenario_id=spec.scenario_id,
                modalix_profile_id=profile.modalix_profile_id,
                status=schema.ResultStatus.FAIL,
                failure_class=schema.FailureClass.HARNESS_ERROR,
                reason_code=schema.ReasonCode.HARNESS_SCHEMA_INVALID,
                metrics=dict(schema.DEFAULT_EMPTY_METRICS),
                run_meta={"phase": "preflight", "unexpected_scenarios": extras},
            )
            write_result(results_dir, result)
        return profile, baseline_map, True

    missing = [spec.scenario_id for spec in scenarios if spec.scenario_id not in baseline_map]
    if missing:
        for spec in report_specs:
            result = build_result(
                scenario_id=spec.scenario_id,
                modalix_profile_id=profile.modalix_profile_id,
                status=schema.ResultStatus.FAIL,
                failure_class=schema.FailureClass.HARNESS_ERROR,
                reason_code=schema.ReasonCode.HARNESS_BASELINE_MISSING,
                metrics=dict(schema.DEFAULT_EMPTY_METRICS),
                run_meta={"phase": "preflight", "missing_scenarios": missing},
            )
            write_result(results_dir, result)
        return profile, baseline_map, True

    return profile, baseline_map, False


def run_cmd(cmd: list[str], cwd: Path, timeout_sec: int | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_sec,
    )


def configure_and_build(repo_root: Path, build_dir: Path, targets: list[str]) -> tuple[bool, str]:
    configure_cmd = ["cmake", "-S", str(repo_root), "-B", str(build_dir)]
    configure_proc = run_cmd(configure_cmd, cwd=repo_root)
    if configure_proc.returncode != 0:
        detail = (
            "cmake configure failed\n"
            f"stdout:\n{configure_proc.stdout}\n"
            f"stderr:\n{configure_proc.stderr}"
        )
        return False, detail

    build_cmd = ["cmake", "--build", str(build_dir), "--target", *targets]
    build_level = os.getenv("CMAKE_BUILD_PARALLEL_LEVEL", "8")
    build_cmd.append(f"-j{build_level}")
    build_proc = run_cmd(build_cmd, cwd=repo_root)
    if build_proc.returncode != 0:
        detail = (
            "cmake build failed\n"
            f"stdout:\n{build_proc.stdout}\n"
            f"stderr:\n{build_proc.stderr}"
        )
        return False, detail

    return True, ""


def run_modalix_preflight(repo_root: Path, ctest_dir: Path) -> tuple[bool, str]:
    proc = run_cmd(
        [
            "ctest",
            "--test-dir",
            str(ctest_dir),
            "--output-on-failure",
            "-R",
            "^unit_modalix_contract_preflight_test$",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        detail = (
            "modalix preflight failed\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        return False, detail
    return True, ""


def validate_encoder_payload(
    payload: dict[str, Any], spec: ScenarioSpec, baseline: schema.ScenarioBaseline,
    median_fps: float | None,
) -> dict[str, float]:
    """Check emitter evidence before it can become a median/reference comparison."""
    def require(ok: bool, message: str) -> None:
        if not ok:
            raise schema.SchemaError(f"{spec.scenario_id}: {message}")

    metrics = schema.parse_metrics_payload(payload)
    paced = median_fps is not None
    require(payload["scenario_id"] == spec.scenario_id, "wrong scenario payload")
    require(payload["run_mode"] == ("paced" if paced else "unpaced"), "wrong run mode")
    require(payload["failure"] == "", "emitter reported a failure")
    require(metrics["throughput"] > 0, "non-positive measured throughput")
    require(metrics["input_drop_count"] == metrics["output_drop_count"] == 0, "Core dropped frames")
    reference = {key: payload[key] for key in ("workload", "native_encoder_settings")}
    require(reference == baseline.encoder_reference, "workload/settings differ from saved Core reference")
    require(payload["workload"]["warmup_frames"] == 200, "warmup must be 200 frames")
    counts = payload["counts"]
    require(all(type(counts[key]) is int and counts[key] >= 0 for key in (
        "attempted", "accepted", "completed", "output", "sent_frames", "sent_packets", "received_packets"
    )), "invalid frame/packet counts")
    require(counts["attempted"] == counts["accepted"] == counts["completed"] == counts["output"],
            "attempted/accepted/completed/output mismatch")
    require(payload["iterations"] == counts["output"] - 200, "measured frame count mismatch")
    duration = payload["measured_seconds"]
    require(math.isfinite(duration) and duration >= (60 if paced else 10), "measurement is too short")
    require(math.isclose(metrics["throughput"], payload["iterations"] / duration, rel_tol=1e-6),
            "throughput disagrees with frame count/duration")
    if "--path" in spec.args and spec.args[spec.args.index("--path") + 1].endswith("sender"):
        require(counts["sent_frames"] == counts["accepted"] and
                counts["sent_packets"] == counts["received_packets"] >= counts["sent_frames"],
                "sender/receiver accounting differs")
    if paced:
        pacing = payload["pacing"]
        target = math.ceil(.95 * median_fps * 60)
        require(math.isclose(pacing["target_fps"], .95 * median_fps, rel_tol=1e-6), "wrong paced rate")
        require(pacing["target_frames"] == target == payload["iterations"] and
                pacing["shortfall_frames"] == 0, "paced producer shortfall")
        require(math.isfinite(pacing["producer_seconds"]) and pacing["producer_seconds"] >= 60,
                "producer did not run for 60 seconds")
    else:
        require(payload["iterations"] >= 1000, "fewer than 1000 measured frames")
    completion = payload["completion"]
    require(all(math.isfinite(completion[key]) and completion[key] >= 0
                for key in ("fps", "p50_ms", "p95_ms", "seconds")) and completion["fps"] > 0,
            "invalid completion timing")
    return metrics


def run_encoder_scenario(
    *, repo_root: Path, executable_dir: Path, results_dir: Path, profile: schema.PerfProfile,
    spec: ScenarioSpec, baseline: schema.ScenarioBaseline, timeout_sec: int,
) -> schema.PerfResult:
    """Three exclusive max-rate runs, then one separate run paced at 95% of their median."""
    raw_root = results_dir / "encoder"
    raw_root.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=spec.scenario_id + "-", dir=raw_root))
    metadata: dict[str, Any] = {
        "phase": "encoder_run", "artifacts": str(artifacts), "runs": [],
        "reference": baseline.encoder_reference,
        "reference_throughput": baseline.metrics_thresholds.throughput_min,
    }
    metrics = None

    def finish(failure_class: schema.FailureClass | None = None,
               reason: schema.ReasonCode | None = None) -> schema.PerfResult:
        return build_result(
            scenario_id=spec.scenario_id, modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL if failure_class else schema.ResultStatus.PASS,
            failure_class=failure_class, reason_code=reason, metrics=metrics, run_meta=metadata,
        )

    env = dict(os.environ)
    env.pop("SIMA_PERF_ITERS", None)  # This emitter owns its duration and minimum count.
    unpaced: list[dict[str, Any]] = []
    median_fps = None
    try:
        for index in range(4):
            phase = f"unpaced-{index + 1}" if index < 3 else "paced"
            command = [str(executable_dir / spec.target), *spec.args]
            if index == 0:
                capture = artifacts / ("reference." + spec.args[spec.args.index("--codec") + 1])
                command += ["--capture", str(capture)]
                metadata["reference_capture"] = str(capture)
            if index == 3:
                # Keep all metrics from the actual run with median FPS, not a synthetic combination.
                median = sorted(unpaced, key=lambda row: row["throughput"])[1]
                median_fps = float(median["throughput"])
                metrics = schema.parse_metrics_payload(median)
                metadata["median_run"] = unpaced.index(median) + 1
                metadata["median_fps"] = median_fps
                command += ["--median-fps", repr(median_fps)]
            record = {"phase": phase, "command": command}
            metadata["runs"].append(record)
            proc = run_cmd(command, cwd=repo_root, timeout_sec=max(timeout_sec, 300), env=env)
            record["exit_code"] = proc.returncode
            (artifacts / f"{phase}.stdout.json").write_text(proc.stdout or "", encoding="utf-8")
            (artifacts / f"{phase}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            try:
                record["payload"] = json.loads(proc.stdout)
            except json.JSONDecodeError:
                if proc.returncode == 0:
                    raise
            if proc.returncode != 0:
                reason = schema.classify_env_failure(
                    proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), timed_out=False,
                )
                return finish(schema.FailureClass.ENV_BROKEN, reason)
            payload = record["payload"]
            if not isinstance(payload, dict):
                raise schema.SchemaError("encoder payload must be an object")
            validate_encoder_payload(payload, spec, baseline, median_fps)
            if index < 3:
                unpaced.append(payload)
    except subprocess.TimeoutExpired as error:
        # Preserve partial diagnostics even when the emitter cannot return JSON.
        for suffix, value in (("stdout.json", error.stdout), ("stderr.log", error.stderr)):
            text = value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            (artifacts / f"{phase}.{suffix}").write_text(text, encoding="utf-8")
        metadata["error"] = str(error)
        return finish(schema.FailureClass.ENV_BROKEN, schema.ReasonCode.ENV_TIMEOUT)
    except (schema.SchemaError, ValueError, KeyError, TypeError) as error:
        metadata["error"] = str(error)
        return finish(schema.FailureClass.HARNESS_ERROR, schema.ReasonCode.HARNESS_SCHEMA_INVALID)
    except OSError as error:
        metadata["error"] = str(error)
        return finish(schema.FailureClass.ENV_BROKEN, schema.ReasonCode.ENV_RUNTIME_CRASH)

    metadata["phase"] = "compare"
    component_failures = schema.compare_component_latency(
        schema.parse_optional_component_latency_payload(median), baseline,
    )
    if component_failures:
        metadata["component_latency_failures"] = [
            schema.component_failure_to_json_dict(failure) for failure in component_failures
        ]
        return finish(component_failures[0].failure_class, component_failures[0].reason_code)
    regressions = schema.compare_metrics(metrics, baseline)
    if regressions:
        metadata["regression_reasons"] = [reason.value for reason in regressions]
        return finish(schema.FailureClass.REGRESSION, regressions[0])
    return finish()


def run_scenario(
    *,
    repo_root: Path,
    executable_dir: Path,
    results_dir: Path,
    profile: schema.PerfProfile,
    spec: ScenarioSpec,
    baseline: schema.ScenarioBaseline,
    timeout_sec: int,
    iterations_override: int | None,
) -> schema.PerfResult:
    if spec.scenario_id in schema.ENCODER_SCENARIO_IDS:
        return run_encoder_scenario(
            repo_root=repo_root, executable_dir=executable_dir, results_dir=results_dir,
            profile=profile, spec=spec, baseline=baseline, timeout_sec=timeout_sec,
        )
    exe_path = executable_dir / spec.target
    if not exe_path.exists():
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.HARNESS_ERROR,
            reason_code=schema.ReasonCode.HARNESS_SCHEMA_INVALID,
            metrics=dict(schema.DEFAULT_EMPTY_METRICS),
            run_meta={"phase": "run", "error": f"missing executable: {exe_path}"},
        )

    scenario_iters = iterations_override if iterations_override is not None else baseline.iterations
    env = dict(os.environ)
    env["SIMA_PERF_ITERS"] = str(scenario_iters)

    try:
        proc = run_cmd([str(exe_path), *spec.args], cwd=repo_root, timeout_sec=timeout_sec, env=env)
    except subprocess.TimeoutExpired:
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.ENV_BROKEN,
            reason_code=schema.ReasonCode.ENV_TIMEOUT,
            metrics=dict(schema.DEFAULT_EMPTY_METRICS),
            run_meta={"phase": "run", "executable": str(exe_path), "timeout_sec": timeout_sec},
        )

    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode == SKIP_RETURN_CODE and spec.allow_skip:
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.SKIP,
            failure_class=None,
            reason_code=None,
            metrics=dict(schema.DEFAULT_EMPTY_METRICS),
            run_meta={
                "phase": "skip",
                "executable": str(exe_path),
                "exit_code": proc.returncode,
                "stdout_tail": proc.stdout[-800:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-800:] if proc.stderr else "",
            },
        )
    if proc.returncode != 0:
        harness_reason = schema.classify_perf_harness_failure(combined_output)
        if harness_reason is not None:
            return build_result(
                scenario_id=spec.scenario_id,
                modalix_profile_id=profile.modalix_profile_id,
                status=schema.ResultStatus.FAIL,
                failure_class=schema.FailureClass.HARNESS_ERROR,
                reason_code=harness_reason,
                metrics=dict(schema.DEFAULT_EMPTY_METRICS),
                run_meta={
                    "phase": "run",
                    "executable": str(exe_path),
                    "exit_code": proc.returncode,
                    "stdout_tail": proc.stdout[-800:] if proc.stdout else "",
                    "stderr_tail": proc.stderr[-800:] if proc.stderr else "",
                },
            )
        reason = schema.classify_env_failure(proc.returncode, combined_output, timed_out=False)
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.ENV_BROKEN,
            reason_code=reason,
            metrics=dict(schema.DEFAULT_EMPTY_METRICS),
            run_meta={
                "phase": "run",
                "executable": str(exe_path),
                "exit_code": proc.returncode,
                "stdout_tail": proc.stdout[-800:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-800:] if proc.stderr else "",
            },
        )

    try:
        payload_raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.HARNESS_ERROR,
            reason_code=schema.ReasonCode.HARNESS_SCHEMA_INVALID,
            metrics=dict(schema.DEFAULT_EMPTY_METRICS),
            run_meta={
                "phase": "parse_output",
                "error": str(exc),
                "stdout_tail": proc.stdout[-800:] if proc.stdout else "",
            },
        )

    if not isinstance(payload_raw, dict):
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.HARNESS_ERROR,
            reason_code=schema.ReasonCode.HARNESS_SCHEMA_INVALID,
            metrics=dict(schema.DEFAULT_EMPTY_METRICS),
            run_meta={"phase": "parse_output", "error": "scenario payload root must be object"},
        )

    try:
        metrics = schema.parse_metrics_payload(payload_raw, context=f"payload:{spec.scenario_id}")
    except schema.SchemaError as exc:
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.HARNESS_ERROR,
            reason_code=schema.ReasonCode.HARNESS_METRIC_MISSING,
            metrics=dict(schema.DEFAULT_EMPTY_METRICS),
            run_meta={"phase": "parse_output", "error": str(exc)},
        )

    try:
        power = schema.parse_optional_power_payload(
            payload_raw, context=f"payload:{spec.scenario_id}"
        )
        measure_report = schema.parse_optional_measure_report_payload(
            payload_raw, context=f"payload:{spec.scenario_id}"
        )
        component_latency = schema.parse_optional_component_latency_payload(
            payload_raw, context=f"payload:{spec.scenario_id}"
        )
    except schema.SchemaError as exc:
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.HARNESS_ERROR,
            reason_code=schema.ReasonCode.HARNESS_SCHEMA_INVALID,
            metrics=metrics,
            run_meta={"phase": "parse_output", "error": str(exc)},
        )

    run_meta = {"phase": "compare", "executable": str(exe_path), "iterations": scenario_iters}
    if power is not None:
        run_meta["power"] = power
    if measure_report is not None:
        run_meta["measure_report"] = measure_report
    if component_latency or baseline.component_latency_thresholds:
        run_meta["component_latency"] = schema.component_latency_to_json_dict(component_latency)

    component_failures = schema.compare_component_latency(component_latency, baseline)
    if component_failures:
        run_meta["component_latency_failures"] = [
            schema.component_failure_to_json_dict(failure) for failure in component_failures
        ]
    harness_failures = [
        failure
        for failure in component_failures
        if failure.failure_class == schema.FailureClass.HARNESS_ERROR
    ]
    if harness_failures:
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.HARNESS_ERROR,
            reason_code=harness_failures[0].reason_code,
            metrics=metrics,
            run_meta=run_meta,
        )

    regressions = schema.compare_metrics(metrics, baseline)
    component_regressions = [
        failure
        for failure in component_failures
        if failure.failure_class == schema.FailureClass.REGRESSION
    ]
    if component_regressions or regressions:
        primary_reason = (
            component_regressions[0].reason_code if component_regressions else regressions[0]
        )
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.REGRESSION,
            reason_code=primary_reason,
            metrics=metrics,
            run_meta={
                **run_meta,
                "regression_reasons": [
                    *[failure.reason_code.value for failure in component_regressions],
                    *[reason.value for reason in regressions],
                ],
            },
        )

    return build_result(
        scenario_id=spec.scenario_id,
        modalix_profile_id=profile.modalix_profile_id,
        status=schema.ResultStatus.PASS,
        failure_class=None,
        reason_code=None,
        metrics=metrics,
        run_meta=run_meta,
    )


def print_summary(results: list[schema.PerfResult]) -> None:
    print("[perf-matrix] scenario summary:")
    for result in results:
        reason = result.reason_code.value if result.reason_code else "-"
        fclass = result.failure_class.value if result.failure_class else "-"
        print(
            f"  - {result.scenario_id}: status={result.status.value} "
            f"failure_class={fclass} reason_code={reason}"
        )

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write("## Perf Matrix Summary\n\n")
            handle.write("| Scenario | Status | Failure Class | Reason Code |\n")
            handle.write("|---|---|---|---|\n")
            for result in results:
                reason = result.reason_code.value if result.reason_code else "-"
                fclass = result.failure_class.value if result.failure_class else "-"
                handle.write(
                    f"| {result.scenario_id} | {result.status.value} | {fclass} | {reason} |\n"
                )
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    repo_root_default = THIS_DIR.parents[2]

    parser = argparse.ArgumentParser(description="Run perf matrix against strict baselines")
    parser.add_argument("--suite", choices=("core", "encoder"), default="core")
    parser.add_argument("--repo-root", type=Path, default=repo_root_default)
    parser.add_argument("--build-dir", type=Path, default=Path("build-perf-gate"))
    parser.add_argument(
        "--prebuilt-tests-dir",
        type=Path,
        default=None,
        help="Run installed test executables from this directory without configuring or building.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--scenario-timeout-sec", type=int, default=int(os.getenv("SIMA_PERF_SCENARIO_TIMEOUT_SEC", "180")))
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--include-long",
        action="store_true",
        help="Include fixture-dependent long scenarios (weekly/device lanes only).",
    )
    parser.add_argument(
        "--failfast-only",
        action="store_true",
        help="Run preflight only (used by unit tests for fail-fast behavior).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    build_dir = (repo_root / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir
    profile_path = args.profile_dir or Path("tests/perf/baselines/v2") / (
        "modalix_encoder" if args.suite == "encoder" else "modalix_default")
    profile_dir = (repo_root / profile_path).resolve() if not profile_path.is_absolute() else profile_path
    default_results = args.build_dir / "perf_results"
    if args.suite == "encoder":
        default_results /= "encoder"
    results_dir_input = args.results_dir if args.results_dir is not None else default_results
    results_dir = (
        (repo_root / results_dir_input).resolve() if not results_dir_input.is_absolute() else results_dir_input
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    for stale in results_dir.glob("*.json"):
        stale.unlink(missing_ok=True)

    registered_scenarios = ENCODER_SCENARIOS if args.suite == "encoder" else SCENARIOS
    selected_scenarios = registered_scenarios if args.include_long or args.suite == "encoder" else STANDARD_SCENARIOS
    profile, baseline_map, preflight_failed = preflight_baselines(
        profile_dir, results_dir, registered_scenarios, selected_scenarios
    )

    if preflight_failed:
        results = [
            schema.load_perf_result(scenario_result_path(results_dir, s.scenario_id))
            for s in selected_scenarios
        ]
        print_summary(results)
        return 1

    assert profile is not None

    if args.failfast_only:
        results = []
        for spec in selected_scenarios:
            result = build_result(
                scenario_id=spec.scenario_id,
                modalix_profile_id=profile.modalix_profile_id,
                status=schema.ResultStatus.PASS,
                failure_class=None,
                reason_code=None,
                metrics=dict(schema.DEFAULT_EMPTY_METRICS),
                run_meta={"phase": "failfast-only"},
            )
            write_result(results_dir, result)
            results.append(result)
        print_summary(results)
        return 0

    if args.prebuilt_tests_dir is not None:
        executable_dir = (
            (repo_root / args.prebuilt_tests_dir).resolve()
            if not args.prebuilt_tests_dir.is_absolute()
            else args.prebuilt_tests_dir
        )
        preflight_dir = executable_dir
    else:
        build_targets = [
            "unit_modalix_contract_preflight_test",
            *dict.fromkeys(spec.target for spec in selected_scenarios),
        ]
        ok, build_error = configure_and_build(repo_root, build_dir, build_targets)
        if not ok:
            results: list[schema.PerfResult] = []
            for spec in selected_scenarios:
                result = build_result(
                    scenario_id=spec.scenario_id,
                    modalix_profile_id=profile.modalix_profile_id,
                    status=schema.ResultStatus.FAIL,
                    failure_class=schema.FailureClass.ENV_BROKEN,
                    reason_code=schema.ReasonCode.ENV_RUNTIME_CRASH,
                    metrics=dict(schema.DEFAULT_EMPTY_METRICS),
                    run_meta={"phase": "build", "error": build_error},
                )
                write_result(results_dir, result)
                results.append(result)
            print_summary(results)
            return 1
        executable_dir = build_dir / "tests"
        preflight_dir = build_dir

    preflight_ok, preflight_error = run_modalix_preflight(repo_root, preflight_dir)
    if not preflight_ok:
        preflight_reason = schema.classify_env_failure(
            exit_code=1,
            combined_output=preflight_error,
            timed_out=False,
        )
        results = []
        for spec in selected_scenarios:
            result = build_result(
                scenario_id=spec.scenario_id,
                modalix_profile_id=profile.modalix_profile_id,
                status=schema.ResultStatus.FAIL,
                failure_class=schema.FailureClass.ENV_BROKEN,
                reason_code=preflight_reason,
                metrics=dict(schema.DEFAULT_EMPTY_METRICS),
                run_meta={"phase": "modalix_preflight", "error": preflight_error},
            )
            write_result(results_dir, result)
            results.append(result)
        print_summary(results)
        return 1

    all_results: list[schema.PerfResult] = []
    for spec in selected_scenarios:
        baseline = baseline_map[spec.scenario_id]
        result = run_scenario(
            repo_root=repo_root,
            executable_dir=executable_dir,
            results_dir=results_dir,
            profile=profile,
            spec=spec,
            baseline=baseline,
            timeout_sec=args.scenario_timeout_sec,
            iterations_override=args.iterations,
        )
        write_result(results_dir, result)
        all_results.append(result)

    print_summary(all_results)

    hard_fail_classes = {
        schema.FailureClass.REGRESSION,
        schema.FailureClass.HARNESS_ERROR,
        schema.FailureClass.ENV_BROKEN,
    }
    for result in all_results:
        if result.failure_class in hard_fail_classes:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
