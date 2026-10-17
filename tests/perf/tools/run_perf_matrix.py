#!/usr/bin/env python3
"""Perf matrix orchestrator: strict schema checks + scenario comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import os
import platform
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


SKIP_RETURN_CODE = 77


def hardware_runner():
    # Source and installed extras have different roots, but both ship the same
    # reviewed supervisor module. Never resolve it from arbitrary PYTHONPATH.
    for root in (THIS_DIR.parents[2] / "tools", THIS_DIR.parents[1] / "tools"):
        path = root / "hardware_test_supervisor.py"
        if path.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location("neat_hardware_supervisor", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.run_owned
    raise RuntimeError("Installed hardware-test supervisor module is missing")


STANDARD_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("runtime_session_sync_rgb", "perf_runtime_graph_sync_rgb_test"),
    ScenarioSpec("runtime_session_async_rgb", "perf_runtime_graph_async_rgb_test"),
    ScenarioSpec("runtime_graph_fanout", "perf_runtime_graph_fanout_test"),
    ScenarioSpec("runtime_graph_join_bundle", "perf_runtime_graph_join_bundle_test"),
    ScenarioSpec("runtime_codec_mjpeg_decode", "perf_runtime_codec_mjpeg_decode_test"),
    ScenarioSpec("runtime_codec_h264_decode", "perf_runtime_codec_h264_decode_test"),
    ScenarioSpec("runtime_codec_h265_decode", "perf_runtime_codec_h265_decode_test"),
    ScenarioSpec("runtime_model_archive_load", "perf_runtime_model_archive_load_test"),
    ScenarioSpec("yolox_seg_pose_payload_decode_representative",
                 "perf_yolox_seg_pose_payload_representative_test"),
    ScenarioSpec("yolox_seg_pose_payload_decode_heavy",
                 "perf_yolox_seg_pose_payload_heavy_test"),
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
    # Keep complete scenario output, not just the JSON result's diagnostic tails.
    # Unique directories preserve earlier attempts when the results directory is reused.
    log_root = results_dir / "scenario-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tempfile.mkdtemp(prefix=f"{spec.scenario_id}-", dir=log_root)).resolve()
    capture: dict[str, Any] = {}
    trace_environment = {
        key: os.environ[key]
        for key in ("SIMA_NEAT_METRICS_RETAIN_TRACE", "SIMA_NEAT_METRICS_TRACE_DIR")
        if key in os.environ
    }
    diagnostic = (trace_environment.get("SIMA_NEAT_METRICS_RETAIN_TRACE") in ("1", "true")
                  or bool(trace_environment.get("SIMA_NEAT_METRICS_TRACE_DIR"))
                  or iterations_override is not None)
    invocation = {
        "command": [str(executable_dir / spec.target)],
        "cwd": str(repo_root),
        "iterations": iterations_override if iterations_override is not None else baseline.iterations,
        "timeout_sec": timeout_sec,
        "trace_environment": trace_environment,
        "purpose": "diagnostic" if diagnostic else "baseline_comparison",
    }
    # Deliberately do not dump the environment: it can contain authentication tokens.
    (log_dir / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n", encoding="utf-8")
    result = _run_scenario(
        repo_root=repo_root, executable_dir=executable_dir, results_dir=results_dir,
        profile=profile, spec=spec, baseline=baseline, timeout_sec=timeout_sec,
        iterations_override=iterations_override, capture=capture,
    )
    for stream in ("stdout", "stderr"):
        output = capture.get(stream) or ""
        if isinstance(output, bytes):  # TimeoutExpired may carry bytes despite text=True.
            (log_dir / f"{stream}.log").write_bytes(output)
        else:
            (log_dir / f"{stream}.log").write_text(output, encoding="utf-8")
    capture.pop("stdout", None)
    capture.pop("stderr", None)
    (log_dir / "outcome.json").write_text(json.dumps({
        **capture, "status": result.status.value,
        "reason_code": result.reason_code.value if result.reason_code else None,
    }, indent=2) + "\n", encoding="utf-8")
    result.run_meta["artifacts"] = {
        "directory": str(log_dir),
        "stdout": str(log_dir / "stdout.log"),
        "stderr": str(log_dir / "stderr.log"),
        "invocation": str(log_dir / "invocation.json"),
        "outcome": str(log_dir / "outcome.json"),
    }
    result.run_meta["execution_purpose"] = invocation["purpose"]
    # Actual trace identity remains in measure_report.plugin_latency.trace_dir.
    # A requested directory is not proof that tracing succeeded or was retained.
    return result


def _run_scenario(
    *,
    repo_root: Path,
    executable_dir: Path,
    results_dir: Path,
    profile: schema.PerfProfile,
    spec: ScenarioSpec,
    baseline: schema.ScenarioBaseline,
    timeout_sec: int,
    iterations_override: int | None,
    capture: dict[str, Any],
) -> schema.PerfResult:
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
        if os.environ.get("SIMA_HARDWARE_SUPERVISOR_SOCKET"):
            owned = hardware_runner()(command=[str(exe_path)], cwd=repo_root, env=env,
                                      deadline_sec=timeout_sec)
            capture.update(stdout=owned.get("stdout", ""), stderr=owned.get("stderr", ""),
                           returncode=owned.get("returncode"), supervisor_status=owned["status"],
                           retirement_proven=owned.get("retirement_proven"))
            if owned["status"] != "completed" or owned.get("retirement_proven") is not True:
                timed_out = owned["status"] == "timeout"
                return build_result(
                    scenario_id=spec.scenario_id, modalix_profile_id=profile.modalix_profile_id,
                    status=schema.ResultStatus.FAIL, failure_class=schema.FailureClass.ENV_BROKEN,
                    reason_code=schema.ReasonCode.ENV_TIMEOUT if timed_out else schema.ReasonCode.ENV_RUNTIME_CRASH,
                    metrics=dict(schema.DEFAULT_EMPTY_METRICS),
                    run_meta={"phase": "run", "executable": str(exe_path), "timeout_sec": timeout_sec,
                              "hardware_fail_stop": True, "supervisor": owned})
            proc = subprocess.CompletedProcess([str(exe_path)], owned["returncode"], owned["stdout"], owned["stderr"])
        elif platform.machine() not in ("aarch64", "arm64") and os.environ.get("SIMA_PERF_HOST_FIXTURES_ONLY") == "1":
            proc = run_cmd([str(exe_path)], cwd=repo_root, timeout_sec=timeout_sec, env=env)
        else:
            raise RuntimeError("Hardware matrix requires an admitted persistent supervisor")
    except (RuntimeError, OSError, ValueError) as exc:
        return build_result(
            scenario_id=spec.scenario_id, modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL, failure_class=schema.FailureClass.ENV_BROKEN,
            reason_code=schema.ReasonCode.ENV_RUNTIME_CRASH, metrics=dict(schema.DEFAULT_EMPTY_METRICS),
            run_meta={"phase": "run", "error": str(exc), "hardware_fail_stop": True})
    except subprocess.TimeoutExpired as exc:
        capture.update(stdout=exc.stdout, stderr=exc.stderr, timed_out=True)
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.ENV_BROKEN,
            reason_code=schema.ReasonCode.ENV_TIMEOUT,
            metrics=dict(schema.DEFAULT_EMPTY_METRICS),
            run_meta={"phase": "run", "executable": str(exe_path), "timeout_sec": timeout_sec},
        )

    capture.update(stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
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
        default=Path("tests/perf/baselines/v2/modalix_default"),
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
    profile_dir = (repo_root / args.profile_dir).resolve() if not args.profile_dir.is_absolute() else args.profile_dir
    results_dir_input = args.results_dir if args.results_dir is not None else (args.build_dir / "perf_results")
    results_dir = (
        (repo_root / results_dir_input).resolve() if not results_dir_input.is_absolute() else results_dir_input
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    for stale in results_dir.glob("*.json"):
        stale.unlink(missing_ok=True)

    selected_scenarios = SCENARIOS if args.include_long else STANDARD_SCENARIOS
    profile, baseline_map, preflight_failed = preflight_baselines(
        profile_dir, results_dir, SCENARIOS, selected_scenarios
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
            *[spec.target for spec in selected_scenarios],
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
        if result.run_meta.get("hardware_fail_stop"):
            # These scenarios were NOT RUN, not accepted skips or fabricated
            # metric results. The matrix cannot qualify after unknown retirement.
            remaining = [item.scenario_id for item in selected_scenarios[len(all_results):]]
            (results_dir / "hardware-fail-stop-summary.json").write_text(json.dumps({
                "status": "FAIL_STOP", "failed_scenario": spec.scenario_id,
                "not_run_scenarios": remaining, "owner_retirement_unknown": True,
            }, sort_keys=True) + "\n")
            print_summary(all_results)
            return 1

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
