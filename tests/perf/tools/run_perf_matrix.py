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
import subprocess
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import perf_schema as schema


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    target: str


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("runtime_session_sync_rgb", "perf_runtime_graph_sync_rgb_test"),
    ScenarioSpec("runtime_session_async_rgb", "perf_runtime_graph_async_rgb_test"),
    ScenarioSpec("runtime_graph_fanout", "perf_runtime_graph_fanout_test"),
    ScenarioSpec("runtime_graph_join_bundle", "perf_runtime_graph_join_bundle_test"),
    ScenarioSpec("runtime_codec_mjpeg_decode", "perf_runtime_codec_mjpeg_decode_test"),
    ScenarioSpec("runtime_codec_h264_decode", "perf_runtime_codec_h264_decode_test"),
)


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
) -> tuple[schema.PerfProfile | None, dict[str, schema.ScenarioBaseline], bool]:
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        profile, baseline_map = schema.validate_baseline_directory(profile_dir)
    except schema.SchemaError as exc:
        for spec in scenarios:
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
        for spec in scenarios:
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
        for spec in scenarios:
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


def run_modalix_preflight(repo_root: Path, build_dir: Path) -> tuple[bool, str]:
    proc = run_cmd(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
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
    build_dir: Path,
    results_dir: Path,
    profile: schema.PerfProfile,
    spec: ScenarioSpec,
    baseline: schema.ScenarioBaseline,
    timeout_sec: int,
    iterations_override: int | None,
) -> schema.PerfResult:
    exe_path = build_dir / "tests" / spec.target
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
        proc = run_cmd([str(exe_path)], cwd=repo_root, timeout_sec=timeout_sec, env=env)
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
    if proc.returncode != 0:
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
        power = schema.parse_optional_power_payload(
            payload_raw, context=f"payload:{spec.scenario_id}"
        )
        measure_report = schema.parse_optional_measure_report_payload(
            payload_raw, context=f"payload:{spec.scenario_id}"
        )
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

    run_meta = {"phase": "compare", "executable": str(exe_path), "iterations": scenario_iters}
    if power is not None:
        run_meta["power"] = power
    if measure_report is not None:
        run_meta["measure_report"] = measure_report

    regressions = schema.compare_metrics(metrics, baseline)
    if regressions:
        return build_result(
            scenario_id=spec.scenario_id,
            modalix_profile_id=profile.modalix_profile_id,
            status=schema.ResultStatus.FAIL,
            failure_class=schema.FailureClass.REGRESSION,
            reason_code=regressions[0],
            metrics=metrics,
            run_meta={**run_meta, "regression_reasons": [reason.value for reason in regressions]},
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
    repo_root_default = THIS_DIR.parents[3]

    parser = argparse.ArgumentParser(description="Run perf matrix against strict baselines")
    parser.add_argument("--repo-root", type=Path, default=repo_root_default)
    parser.add_argument("--build-dir", type=Path, default=Path("build-perf-gate"))
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("tests/perf/baselines/v2/modalix_default"),
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--scenario-timeout-sec", type=int, default=int(os.getenv("SIMA_PERF_SCENARIO_TIMEOUT_SEC", "180")))
    parser.add_argument("--iterations", type=int, default=None)
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

    profile, baseline_map, preflight_failed = preflight_baselines(profile_dir, results_dir, SCENARIOS)

    if preflight_failed:
        results = [schema.load_perf_result(scenario_result_path(results_dir, s.scenario_id)) for s in SCENARIOS]
        print_summary(results)
        return 1

    assert profile is not None

    if args.failfast_only:
        results = []
        for spec in SCENARIOS:
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

    build_targets = ["unit_modalix_contract_preflight_test", *[spec.target for spec in SCENARIOS]]
    ok, build_error = configure_and_build(repo_root, build_dir, build_targets)
    if not ok:
        results: list[schema.PerfResult] = []
        for spec in SCENARIOS:
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

    preflight_ok, preflight_error = run_modalix_preflight(repo_root, build_dir)
    if not preflight_ok:
        preflight_reason = schema.classify_env_failure(
            exit_code=1,
            combined_output=preflight_error,
            timed_out=False,
        )
        results = []
        for spec in SCENARIOS:
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
    for spec in SCENARIOS:
        baseline = baseline_map[spec.scenario_id]
        result = run_scenario(
            repo_root=repo_root,
            build_dir=build_dir,
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
