#!/usr/bin/env python3
"""Strict typed schema + comparator helpers for perf baselines and results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
import json


class SchemaError(ValueError):
    """Raised when a baseline/result file violates schema."""


class FailureClass(str, Enum):
    REGRESSION = "REGRESSION"
    HARNESS_ERROR = "HARNESS_ERROR"
    ENV_BROKEN = "ENV_BROKEN"


class ReasonCode(str, Enum):
    REGRESSION_THROUGHPUT = "REGRESSION_THROUGHPUT"
    REGRESSION_P50 = "REGRESSION_P50"
    REGRESSION_P95 = "REGRESSION_P95"
    REGRESSION_STARTUP = "REGRESSION_STARTUP"
    REGRESSION_RSS = "REGRESSION_RSS"
    REGRESSION_DROPS = "REGRESSION_DROPS"
    HARNESS_SCHEMA_INVALID = "HARNESS_SCHEMA_INVALID"
    HARNESS_METRIC_MISSING = "HARNESS_METRIC_MISSING"
    HARNESS_BASELINE_MISSING = "HARNESS_BASELINE_MISSING"
    ENV_SIMA_CLI_FAIL = "ENV_SIMA_CLI_FAIL"
    ENV_MODEL_DOWNLOAD_FAIL = "ENV_MODEL_DOWNLOAD_FAIL"
    ENV_RUNTIME_CRASH = "ENV_RUNTIME_CRASH"
    ENV_TIMEOUT = "ENV_TIMEOUT"


class ResultStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


METRIC_KEYS = (
    "throughput",
    "p50",
    "p95",
    "startup",
    "rss_peak_kb",
    "input_drop_count",
    "output_drop_count",
)

DEFAULT_EMPTY_METRICS: dict[str, float] = {
    "throughput": 0.0,
    "p50": 0.0,
    "p95": 0.0,
    "startup": 0.0,
    "rss_peak_kb": 0.0,
    "input_drop_count": 0.0,
    "output_drop_count": 0.0,
}

REQUIRED_SCENARIO_IDS = (
    "runtime_session_sync_rgb",
    "runtime_session_async_rgb",
    "runtime_graph_fanout",
    "runtime_graph_join_bundle",
)


@dataclass(frozen=True)
class PerfProfile:
    modalix_profile_id: str
    board_class: str
    sdk_version: str
    compiler: str
    gstreamer_version: str
    runtime_plugin_bundle_hash: str


@dataclass(frozen=True)
class MetricsThresholds:
    throughput_min: float
    p50_max: float
    p95_max: float
    startup_max: float
    rss_peak_kb_max: float
    input_drop_count_max: float
    output_drop_count_max: float
    regression_tolerance_percent: float


@dataclass(frozen=True)
class ScenarioBaseline:
    scenario_id: str
    model_id: str
    pipeline_id: str
    run_mode: str
    iterations: int
    metrics_thresholds: MetricsThresholds


@dataclass(frozen=True)
class PerfResult:
    scenario_id: str
    modalix_profile_id: str
    status: ResultStatus
    failure_class: FailureClass | None
    reason_code: ReasonCode | None
    metrics: dict[str, float]
    run_meta: dict[str, Any]
    timestamp: str


def _raise(context: str, message: str) -> None:
    raise SchemaError(f"{context}: {message}")


def _expect_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise(context, "expected JSON object")
    return value


def _reject_unknown_keys(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data.keys()) - allowed)
    if unknown:
        _raise(context, f"unknown fields: {', '.join(unknown)}")


def _require_keys(data: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(data.keys()))
    if missing:
        _raise(context, f"missing required fields: {', '.join(missing)}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        _raise(context, "expected non-empty string")
    return value


def _as_int(value: Any, context: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _raise(context, "expected integer")
    if minimum is not None and value < minimum:
        _raise(context, f"expected integer >= {minimum}")
    return value


def _as_float(value: Any, context: str, minimum: float | None = None) -> float:
    if not _is_number(value):
        _raise(context, "expected numeric value")
    casted = float(value)
    if minimum is not None and casted < minimum:
        _raise(context, f"expected numeric value >= {minimum}")
    return casted


def _as_dict(value: Any, context: str) -> dict[str, Any]:
    mapping = _expect_mapping(value, context)
    return dict(mapping)


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchemaError(f"{path}: file not found") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: root must be a JSON object")
    return data


def parse_profile(data: Mapping[str, Any], context: str = "profile") -> PerfProfile:
    required = {
        "modalix_profile_id",
        "board_class",
        "sdk_version",
        "compiler",
        "gstreamer_version",
        "runtime_plugin_bundle_hash",
    }
    _require_keys(data, required, context)
    _reject_unknown_keys(data, required, context)

    return PerfProfile(
        modalix_profile_id=_as_str(data["modalix_profile_id"], f"{context}.modalix_profile_id"),
        board_class=_as_str(data["board_class"], f"{context}.board_class"),
        sdk_version=_as_str(data["sdk_version"], f"{context}.sdk_version"),
        compiler=_as_str(data["compiler"], f"{context}.compiler"),
        gstreamer_version=_as_str(data["gstreamer_version"], f"{context}.gstreamer_version"),
        runtime_plugin_bundle_hash=_as_str(
            data["runtime_plugin_bundle_hash"], f"{context}.runtime_plugin_bundle_hash"
        ),
    )


def parse_metrics_thresholds(
    data: Mapping[str, Any], context: str = "metrics_thresholds"
) -> MetricsThresholds:
    required = {
        "throughput_min",
        "p50_max",
        "p95_max",
        "startup_max",
        "rss_peak_kb_max",
        "input_drop_count_max",
        "output_drop_count_max",
        "regression_tolerance_percent",
    }
    _require_keys(data, required, context)
    _reject_unknown_keys(data, required, context)

    return MetricsThresholds(
        throughput_min=_as_float(data["throughput_min"], f"{context}.throughput_min", minimum=0.0),
        p50_max=_as_float(data["p50_max"], f"{context}.p50_max", minimum=0.0),
        p95_max=_as_float(data["p95_max"], f"{context}.p95_max", minimum=0.0),
        startup_max=_as_float(data["startup_max"], f"{context}.startup_max", minimum=0.0),
        rss_peak_kb_max=_as_float(data["rss_peak_kb_max"], f"{context}.rss_peak_kb_max", minimum=0.0),
        input_drop_count_max=_as_float(
            data["input_drop_count_max"], f"{context}.input_drop_count_max", minimum=0.0
        ),
        output_drop_count_max=_as_float(
            data["output_drop_count_max"], f"{context}.output_drop_count_max", minimum=0.0
        ),
        regression_tolerance_percent=_as_float(
            data["regression_tolerance_percent"],
            f"{context}.regression_tolerance_percent",
            minimum=0.0,
        ),
    )


def parse_scenario_baseline(
    data: Mapping[str, Any], context: str = "scenario_baseline"
) -> ScenarioBaseline:
    required = {
        "scenario_id",
        "model_id",
        "pipeline_id",
        "run_mode",
        "iterations",
        "metrics_thresholds",
    }
    _require_keys(data, required, context)
    _reject_unknown_keys(data, required, context)

    thresholds = parse_metrics_thresholds(
        _as_dict(data["metrics_thresholds"], f"{context}.metrics_thresholds"),
        context=f"{context}.metrics_thresholds",
    )

    return ScenarioBaseline(
        scenario_id=_as_str(data["scenario_id"], f"{context}.scenario_id"),
        model_id=_as_str(data["model_id"], f"{context}.model_id"),
        pipeline_id=_as_str(data["pipeline_id"], f"{context}.pipeline_id"),
        run_mode=_as_str(data["run_mode"], f"{context}.run_mode"),
        iterations=_as_int(data["iterations"], f"{context}.iterations", minimum=1),
        metrics_thresholds=thresholds,
    )


def parse_metrics(data: Mapping[str, Any], context: str = "metrics") -> dict[str, float]:
    required = set(METRIC_KEYS)
    _require_keys(data, required, context)
    _reject_unknown_keys(data, required, context)

    parsed: dict[str, float] = {}
    for key in METRIC_KEYS:
        parsed[key] = _as_float(data[key], f"{context}.{key}", minimum=0.0)
    return parsed


def parse_metrics_payload(data: Mapping[str, Any], context: str = "scenario_payload") -> dict[str, float]:
    """Parse direct scenario binary output payload.

    Expected shape is flat top-level fields:
      scenario_id, iterations, run_mode, <metric keys>
    Unknown fields are allowed for payload only, but all metric keys are required.
    """

    missing = [key for key in METRIC_KEYS if key not in data]
    if missing:
        _raise(context, f"missing metrics: {', '.join(sorted(missing))}")

    metrics_obj: dict[str, Any] = {}
    for key in METRIC_KEYS:
        metrics_obj[key] = data[key]
    return parse_metrics(metrics_obj, context=f"{context}.metrics")


def parse_optional_power_payload(data: Mapping[str, Any], context: str = "scenario_payload") -> dict[str, Any] | None:
    """Return optional power metadata from a direct scenario payload.

    Power is observability metadata, not a baseline-gated metric. Keep the shape
    intentionally light here: if present it must be a JSON object so result files
    remain predictable, but detailed field evolution is allowed inside run_meta.
    """

    if "power" not in data:
        return None
    return _as_dict(data["power"], f"{context}.power")


def parse_optional_measure_report_payload(
    data: Mapping[str, Any], context: str = "scenario_payload"
) -> dict[str, Any] | None:
    """Return optional MeasureReport metadata from a scenario payload."""

    if "measure_report" not in data:
        return None
    return _as_dict(data["measure_report"], f"{context}.measure_report")


def parse_perf_result(data: Mapping[str, Any], context: str = "perf_result") -> PerfResult:
    required = {
        "scenario_id",
        "modalix_profile_id",
        "status",
        "failure_class",
        "reason_code",
        "metrics",
        "run_meta",
        "timestamp",
    }
    _require_keys(data, required, context)
    _reject_unknown_keys(data, required, context)

    status_raw = _as_str(data["status"], f"{context}.status")
    try:
        status = ResultStatus(status_raw)
    except ValueError as exc:
        raise SchemaError(f"{context}.status: invalid enum value '{status_raw}'") from exc

    failure_class: FailureClass | None
    if data["failure_class"] is None:
        failure_class = None
    else:
        try:
            failure_class = FailureClass(_as_str(data["failure_class"], f"{context}.failure_class"))
        except ValueError as exc:
            raise SchemaError(
                f"{context}.failure_class: invalid enum value '{data['failure_class']}'"
            ) from exc

    reason_code: ReasonCode | None
    if data["reason_code"] is None:
        reason_code = None
    else:
        try:
            reason_code = ReasonCode(_as_str(data["reason_code"], f"{context}.reason_code"))
        except ValueError as exc:
            raise SchemaError(
                f"{context}.reason_code: invalid enum value '{data['reason_code']}'"
            ) from exc

    if status == ResultStatus.PASS and (failure_class is not None or reason_code is not None):
        _raise(context, "PASS result must have null failure_class and reason_code")
    if status == ResultStatus.FAIL and (failure_class is None or reason_code is None):
        _raise(context, "FAIL result must have non-null failure_class and reason_code")

    run_meta = _as_dict(data["run_meta"], f"{context}.run_meta")
    # strict unknown-field policy applies to top-level schema; run_meta is free-form metadata.

    metrics = parse_metrics(_as_dict(data["metrics"], f"{context}.metrics"), f"{context}.metrics")

    return PerfResult(
        scenario_id=_as_str(data["scenario_id"], f"{context}.scenario_id"),
        modalix_profile_id=_as_str(data["modalix_profile_id"], f"{context}.modalix_profile_id"),
        status=status,
        failure_class=failure_class,
        reason_code=reason_code,
        metrics=metrics,
        run_meta=run_meta,
        timestamp=_as_str(data["timestamp"], f"{context}.timestamp"),
    )


def load_profile(path: Path) -> PerfProfile:
    return parse_profile(load_json_file(path), context=str(path))


def load_scenario_baseline(path: Path) -> ScenarioBaseline:
    return parse_scenario_baseline(load_json_file(path), context=str(path))


def load_perf_result(path: Path) -> PerfResult:
    return parse_perf_result(load_json_file(path), context=str(path))


def validate_baseline_directory(profile_dir: Path) -> tuple[PerfProfile, dict[str, ScenarioBaseline]]:
    profile_path = profile_dir / "profile.json"
    profile = load_profile(profile_path)

    scenario_map: dict[str, ScenarioBaseline] = {}
    for path in sorted(profile_dir.glob("*.json")):
        if path.name == "profile.json":
            continue
        baseline = load_scenario_baseline(path)
        if path.stem != baseline.scenario_id:
            raise SchemaError(
                f"{path}: filename '{path.stem}' must match scenario_id '{baseline.scenario_id}'"
            )
        if baseline.scenario_id in scenario_map:
            raise SchemaError(f"{path}: duplicate scenario_id '{baseline.scenario_id}'")
        scenario_map[baseline.scenario_id] = baseline

    if not scenario_map:
        raise SchemaError(f"{profile_dir}: no scenario baseline files found")

    return profile, scenario_map


def _threshold_failures(metrics: Mapping[str, float], thresholds: MetricsThresholds) -> list[ReasonCode]:
    tol = thresholds.regression_tolerance_percent / 100.0

    failures: list[ReasonCode] = []

    throughput_floor = thresholds.throughput_min * (1.0 - tol)
    if metrics["throughput"] < throughput_floor:
        failures.append(ReasonCode.REGRESSION_THROUGHPUT)

    p50_ceiling = thresholds.p50_max * (1.0 + tol)
    if metrics["p50"] > p50_ceiling:
        failures.append(ReasonCode.REGRESSION_P50)

    p95_ceiling = thresholds.p95_max * (1.0 + tol)
    if metrics["p95"] > p95_ceiling:
        failures.append(ReasonCode.REGRESSION_P95)

    startup_ceiling = thresholds.startup_max * (1.0 + tol)
    if metrics["startup"] > startup_ceiling:
        failures.append(ReasonCode.REGRESSION_STARTUP)

    rss_ceiling = thresholds.rss_peak_kb_max * (1.0 + tol)
    if metrics["rss_peak_kb"] > rss_ceiling:
        failures.append(ReasonCode.REGRESSION_RSS)

    input_drop_ceiling = thresholds.input_drop_count_max * (1.0 + tol)
    output_drop_ceiling = thresholds.output_drop_count_max * (1.0 + tol)
    if metrics["input_drop_count"] > input_drop_ceiling or metrics["output_drop_count"] > output_drop_ceiling:
        failures.append(ReasonCode.REGRESSION_DROPS)

    return failures


def compare_metrics(metrics: Mapping[str, float], baseline: ScenarioBaseline) -> list[ReasonCode]:
    missing = sorted(set(METRIC_KEYS) - set(metrics.keys()))
    if missing:
        raise SchemaError(f"missing metrics: {', '.join(missing)}")

    parsed_metrics = parse_metrics(dict(metrics), context="compare.metrics")
    return _threshold_failures(parsed_metrics, baseline.metrics_thresholds)


def classify_env_failure(exit_code: int, combined_output: str, timed_out: bool = False) -> ReasonCode:
    if timed_out or exit_code == 124:
        return ReasonCode.ENV_TIMEOUT

    text = combined_output.lower()
    if "sima-cli" in text and ("fail" in text or "error" in text or "not found" in text):
        return ReasonCode.ENV_SIMA_CLI_FAIL
    if "model" in text and "download" in text and ("fail" in text or "error" in text):
        return ReasonCode.ENV_MODEL_DOWNLOAD_FAIL
    return ReasonCode.ENV_RUNTIME_CRASH


def result_to_json_dict(result: PerfResult) -> dict[str, Any]:
    return {
        "scenario_id": result.scenario_id,
        "modalix_profile_id": result.modalix_profile_id,
        "status": result.status.value,
        "failure_class": result.failure_class.value if result.failure_class else None,
        "reason_code": result.reason_code.value if result.reason_code else None,
        "metrics": result.metrics,
        "run_meta": result.run_meta,
        "timestamp": result.timestamp,
    }


def write_result(path: Path, result: PerfResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_to_json_dict(result), indent=2) + "\n", encoding="utf-8")
