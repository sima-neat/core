#!/usr/bin/env python3
"""Validate per-scenario perf result files and print summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import perf_schema as schema


def parse_args() -> argparse.Namespace:
    repo_root = THIS_DIR.parents[3]
    parser = argparse.ArgumentParser(description="Validate perf result JSON schema")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "build-perf-gate" / "perf_results",
        help="Directory containing per-scenario result JSON files.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print one-line summary per scenario.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()

    if not results_dir.exists():
        print(f"[validate_perf_result] FAIL: results dir missing: {results_dir}")
        return 1

    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        print(f"[validate_perf_result] FAIL: no result json files in {results_dir}")
        return 1

    parsed: list[schema.PerfResult] = []
    for path in result_files:
        try:
            parsed.append(schema.load_perf_result(path))
        except schema.SchemaError as exc:
            print(f"[validate_perf_result] FAIL: {exc}")
            return 1

    by_scenario: dict[str, schema.PerfResult] = {}
    for result in parsed:
        if result.scenario_id in by_scenario:
            print(
                "[validate_perf_result] FAIL: duplicate scenario result: "
                f"{result.scenario_id}"
            )
            return 1
        by_scenario[result.scenario_id] = result

    expected = set(schema.REQUIRED_SCENARIO_IDS)
    present = set(by_scenario.keys())
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing:
        print(
            "[validate_perf_result] FAIL: missing required scenario results: "
            + ", ".join(missing)
        )
        return 1
    if extra:
        print(
            "[validate_perf_result] FAIL: unknown scenario results present: "
            + ", ".join(extra)
        )
        return 1

    if args.summary:
        print("[validate_perf_result] summary:")
        for scenario_id in sorted(expected):
            result = by_scenario[scenario_id]
            fclass = result.failure_class.value if result.failure_class else "-"
            reason = result.reason_code.value if result.reason_code else "-"
            print(
                f"  - {result.scenario_id}: status={result.status.value} "
                f"failure_class={fclass} reason_code={reason}"
            )

    print(f"[validate_perf_result] PASS: validated {len(parsed)} files in {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
