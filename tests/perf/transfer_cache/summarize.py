"""Summarize paired round medians; bootstrap whole pairs, not individual frames."""

import argparse
import json
import random
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", default="pr1274")
    parser.add_argument("--candidate", default="pr1322")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = json.loads(args.results.read_text())
    report = []
    for workload in sorted({r["workload"] for r in records}):
        rows = {
            v: {
                r["round"]: r
                for r in records
                if r["variant"] == v and r["workload"] == workload
            }
            for v in [args.baseline, args.candidate]
        }
        rounds = sorted(rows[args.baseline].keys() & rows[args.candidate].keys())
        if not rounds:
            continue
        medians = {
            v: [statistics.median(rows[v][i]["samples_ms"]) for i in rounds]
            for v in rows
        }
        ratios = [
            b / a for a, b in zip(medians[args.baseline], medians[args.candidate])
        ]
        rng = random.Random(1322)
        boot = sorted(
            statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(20000)
        )
        item = dict(
            workload=workload,
            paired_rounds=len(rounds),
            baseline_p50_ms=statistics.median(medians[args.baseline]),
            candidate_p50_ms=statistics.median(medians[args.candidate]),
            paired_change_percent=(statistics.median(ratios) - 1) * 100,
            paired_bootstrap_95_percent=[
                (boot[int(len(boot) * q)] - 1) * 100 for q in [0.025, 0.975]
            ],
            round_p50_ms=medians,
        )
        report.append(item)
        lo, hi = item["paired_bootstrap_95_percent"]
        print(
            f"{workload}: {item['baseline_p50_ms']:.3f} -> {item['candidate_p50_ms']:.3f} ms; "
            f"paired change {item['paired_change_percent']:+.2f}% [{lo:+.2f}%, {hi:+.2f}%]"
        )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
