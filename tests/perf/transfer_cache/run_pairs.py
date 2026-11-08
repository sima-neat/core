"""Run fresh-process, alternating-order comparisons on a DevKit.

Run through dk/devkit-run. Library directories select Core; all other runtime
providers and the Python binding stay fixed. No global packages are changed.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def sensors():
    paths = [
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "/sys/class/thermal/thermal_zone0/temp",
    ]
    return {p: Path(p).read_text().strip() for p in paths if Path(p).exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="label=/absolute/Core/library/directory",
    )
    parser.add_argument("--runtime-library-path", required=True)
    parser.add_argument("--transfer-bench", type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument(
        "--workload", action="append", help="run only these named workloads"
    )
    parser.add_argument(
        "--cpu", type=int, help="pin this benchmark and its children to one CPU"
    )
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--pressure-iterations", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    variants = {}
    for item in args.variant:
        label, path = item.split("=", 1)
        variants[label] = str(Path(path).resolve(strict=True))
    if len(variants) < 2:
        parser.error("provide at least two distinct variant labels")
    if args.rounds < 1 or args.iterations < 1:
        parser.error("rounds and iterations must be positive")
    if args.model_only and not args.model:
        parser.error("--model-only requires --model")
    if not args.model_only and not args.transfer_bench:
        parser.error("--transfer-bench is required for transfer workloads")
    workloads = []
    if not args.model_only:
        workloads += [
            ("cpp_46MiB", "cpp", 48234496, 0, 1, None),
            ("cpp_centerpoint_bytes", "cpp", 46080000, 0, 1, None),
            ("cpp_centerpoint_held2", "cpp", 46080000, 2, 1, None),
            ("numpy_centerpoint_bytes", "numpy_transfer", 46080000, 0, 1, None),
        ]
    for model in args.model:
        model = model.resolve(strict=True)
        for mode in ["model_numpy", "model_device"]:
            workloads.append((model.name + ":" + mode, mode, 0, 0, 1, model))
    if not args.model_only:
        workloads.append(("over_budget_six_sizes", "cpp", 46080000, 0, 6, None))
    if args.workload:
        wanted = set(args.workload)
        workloads = [item for item in workloads if item[0] in wanted]
        if len(workloads) != len(wanted):
            parser.error("unknown or duplicate workload selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    linkage = {}
    labels = list(variants)
    for round_id in range(args.rounds):
        offset = round_id % len(labels)
        order = labels[offset:] + labels[:offset]
        if len(labels) == 2:
            order = labels if round_id % 2 == 0 else labels[::-1]
        elif round_id % 2:
            order.reverse()
        tasks = list(workloads)
        random.Random(4127 + round_id).shuffle(tasks)
        for variant in order:
            env = os.environ.copy()
            env.update(
                LD_LIBRARY_PATH=variants[variant] + ":" + args.runtime_library_path,
                GST_PLUGIN_SYSTEM_PATH_1_0="",
                GST_PLUGIN_PATH="/usr/lib/aarch64-linux-gnu/neat/gst-plugins",
            )
            if args.transfer_bench and variant not in linkage:
                linkage[variant] = subprocess.run(
                    ["ldd", str(args.transfer_bench)],
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout
                args.output.with_suffix(".linkage.json").write_text(
                    json.dumps(linkage, indent=2) + "\n"
                )
            for workload, mode, size, held, cycles, model in tasks:
                count = args.pressure_iterations if cycles > 1 else args.iterations
                if mode == "cpp":
                    cmd = [
                        str(args.transfer_bench),
                        str(size),
                        str(held),
                        str(count),
                        str(cycles),
                    ]
                else:
                    cmd = [
                        args.python_executable,
                        str(Path(__file__).with_name("python_bench.py")),
                        "--mode",
                        mode,
                        "--bytes",
                        str(size),
                        "--iterations",
                        str(count),
                        "--warmup",
                        "10",
                    ]
                    if model:
                        cmd += ["--model", str(model)]
                before = sensors()
                result = subprocess.run(
                    cmd, env=env, text=True, capture_output=True, timeout=120
                )
                if result.returncode:
                    log = args.output.with_name(f"failed-{variant}-{mode}.log")
                    log.write_text(result.stdout + "\n" + result.stderr)
                    raise RuntimeError(f"{variant} {workload} failed; see {log}")
                payload = next(
                    (
                        line.removeprefix("BENCH_JSON=")
                        for line in result.stdout.splitlines()
                        if line.startswith("BENCH_JSON=")
                    ),
                    result.stdout,
                )
                data = json.loads(payload)
                data.update(
                    round=round_id,
                    variant=variant,
                    workload=workload,
                    order=order,
                    wall_time=time.time(),
                    affinity=sorted(os.sched_getaffinity(0)),
                    sensors_before=before,
                    sensors_after=sensors(),
                )
                records.append(data)
                temporary = args.output.with_suffix(".partial")
                temporary.write_text(json.dumps(records, indent=2) + "\n")
                temporary.replace(args.output)
        print(
            f"Completed matched round {round_id + 1}/{args.rounds} ({len(records)} measurements)",
            flush=True,
        )


if __name__ == "__main__":
    main()
