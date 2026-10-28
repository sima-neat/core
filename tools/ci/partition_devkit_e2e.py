#!/usr/bin/env python3
"""Split the canonical CTest selection, then run the unchanged coverage gate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def partition(names, weights):
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError('E2E sharding requires at least two unique tests')
    bins, costs = [[], []], [0, 0]
    for name in sorted(names, key=lambda n: (-weights.get(n, 10), n)):
        i = min(range(2), key=lambda i: (costs[i], i))
        bins[i].append(name)
        costs[i] += weights.get(name, 10)
    return [sorted(b) for b in bins]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=(0, 1), required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command or command[0] != 'ctest' or '-R' in command or '--tests-regex' in command:
        raise ValueError('Expected canonical CTest argv without a name regex')
    inventory = json.loads(subprocess.check_output(command + ['--show-only=json-v1'], text=True))
    names = sorted(t['name'] for t in inventory['tests'])
    weights = json.loads((ROOT / 'tests/devkit-test-durations.json').read_text())['test_seconds']
    selected = partition(names, weights)[args.index]
    coverage = Path(os.environ['COVERAGE_DIR'])
    regex = '^(' + '|'.join(re.escape(n) for n in selected) + ')$'
    # Required zero-skip accounting still discovers and validates the exact
    # partial selection and executes CTest with its original timeout/jobs/labels.
    result = subprocess.run([sys.executable, str(ROOT / 'tools/ci/check_test_coverage.py'),
                    '--profile', os.environ.get('COVERAGE_PROFILE', 'strict'),
                    '--output-dir', str(coverage), '--', *command, '-R', regex])
    coverage.mkdir(parents=True, exist_ok=True)
    (coverage / 'e2e-partition.json').write_text(json.dumps(
        {'index': args.index, 'canonical': names, 'selected': selected}, indent=2) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
