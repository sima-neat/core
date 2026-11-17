#!/usr/bin/env python3
"""Reject missing/duplicate exact-ID lane selections without executing tests."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys


def expected_ids(regex: str) -> list[str]:
    match = re.fullmatch(r'\^\(([A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*)\)\$', regex)
    if not match:
        raise ValueError('Exact-ID gate requires ^(literal_test_id|...)$')
    names = match[1].split('|')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate requested test IDs')
    return names


def validate_selection(regex: str, inventory: dict) -> list[str]:
    expected = expected_ids(regex)
    tests = inventory.get('tests', [])
    actual = [test.get('name') for test in tests]
    if any(not isinstance(name, str) or not name for name in actual):
        raise ValueError('Invalid selected test ID')
    if Counter(actual) != Counter(expected):
        raise ValueError(f'Exact test IDs differ: missing={list((Counter(expected)-Counter(actual)).elements())}, '
                         f'unexpected={list((Counter(actual)-Counter(expected)).elements())}')
    for test in tests:
        properties = {prop['name']: prop['value'] for prop in test.get('properties', [])}
        if properties.get('DISABLED', False):
            raise ValueError(f"Disabled requested test: {test['name']}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--regex', required=True)
    args = parser.parse_args()
    try:
        expected_ids(args.regex)
        # Check only explicitly requested IDs. The subsequent, unchanged CTest
        # execution still owns automatic setup/cleanup fixture dependencies.
        result = subprocess.run(
            ['ctest', '--test-dir', str(args.test_dir), '--show-only=json-v1',
             '--fixture-exclude-any', '.*', '-R', args.regex],
            check=True, text=True, capture_output=True)
        names = validate_selection(args.regex, json.loads(result.stdout))
        print(json.dumps({'status': 'EXACT_SELECTION_VALID', 'selected_ids': names,
                          'native_execution': False}))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'[exact-ctest-selection] FAIL: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
