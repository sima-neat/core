#!/usr/bin/env python3
"""Run the PCIe package's CPU coverage without provisioning or opening a card."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


CPU_TESTS = (
    'test_api_surface.py', 'test_genai_tutorials.py', 'test_genai_values.py',
    'test_metadata.py', 'test_wheel_metadata.py',
    'test_hardware_smoke.py::test_async_test_runner_cancels_blocked_peer',
    'test_hardware_smoke.py::test_metadata_yolov8',
)
MODEL_VARIABLES = (
    'SIMAPCIE_YOLOV8_MODEL', 'SIMAPCIE_MLA_ONLY_MODEL', 'SIMAPCIE_MLA_ONLY_BF16_MODEL',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_models(environment):
    for name in MODEL_VARIABLES:
        path = Path(environment.get(name, ''))
        require(path.is_file() and path.stat().st_size > 0,
                f'Missing required CPU metadata model: {name}={path}')


def resolve_models(workspace, work, environment):
    if not all(environment.get(name) for name in MODEL_VARIABLES):
        # The persistent CI account must have its existing Developer Portal
        # session; diagnose its absence with the bounded check, not a hidden
        # interactive login inside modelzoo. Explicit offline fixtures need none.
        subprocess.run(['bash', str(workspace / 'scripts/ci/check_sima_cli_session.sh')],
                       env=environment, cwd=workspace, check=True)
        subprocess.run(['bash', str(workspace / 'scripts/ci/pciehost/resolve_assets.sh')],
                       env=environment, cwd=workspace, check=True)
        for line in (work / 'assets.env').read_text().splitlines():
            name, separator, value = line.partition('=')
            if separator and name in MODEL_VARIABLES:
                environment[name] = value
    validate_models(environment)


def verify_tutorials(workspace, extras):
    relative = Path('029_run_genai_over_pcie')
    for name in ('run_llm.py', 'run_vlm.py', 'run_whisper.py'):
        source = workspace / 'pcie_host/tutorials' / relative / name
        installed = extras / 'share/sima-pcie-host/tutorials' / relative / name
        require(installed.is_file() and installed.read_bytes() == source.read_bytes(),
                f'Installed CPU tutorial differs from selected source: {name}')


def collected_tests(output):
    nodes = [line.strip() for line in output.splitlines() if '.py::' in line and not line.startswith(' ')]
    require(nodes and len(nodes) == len(set(nodes)), 'CPU pytest collection is empty or duplicated')
    # Normalize only the address. Parameter IDs can contain paths, separators
    # and brackets which pytest preserves verbatim in the JUnit case name.
    identities = set()
    for node in nodes:
        address, bracket, parameters = node.partition('[')
        identities.add(address.replace('.py::', '::', 1).replace('/', '.').replace('::', '.')
                       + bracket + parameters)
    return identities


def verify_python_report(collected, report):
    cases = list(ET.parse(report).iter('testcase'))
    identities = [f"{case.get('classname')}.{case.get('name')}" for case in cases]
    require(len(identities) == len(set(identities)) and set(identities) == collected,
            'Executed CPU pytest inventory differs from collection')
    require(not any(case.find(tag) is not None for case in cases for tag in ('skipped', 'failure', 'error')),
            'CPU pytest coverage contains a skip, failure or error')


def verify_metadata_report(report):
    cases = list(ET.parse(report).iter('testcase'))
    require([case.get('name') for case in cases] == ['test_metadata'],
            'Executed CPU CTest inventory is not exactly test_metadata')
    require(not any(case.find(tag) is not None for case in cases for tag in ('skipped', 'failure', 'error')),
            'CPU CTest coverage contains a skip, failure or error')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--distribution', type=Path, required=True)
    parser.add_argument('--work-directory', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    work = args.work_directory.resolve()
    require(not work.exists(), f'CPU package tests require a fresh work directory: {work}')
    archives = list(args.distribution.resolve().glob('sima-pcie-host-*-Linux-*-extras.tar.gz'))
    require(len(archives) == 1, 'Expected exactly one selected PCIe extras archive')
    # The existing smoke step has installed this distribution's wheel and
    # supplies the relocated Scheduler runtime before this interpreter starts.
    package = importlib.import_module('pyneatpcie')
    extension = importlib.import_module('pyneatpcie._pyneatpcie_core')
    prefix = Path(sys.prefix).resolve()
    require(sys.prefix != sys.base_prefix and all(
        Path(module.__file__).resolve().is_relative_to(prefix) for module in (package, extension)),
        'CPU tests must import the installed wheel from the selected smoke-test venv')
    for name in ('pytest', 'numpy', 'cv2'):
        importlib.import_module(name)
    extras = work / 'extras'
    extras.mkdir(parents=True)
    subprocess.run(['tar', '-xzf', str(archives[0]), '-C', str(extras)], check=True)
    verify_tutorials(workspace, extras)

    environment = dict(os.environ)
    environment.update(GITHUB_WORKSPACE=str(workspace), SIMAPCIE_WORK_DIR=str(work),
                       SIMAPCIE_EXTRAS_DIR=str(extras),
                       SIMAPCIE_TEST_IMAGE=str(workspace / 'tests/images/people.jpg'),
                       SIMAPCIE_BOXDECODE_IMAGE=str(workspace / 'tests/images/people.jpg'))
    resolve_models(workspace, work, environment)
    # A source test's relative imports need the source layout. Verify its
    # installed tutorial dependencies above instead of silently testing a stale
    # or incomplete extras package. Select only the two CPU cases from the
    # mixed hardware-smoke module, never that whole module or hardware suites.
    selected = [str(workspace / 'pcie_host/python/tests' / name) for name in CPU_TESTS]
    pytest = [sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib',
              '--rootdir', str(workspace), *selected]
    collection = subprocess.check_output([*pytest, '--collect-only'], cwd=workspace,
                                         env=environment, text=True)
    print(collection, end='')
    collected = collected_tests(collection)
    report = work / 'python-cpu.xml'
    subprocess.run([*pytest, '--junitxml', str(report)], cwd=workspace,
                   env=environment, check=True)
    verify_python_report(collected, report)

    test_dir = extras / 'lib/sima-pcie-host/tests'
    binary = test_dir / 'test_metadata'
    identity = subprocess.check_output(['file', '-L', str(binary)], text=True)
    print(identity, end='')
    machine = {'x86_64': 'x86-64', 'aarch64': 'ARM aarch64'}.get(platform.machine())
    require(machine and 'ELF ' in identity and machine in identity,
            'Installed metadata executable does not match this native host')
    ctest = ['ctest', '--test-dir', str(test_dir), '-R', '^test_metadata$', '--no-tests=error']
    inventory = json.loads(subprocess.check_output([*ctest, '--show-only=json-v1'], text=True))
    require([test['name'] for test in inventory['tests']] == ['test_metadata'],
            'Installed CPU CTest selection is not exactly test_metadata')
    command = inventory['tests'][0]['command']
    require(len(command) == 1 and (test_dir / command[0]).resolve() == binary.resolve(),
            'Installed metadata CTest does not execute the selected extras binary')
    metadata_report = work / 'metadata-cpu.xml'
    subprocess.run([*ctest, '--output-on-failure', '--output-junit', str(metadata_report)],
                   env=environment, check=True)
    verify_metadata_report(metadata_report)
    print(f'PCIe installed-package CPU coverage: {len(collected)} Python cases + test_metadata PASS')


if __name__ == '__main__':
    main()
