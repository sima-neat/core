"""Every allocated shard prepares dependencies independently of persistent state."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dependencies', ROOT / 'tools/ci/install_test_dependencies.py')
dependencies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dependencies)


def test_installs_complete_declaration_with_selected_interpreter(tmp_path):
    project = tmp_path / 'pyproject.toml'
    project.write_text('[project]\ndependencies = ["runtime-fixture>=1"]\n'
                       '[project.optional-dependencies]\ndev = ["future-test-fixture>=2"]\n')
    with patch.object(dependencies.subprocess, 'run') as run:
        dependencies.install(project)
    run.assert_called_once_with([sys.executable, '-m', 'pip', 'install',
                                 'runtime-fixture>=1', 'future-test-fixture>=2'], check=True)


def test_pip_failure_blocks_preparation():
    with patch.object(dependencies.subprocess, 'run',
                      side_effect=subprocess.CalledProcessError(1, 'pip')):
        with pytest.raises(subprocess.CalledProcessError):
            dependencies.install(ROOT / 'pyproject.toml')


def test_every_shard_prepares_before_tests_without_lane_condition():
    workflow = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())
    steps = workflow['jobs']['test-devkit-shard']['steps']
    index = next(i for i, step in enumerate(steps)
                 if step.get('name') == 'Install test dependencies on every allocated DevKit')
    assert 'if' not in steps[index]
    assert '"${PYTHON_BIN}" tools/ci/install_test_dependencies.py' in steps[index]['run']
    assert index < next(i for i, step in enumerate(steps)
                        if step.get('name') == 'Smoke-test installed PyNeat wheel')
