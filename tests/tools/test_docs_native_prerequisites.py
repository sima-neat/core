"""The native header probe is compile-only and precedes the real examples."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def invoke(tmp_path, environment, *, fail=False):
    commands = tmp_path / "commands.jsonl"
    cmake = tmp_path / "cmake"
    cmake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['COMMAND_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('FAIL_PROBE') == '1' and '--build' in sys.argv "
        "and any('native-prerequisites' in a for a in sys.argv):\n"
        "    sys.exit(2)\n"
    )
    cmake.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               COMMAND_LOG=str(commands), FAIL_PROBE="1" if fail else "0")
    result = subprocess.run(
        ["bash", str(ROOT / "tests/docs/build_docs_examples.sh"),
         "--environment", environment, "--build-root", str(tmp_path / "build"),
         "--sdk-sysroot", str(tmp_path)],
        env=env, text=True, capture_output=True, check=False,
    )
    return result, [json.loads(line) for line in commands.read_text().splitlines()]


def test_native_probe_precedes_both_examples_and_forces_recompile(tmp_path):
    result, calls = invoke(tmp_path, "devkit")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 6
    assert calls[0][0] == "-S" and calls[0][1].endswith("native-prerequisites")
    assert calls[1][0] == "--build" and "--clean-first" in calls[1]
    assert calls[2][1].endswith("hello-neat")
    assert calls[4][1].endswith("run-an-app")
    source = ROOT / "tests/docs/native-prerequisites/CMakeLists.txt"
    assert "OBJECT headers.cpp" in source.read_text()
    assert "add_executable" not in source.read_text()


def test_native_header_failure_blocks_examples_with_preparation_diagnostic(tmp_path):
    result, calls = invoke(tmp_path, "devkit", fail=True)
    assert result.returncode != 0
    assert "PREPARATION_BLOCKED" in result.stderr
    assert len(calls) == 2


def test_sdk_build_does_not_run_native_probe(tmp_path):
    result, calls = invoke(tmp_path, "sdk")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 4
    assert all("native-prerequisites" not in str(call) for call in calls)
