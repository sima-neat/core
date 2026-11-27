"""Tests for the pre-install DevKit storage cleanup."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "cleanup_devkit_storage.sh"

FAKE_SUDO = """#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do
  case "$1" in
    -S) shift ;;
    -p) shift 2 ;;
    *) break ;;
  esac
done
cat >/dev/null
exec "$@"
"""

LEFTOVERS = (
    "sima-neat-install.Ab12Cd/bundle.deb",
    "sima-neat-platform-admission-k3j4h5/prepared/pkg/file",
    "sima-neat-apt-simulate-FxhqB6",
    "core-scheduler-admission-z4_c1_rg/package/file",
    "neat-apps-core-install.9Qljgt/core.deb",
    "neat-apps-latest.j3fRyu/tags",
    "neat-sysroot-deps-a1b2c3/file",
    "sima-cli-artifact-install.Q1w2E3/file",
    "install-neat-from-a-branch.Zx98Qw.sh",
    "pytest-of-sima/pytest-117/test_case0/file",
    "simaai-session/file",
)

KEPT = (
    "sima_neat_model_archive_tests/model.tar.gz",
    "sima-yolov8s-cache/model.bin",
    "systemd-private-0123-systemd-resolved.service-AbCdEf/tmp",
    "other-tool.txt",
)


class CleanupDevkitStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.storage_root = base / "root"
        self.tmp_root = self.storage_root / "tmp"
        self.tmp_root.mkdir(parents=True)
        self.coprocessing = self.storage_root / "data" / "simaai" / "coprocessing"
        (self.coprocessing / "job").mkdir(parents=True)
        (self.coprocessing / "buffer.bin").write_text("x")
        bin_dir = base / "bin"
        bin_dir.mkdir()
        sudo = bin_dir / "sudo"
        sudo.write_text(FAKE_SUDO)
        sudo.chmod(0o755)
        self.path = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
        for relative in LEFTOVERS + KEPT:
            entry = self.tmp_root / relative
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text("x")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_cleanup(self, threshold_mb: int) -> subprocess.CompletedProcess[str]:
        env = dict(
            os.environ,
            PATH=self.path,
            DEVKIT_STORAGE_ROOT=str(self.storage_root),
            TMP_FREE_SPACE_THRESHOLD_MB=str(threshold_mb),
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), "password"],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def top_level(self) -> set[str]:
        return {entry.name for entry in self.tmp_root.iterdir()}

    def test_removes_interrupted_run_scratch_while_tmp_has_room(self) -> None:
        self.run_cleanup(threshold_mb=1)

        self.assertEqual(self.top_level(), {Path(relative).parts[0] for relative in KEPT})

    def test_removes_sima_caches_only_when_tmp_is_low(self) -> None:
        result = self.run_cleanup(threshold_mb=10**9)

        self.assertNotIn("sima_neat_model_archive_tests", self.top_level())
        self.assertIn("sima-yolov8s-cache", self.top_level())
        self.assertIn(f"{self.tmp_root} free space is", result.stdout)

    def test_empties_coprocessing_directory(self) -> None:
        self.run_cleanup(threshold_mb=1)

        self.assertTrue(self.coprocessing.is_dir())
        self.assertEqual(list(self.coprocessing.iterdir()), [])

    def test_reports_free_space_on_root_and_tmp(self) -> None:
        result = self.run_cleanup(threshold_mb=1)

        self.assertIn(f"[storage-cleanup] {self.storage_root} free space:", result.stdout)
        self.assertIn(f"[storage-cleanup] {self.tmp_root} free space:", result.stdout)


if __name__ == "__main__":
    unittest.main()
