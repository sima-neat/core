#!/usr/bin/env python3
"""Contracts for the native Windows Core PCIe-host producer."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "pciehost" / "build_windows.py"
WORKFLOW = ROOT / ".github" / "workflows" / "pcie-host-windows.yml"
SPEC = importlib.util.spec_from_file_location("build_windows", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class WindowsCiContracts(unittest.TestCase):
  def scheduler_candidate(self, root: Path) -> Path:
    msi = root / "scheduler.msi"
    msi.write_bytes(b"scheduler-msi")
    receipt = root / "host-candidate.json"
    receipt.write_text("{}\n", encoding="utf-8")
    resources = {path.name: MODULE.sha256(path) for path in (msi, receipt)}
    (root / "metadata.json").write_text(json.dumps({
        "artifact": {"architecture": "amd64", "profile": "pcie-host-windows"},
        "commit": "a" * 40,
        "installation": {"msi": msi.name, "type": "windows-msi"},
        "repository": "sima-neat/scheduler",
        "resources": sorted(resources),
        "resources-checksum": resources,
        "version": "a" * 12,
    }), encoding="utf-8")
    return msi

  def test_scheduler_candidate_accepts_exact_inventory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      expected = self.scheduler_candidate(root)
      metadata, actual = MODULE.verify_scheduler_candidate(root)
      self.assertEqual(actual, expected)
      self.assertEqual(metadata["commit"], "a" * 40)

  def test_scheduler_candidate_rejects_changed_msi(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      msi = self.scheduler_candidate(root)
      msi.write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "checksum differs"):
        MODULE.verify_scheduler_candidate(root)

  def test_scheduler_candidate_rejects_non_mapping_checksums(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.scheduler_candidate(root)
      metadata_path = root / "metadata.json"
      metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      metadata["resources-checksum"] = metadata["resources"]
      metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
        MODULE.verify_scheduler_candidate(root)

  def test_workflow_uses_exact_scheduler_artifact_and_restricted_runner(self):
    source = WORKFLOW.read_text(encoding="utf-8")
    self.assertIn("scheduler_run_id", source)
    self.assertIn("repository: sima-neat/scheduler", source)
    self.assertIn("scheduler-native-windows-amd64", source)
    self.assertIn("pcie-windows-build", source)
    self.assertNotIn("windows-latest", source)
    self.assertIn("scripts/ci/pciehost/build_windows.py", source)

  def test_producer_records_runtime_inputs_and_repeat_installation(self):
    source = SCRIPT.read_text(encoding="utf-8")
    self.assertIn('"windows-dependencies.json"', source)
    self.assertIn('"repeat_install": "PASS"', source)
    self.assertNotIn("copytree(platform_runtime", source)

  def test_wheel_payload_reader_checks_crc(self):
    with tempfile.TemporaryDirectory() as directory:
      wheel = Path(directory) / "test.whl"
      import zipfile
      with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pyneatpcie/__init__.py", b"")
      self.assertEqual(MODULE.wheel_payload_names(wheel), {"pyneatpcie/__init__.py"})


if __name__ == "__main__":
  unittest.main()
