#!/usr/bin/env python3
"""Contracts for the native Windows Core PCIe-host producer."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "pciehost" / "build_windows.py"
SCRIPT_DIR = SCRIPT.parent
WORKFLOW = ROOT / ".github" / "workflows" / "pcie-host-windows.yml"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("build_windows", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
from windows_candidate import sha256, validate_candidate  # noqa: E402
from qualify_windows_hardware import ExclusiveLock  # noqa: E402
from verify_windows_release import verify  # noqa: E402


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
    self.assertIn("pcie-windows-hardware", source)
    self.assertNotIn("windows-latest", source)
    self.assertIn("scripts/ci/pciehost/build_windows.py", source)
    self.assertIn("scripts/ci/pciehost/qualify_windows_hardware.py", source)
    self.assertIn("scripts/ci/pciehost/cleanup_windows_hardware.py", source)
    self.assertIn("needs: hardware", source)
    self.assertIn("--require-signatures", source)
    self.assertIn("Recover card and native installation", source)
    self.assertIn("if: always()", source)

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

  def windows_candidate(self, root: Path, *, signed: bool = True):
    scheduler = root / "scheduler.msi"
    core = root / "core.msi"
    wheel = root / "pyneatpcie.whl"
    for path, content in ((scheduler, b"scheduler"), (core, b"core"), (wheel, b"wheel")):
      path.write_bytes(content)
    resources = {path.name: sha256(path) for path in (scheduler, core, wheel)}
    commit = "c" * 40
    (root / "metadata.json").write_text(json.dumps({
        "artifact": {"package_path": "pciehost/windows/amd64"},
        "commit": commit,
        "installation": {
            "native-packages": [
                {"owner": "scheduler", "msi": scheduler.name},
                {"owner": "core", "msi": core.name},
            ],
            "python": {"wheel": wheel.name},
            "type": "windows-msi",
        },
        "name": "gh:sima-neat/core/pciehost/windows/amd64",
        "repository": "sima-neat/core",
        "resources": list(resources),
        "resources-checksum": resources,
    }), encoding="utf-8")
    (root / "host-candidate.json").write_text(json.dumps({
        "core_source_commit": commit,
        "profile": "pciehost/windows/amd64",
        "repository": "sima-neat/core",
        "scheduler_source_commit": "d" * 40,
    }), encoding="utf-8")
    (root / "windows-signatures.json").write_text(json.dumps({
        "certificate_sha1": "A" * 40 if signed else "",
        "records": [{
            "path": scheduler.name,
            "status": "Valid" if signed else "NotSigned",
            "timestamp_thumbprint": "B" * 40 if signed else None,
            "thumbprint": "A" * 40 if signed else None,
        }],
        "release_required": signed,
        "timestamp_url": "https://timestamp.example" if signed else None,
    }), encoding="utf-8")
    qualification = root / "qualification.json"
    qualification.write_text(json.dumps({
        "candidate": {"core_commit": commit, "resources": resources},
        "card_final": {"genai_workers": "0", "workers": "0"},
        "cleanup": {"errors": []},
        "result": "PASS",
        "tests": {"cpp": ["test_tensor_run"], "python": ["test_tensor_run_yolov8"]},
    }), encoding="utf-8")
    return commit, resources, qualification

  def test_candidate_and_release_gate_bind_exact_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, resources, qualification = self.windows_candidate(root)
      candidate = validate_candidate(root)
      self.assertEqual(candidate.resources, resources)
      admission = verify(root, qualification, commit, True)
      self.assertTrue(admission["hardware_qualified"])
      self.assertTrue(admission["signatures_valid"])
      candidate.core_msi.write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "checksum differs"):
        validate_candidate(root)

  def test_release_gate_rejects_unsigned_candidate(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root, signed=False)
      with self.assertRaisesRegex(ValueError, "not built in signed release mode"):
        verify(root, qualification, commit, True)

  def test_release_gate_rejects_missing_genai_matrix(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      with self.assertRaisesRegex(ValueError, "complete Windows GenAI matrix"):
        verify(root, qualification, commit, True, True)

  def test_producer_retains_cpp_hardware_extras_and_signature_receipt(self):
    source = SCRIPT.read_text(encoding="utf-8")
    self.assertIn('"PcieHostExtras"', source)
    self.assertIn('"windows-signatures.json"', source)
    self.assertIn("signing_certificate_sha1", source)

  def test_signing_inputs_fail_before_signtool(self):
    with self.assertRaisesRegex(ValueError, "certificate SHA-1"):
      MODULE.sign_files([], "signtool", "not-a-thumbprint", "https://timestamp.example", False)
    with self.assertRaisesRegex(ValueError, "must use HTTPS"):
      MODULE.sign_files([], "signtool", "A" * 40, "http://timestamp.example", False)

  def test_signing_selects_only_installed_native_outputs(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      build = root / "build"
      stage = root / "stage"
      build.mkdir()
      stage.mkdir()
      selected = build / "customer.dll"
      selected.write_bytes(b"dll")
      (build / "unit-test.exe").write_bytes(b"test")
      (stage / selected.name).write_bytes(b"dll")
      self.assertEqual(MODULE.build_sources_for_installed(build, [stage]), [selected])

  def test_hardware_lock_is_atomic_and_can_be_retained_for_recovery(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "card.lock"
      with ExclusiveLock(path):
        self.assertTrue(path.is_file())
        with self.assertRaisesRegex(RuntimeError, "already exists"):
          with ExclusiveLock(path):
            pass
      self.assertFalse(path.exists())
      with ExclusiveLock(path) as lock:
        lock.retain()
      self.assertTrue(path.is_file())


if __name__ == "__main__":
  unittest.main()
