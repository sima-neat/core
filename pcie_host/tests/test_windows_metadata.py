#!/usr/bin/env python3
"""Validate Windows package metadata composition."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "create_windows_metadata.py"
SKILL_MANIFEST = (
    Path(__file__).parents[1]
    / "skills"
    / "neat-pcie-application-builder"
    / "playbook.yml"
)
BUILD_INPUTS = (
    Path(__file__).parents[2] / "scripts" / "ci" / "pciehost" / "windows_build_inputs.json")
SPEC = importlib.util.spec_from_file_location("create_windows_metadata", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WindowsMetadataTests(unittest.TestCase):
  def test_msi_and_wheel_have_declarative_ownership_and_checksums(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      msi = root / "sima-pcie-host-0.6.1-win64.msi"
      wheel = root / "pyneatpcie-0.6.1-cp314-cp314-win_amd64.whl"
      msi.write_bytes(b"msi")
      wheel.write_bytes(b"wheel")
      result = MODULE.metadata(
          msi, wheel, "0.6.1", "develop", "a" * 40,
          platform_version="3.0.0.2", gstreamer_version="1.28.6")

    self.assertEqual(result["artifact"]["package_path"], "pciehost/windows/amd64")
    self.assertEqual(result["installation"]["type"], "windows-msi")
    self.assertEqual(result["installation"]["msi"], msi.name)
    self.assertEqual(result["installation"]["python"]["wheel"], wheel.name)
    self.assertEqual(result["installation"]["python"]["version"], "0.6.1")
    self.assertEqual(result["installation"]["python"]["requires-python"], "3.14")
    self.assertEqual(result["installation"]["python"]["arch"], "amd64")
    self.assertEqual(
        result["installation"]["verification"]["command"],
        {
            "environment": "ProgramFiles",
            "relative": "SimaPCIeHost\\bin\\sima-pcie-host-doctor.cmd",
        },
    )
    self.assertEqual(
        result["installation"]["skills"],
        [{
            "id": "neat-pcie-application-builder",
            "environment": "ProgramFiles",
            "relative": (
                "SimaPCIeHost\\share\\sima-pcie-host\\skills\\"
                "neat-pcie-application-builder"
            ),
        }],
    )
    manifest_id = next(
        line.split(":", 1)[1].strip()
        for line in SKILL_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.startswith("id:")
    )
    self.assertEqual(result["installation"]["skills"][0]["id"], manifest_id)
    prerequisites = result["installation"]["prerequisites"]
    self.assertEqual(
        [item["value"] for item in prerequisites if item["type"] == "executable"],
        ["ssh.exe", "scp.exe", "ssh-keygen.exe"],
    )
    self.assertIn(
        "SiMaMlaDaemon",
        [item.get("value") for item in prerequisites],
    )
    platform_drivers = [
        item for item in prerequisites if item["type"] == "windows-pnp-driver"
    ]
    self.assertEqual(
        [item["value"] for item in platform_drivers],
        ["SiMa.ai MLA Device", "SiMa.ai Virtual Ethernet Adapter"],
    )
    self.assertEqual(
        {item["version"] for item in platform_drivers}, {"3.0.0.2"})
    gstreamer = prerequisites[-1]
    self.assertEqual(gstreamer["type"], "command-version")
    self.assertEqual(gstreamer["version"], "1.28.6")
    self.assertEqual(
        [candidate["environment"] for candidate in gstreamer["candidates"]],
        ["GSTREAMER_1_0_ROOT_MSVC_X86_64", "GSTREAMER_ROOT_X86_64", "ProgramFiles"],
    )
    self.assertTrue(all(
        candidate["relative"].endswith("bin\\gst-inspect-1.0.exe")
        for candidate in gstreamer["candidates"]))
    build_inputs = json.loads(BUILD_INPUTS.read_text(encoding="utf-8"))
    installer = next(item for item in build_inputs["files"] if item["use"] == "gstreamer")
    self.assertIn(installer["url"], gstreamer["guidance"])
    self.assertIn(f"-{gstreamer['version']}.exe", installer["url"])
    self.assertEqual(set(result["resources"]), {msi.name, wheel.name})
    self.assertEqual(len(result["resources-checksum"][msi.name]), 64)
    self.assertEqual(len(result["resources-checksum"][wheel.name]), 64)
    self.assertIn("%USERPROFILE%\\pyneatpcie", result["installation"]["post-message"])
    self.assertIn("--native-only", result["installation"]["post-message"])
    self.assertNotIn("script", result["installation"])

  def test_native_only_metadata_does_not_request_python_installation(self):
    with tempfile.TemporaryDirectory() as temporary:
      msi = Path(temporary) / "sima-pcie-host-0.6.1-win64.msi"
      msi.write_bytes(b"msi")
      result = MODULE.metadata(
          msi, None, "0.6.1", "develop", "a" * 40,
          platform_version="3.0.0.2", gstreamer_version="1.28.6")

    self.assertEqual(result["resources"], [msi.name])
    self.assertNotIn("python", result["installation"])

  def test_scheduler_and_core_are_one_ordered_native_cohort(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      scheduler = root / "simaai-scheduler-pcie-host-0.6.1-win64.msi"
      core = root / "sima-pcie-host-0.6.1-win64.msi"
      scheduler.write_bytes(b"s" * 1024 * 1024)
      core.write_bytes(b"c" * 1024 * 1024)
      result = MODULE.metadata(
          core, None, "0.6.2", "develop", "a" * 40, scheduler, "0.6.1",
          "3.0.0.2", "1.28.6")

    self.assertEqual(result["resources"], [scheduler.name, core.name])
    self.assertEqual(
        [package["owner"] for package in result["installation"]["native-packages"]],
        ["scheduler", "core"],
    )
    self.assertEqual(
        [package["version"] for package in result["installation"]["native-packages"]],
        ["0.6.1", "0.6.2"],
    )
    self.assertNotIn("msi", result["installation"])
    self.assertEqual(set(result["resources-checksum"]), {scheduler.name, core.name})
    self.assertEqual(result["size"], {"download": "2MB", "install": "2MB"})

  def test_scheduler_cohort_rejects_missing_or_artifact_selector_version(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      scheduler = root / "scheduler.msi"
      core = root / "core.msi"
      scheduler.write_bytes(b"scheduler")
      core.write_bytes(b"core")
      for version in (None, "abcdef123456", "0.6"):
        with self.subTest(version=version), self.assertRaisesRegex(
            ValueError, "exact numeric package version"):
          MODULE.metadata(
              core, None, "0.6.2", "develop", "a" * 40,
              scheduler, version, "3.0.0.2", "1.28.6")


if __name__ == "__main__":
  unittest.main()
