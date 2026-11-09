import importlib.util
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "build_wheel.py"


def load_wheel_builder(name):
  spec = importlib.util.spec_from_file_location(name, SCRIPT)
  module = importlib.util.module_from_spec(spec)
  if spec.loader is None:
    raise RuntimeError(f"cannot load wheel builder: {SCRIPT}")
  spec.loader.exec_module(module)
  return module


class WheelMetadataTest(unittest.TestCase):
  def test_wheel_declares_image_dependencies(self):
    module = load_wheel_builder("build_pyneatpcie_wheel")
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      stage = root / "stage"
      package = stage / "pyneatpcie"
      package.mkdir(parents=True)
      (package / "__init__.py").write_text("", encoding="utf-8")

      wheel = module.write_wheel(stage, root / "dist", "1.2.3")
      with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith("/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")

      self.assertIn("Requires-Dist: numpy\n", metadata)
      self.assertIn("Requires-Dist: opencv-python\n", metadata)

  def test_wheel_platform_tag_uses_interpreter_platform(self):
    module = load_wheel_builder("build_pyneatpcie_wheel_platform")
    with mock.patch.object(module.sysconfig, "get_platform", return_value="win-amd64"):
      self.assertEqual(module.wheel_platform_tag(), "win_amd64")

      with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        stage = root / "stage"
        package = stage / "pyneatpcie"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")

        wheel = module.write_wheel(stage, root / "dist", "1.2.3")
        self.assertTrue(wheel.name.endswith("-win_amd64.whl"))

  def test_wheel_cli_supports_spaces_and_unicode(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      stage = root / "staged package δ"
      package = stage / "pyneatpcie"
      package.mkdir(parents=True)
      (package / "__init__.py").write_text("", encoding="utf-8")
      output = root / "wheel output δ"

      result = subprocess.run(
          [
              sys.executable,
              str(SCRIPT),
              "--stage-dir",
              str(stage),
              "--output-dir",
              str(output),
              "--version",
              "1.2.3",
          ],
          check=True,
          capture_output=True,
          text=True,
      )

      wheel = Path(result.stdout.strip())
      self.assertEqual(wheel.parent, output.resolve())
      self.assertTrue(wheel.is_file())
      with zipfile.ZipFile(wheel) as archive:
        self.assertIsNone(archive.testzip())


if __name__ == "__main__":
  unittest.main()
