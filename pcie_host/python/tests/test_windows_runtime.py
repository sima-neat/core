import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).parents[1] / "pyneatpcie" / "_windows_runtime.py"
SPEC = importlib.util.spec_from_file_location("pyneatpcie_windows_runtime", MODULE)
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


class WindowsRuntimeTests(unittest.TestCase):
  def test_runtime_directories_have_no_developer_fallback(self):
    package = Path("/customer/pyneatpcie/lib/site-packages/pyneatpcie")
    environment = {
        "ProgramFiles": "/program-files",
        "GSTREAMER_ROOT_X86_64": "/gstreamer",
        "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR": "/platform",
        "SIMA_PCIE_HOST_LIBRARY_PATH": "/explicit",
    }
    directories = RUNTIME.runtime_directories(
        package, environment, service_directory=Path("/service"))

    self.assertIn(Path("/explicit"), directories)
    self.assertIn(Path("/platform"), directories)
    self.assertIn(Path("/service"), directories)
    self.assertIn(Path("/gstreamer/bin"), directories)
    self.assertTrue(all(r"simaai\build" not in str(path).lower() for path in directories))

  def test_one_gstreamer_root_is_selected_and_the_explicit_override_wins(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      explicit, installer = root / "explicit", root / "per-user" / "msvc_x86_64"
      default = root / "program-files" / "gstreamer" / "1.0" / "msvc_x86_64"
      environment = {
          "ProgramFiles": str(root / "program-files"),
          "GSTREAMER_ROOT_X86_64": str(explicit),
          "GSTREAMER_1_0_ROOT_MSVC_X86_64": str(installer),
      }
      gstreamer = [explicit / "bin", installer / "bin", default / "bin"]
      select = lambda: [path for path in RUNTIME.runtime_directories(
          Path("/customer/pyneatpcie"), environment, service_directory=None)
          if path in gstreamer]
      # No installation: every candidate is listed, override first.
      self.assertEqual(select(), gstreamer)
      for path in gstreamer:
        path.mkdir(parents=True)
        (path / "gstreamer-1.0-0.dll").write_bytes(b"")
      self.assertEqual(select(), [explicit / "bin"])
      (explicit / "bin" / "gstreamer-1.0-0.dll").unlink()
      self.assertEqual(select(), [installer / "bin"])
      del environment["GSTREAMER_1_0_ROOT_MSVC_X86_64"]
      self.assertEqual(select(), [default / "bin"])

  def test_extension_load_failure_names_the_searched_directories(self):
    error = RUNTIME.extension_load_error(
        ImportError("DLL load failed while importing _pyneatpcie_core"),
        [Path("/program-files/SimaPCIeHost/bin"), Path("/gstreamer/bin")])

    message = str(error)
    self.assertIn("DLL load failed while importing _pyneatpcie_core", message)
    self.assertIn(str(Path("/program-files/SimaPCIeHost/bin")), message)
    self.assertIn(str(Path("/gstreamer/bin")), message)
    self.assertIn("SIMA_PCIE_HOST_LIBRARY_PATH", message)

  def test_loaded_dll_outside_approved_directories_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      approved = Path(directory) / "approved"
      approved.mkdir()
      (approved / "SiMaPCIe.dll").write_bytes(b"fixture")

      conflict = RUNTIME.find_loaded_dll_conflicts(
          [approved], lambda name: Path(directory) / "other" / name)

    self.assertEqual(
        conflict, [("simapcie.dll", Path(directory) / "other" / "simapcie.dll")])

  def test_unrelated_dll_loaded_from_elsewhere_is_accepted(self):
    with tempfile.TemporaryDirectory() as directory:
      approved = Path(directory) / "approved"
      approved.mkdir()
      (approved / "libssl-3-x64.dll").write_bytes(b"fixture")

      self.assertEqual(RUNTIME.find_loaded_dll_conflicts(
          [approved], lambda name: Path(directory) / "conda" / name), [])

  def test_admission_reports_the_conflicting_module_and_recovery(self):
    with tempfile.TemporaryDirectory() as directory:
      approved = Path(directory) / "approved"
      approved.mkdir()
      (approved / "SiMaPCIe.dll").write_bytes(b"fixture")
      loaded = Path(directory) / "old-runtime" / "SiMaPCIe.dll"

      with self.assertRaisesRegex(
          ImportError, r"simapcie[.]dll.*old-runtime.*Start a fresh process"):
        RUNTIME.admit_loaded_dlls([approved], lambda name: loaded)

  def test_loaded_dll_from_an_approved_directory_is_accepted(self):
    with tempfile.TemporaryDirectory() as directory:
      approved = Path(directory) / "approved"
      approved.mkdir()
      dll = approved / "SiMaPCIe.dll"
      dll.write_bytes(b"fixture")

      self.assertEqual(RUNTIME.find_loaded_dll_conflicts([approved], lambda name: dll), [])


if __name__ == "__main__":
  unittest.main()
