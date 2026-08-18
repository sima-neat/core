import importlib.util
import zipfile
from pathlib import Path


def test_wheel_declares_image_dependencies(tmp_path):
  script = Path(__file__).parents[1] / "build_wheel.py"
  spec = importlib.util.spec_from_file_location("build_pyneatpcie_wheel", script)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)

  stage = tmp_path / "stage"
  package = stage / "pyneatpcie"
  package.mkdir(parents=True)
  (package / "__init__.py").write_text("", encoding="utf-8")

  wheel = module.write_wheel(stage, tmp_path / "dist", "1.2.3")
  with zipfile.ZipFile(wheel) as archive:
    metadata_name = next(name for name in archive.namelist() if name.endswith("/METADATA"))
    metadata = archive.read(metadata_name).decode("utf-8")

  assert "Requires-Dist: numpy\n" in metadata
  assert "Requires-Dist: opencv-python\n" in metadata
