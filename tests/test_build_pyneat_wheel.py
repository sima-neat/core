import base64
import csv
import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from packaging.metadata import Metadata
from packaging.requirements import Requirement
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build" / "build_pyneat_wheel.py"
SPEC = importlib.util.spec_from_file_location("build_pyneat_wheel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildPyneatWheelTests(unittest.TestCase):
    def test_current_numpy_markers_resolve_and_survive_cp313_wheel(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        requirements = [Requirement(item) for item in project["dependencies"]]
        for python, accepted, rejected in (("3.12", "1.26.4", "2.1.0"),
                                           ("3.13", "2.1.0", "1.26.4"),
                                           ("3.14", "2.5.3", "3.0.0")):
            with self.subTest(python=python):
                active = [item for item in requirements if item.name == "numpy" and
                          item.marker.evaluate({"python_version": python})]
                self.assertEqual(len(active), 1)
                self.assertIn(accepted, active[0].specifier)
                self.assertNotIn(rejected, active[0].specifier)
        with tempfile.TemporaryDirectory() as directory:
            extension = Path(directory) / "_pyneat_core.cpython-313-aarch64-linux-gnu.so"
            extension.write_bytes(b"HOST_METADATA_FIXTURE_NOT_ELF")
            wheel = MODULE.build_wheel(ROOT, extension, Path(directory) / "dist", "1.2.3",
                                       "cp313", "cp313", "linux_aarch64")
            with zipfile.ZipFile(wheel) as archive:
                metadata = Metadata.from_email(archive.read("pyneat-1.2.3.dist-info/METADATA"), validate=True)
                numpy = [item for item in metadata.requires_dist if item.name == "numpy"]
                self.assertEqual(len(numpy), 2)
                active = [item for item in numpy if item.marker.evaluate({"python_version": "3.13"})]
                self.assertEqual(len(active), 1)
                self.assertIn("2.5.3", active[0].specifier)

    def test_packages_existing_extension_with_valid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "python" / "pyneat").mkdir(parents=True)
            (root / "python" / "pyneat" / "__init__.py").write_text(
                "from . import _pyneat_core\n", encoding="utf-8"
            )
            (root / "python" / "pyneat" / "py.typed").write_text("", encoding="utf-8")
            (root / "LICENSE").write_text("test license\n", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                """
[project]
name = "pyneat"
version = "1.2.3"
description = "test wheel"
requires-python = ">=3.9"
dependencies = ["numpy>=1.24,<2"]
authors = [{ name = "SiMa.ai" }]
classifiers = ["Programming Language :: Python :: 3"]

[project.optional-dependencies]
torch = ["torch>=2.3.0"]
""".lstrip(),
                encoding="utf-8",
            )
            extension = root / "_pyneat_core.cpython-311-aarch64-linux-gnu.so"
            extension.write_bytes(b"compiled-extension")

            wheel = MODULE.build_wheel(
                root,
                extension,
                root / "dist",
                "1.2.3",
                "cp311",
                "cp311",
                "manylinux_2_31_aarch64",
            )

            self.assertEqual(
                wheel.name,
                "pyneat-1.2.3-cp311-cp311-manylinux_2_31_aarch64.whl",
            )
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                self.assertIn(f"pyneat/{extension.name}", names)
                self.assertIn("pyneat/__init__.py", names)
                metadata = archive.read("pyneat-1.2.3.dist-info/METADATA").decode()
                Metadata.from_email(metadata.encode(), validate=True)
                self.assertIn("Metadata-Version: 2.3", metadata)
                self.assertNotIn("License-File:", metadata)
                self.assertIn("Requires-Dist: numpy>=1.24,<2", metadata)
                self.assertIn(
                    'Requires-Dist: torch>=2.3.0; extra == "torch"',
                    metadata,
                )

                record = archive.read("pyneat-1.2.3.dist-info/RECORD").decode()
                rows = list(csv.reader(io.StringIO(record)))
                for path, digest, size in rows:
                    if path.endswith("/RECORD"):
                        self.assertEqual((digest, size), ("", ""))
                        continue
                    data = archive.read(path)
                    expected = "sha256=" + base64.urlsafe_b64encode(
                        hashlib.sha256(data).digest()
                    ).rstrip(b"=").decode()
                    self.assertEqual(digest, expected)
                    self.assertEqual(int(size), len(data))


if __name__ == "__main__":
    unittest.main()
