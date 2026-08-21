"""Focused tests for Model Compiler detection in the `neat` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
import venv
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
NEAT = ROOT / "scripts" / "neat"


def make_venv(path: Path, distribution: str | None = None, version: str = "1.2.3") -> Path:
    """Create a venv, optionally with `distribution` metadata installed in it."""
    venv.create(path, with_pip=False, symlinks=True)
    if distribution is not None:
        site_packages = next(path.glob("lib/python*/site-packages"))
        dist_info = site_packages / f"{distribution.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir(parents=True)
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
            encoding="utf-8",
        )
        (dist_info / "RECORD").write_text("", encoding="utf-8")
    return path


def model_compiler_status(extension_dir: Path | str) -> dict:
    """Run `neat status --json --offline` and return the Model Compiler component."""
    result = subprocess.run(
        # Resolve NEAT at call time so --neat still applies.
        ["bash", str(NEAT), "status", "--json", "--offline"],
        check=False,
        text=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "NEAT_MODEL_COMPILER_DIR": str(extension_dir),
            # Keep discovery hermetic: no registry, no package metadata lookups.
            "SIMA_CLI_REGISTRY_FILE": "/nonexistent/registry.json",
        },
    )
    payload = json.loads(result.stdout)
    components = payload.get("components")
    entries = components.values() if isinstance(components, dict) else components
    for component in entries:
        if isinstance(component, dict) and component.get("name") == "Model Compiler":
            return component
    raise AssertionError("Model Compiler component missing from neat status output")


class ModelCompilerDiscoveryTest(unittest.TestCase):
    def test_missing_extension_dir_reports_not_installed(self) -> None:
        component = model_compiler_status("/nonexistent/model-compiler")
        self.assertFalse(component["installed"])

    def test_bare_venv_is_not_reported_as_installed(self) -> None:
        """A venv without any Model Compiler distribution is not a usable install."""
        with TemporaryDirectory() as tmp:
            extension_dir = make_venv(Path(tmp) / "model-compiler")
            component = model_compiler_status(extension_dir)
        self.assertFalse(
            component["installed"],
            "bare venv (no Model Compiler distribution) must report as missing",
        )
        self.assertIsNone(component["version"])

    def test_installed_distribution_reports_version(self) -> None:
        for distribution in ("sima-frontend", "sima-mlc", "sima-lmm"):
            with self.subTest(distribution=distribution):
                with TemporaryDirectory() as tmp:
                    extension_dir = make_venv(
                        Path(tmp) / "model-compiler", distribution, "1.2.3"
                    )
                    component = model_compiler_status(extension_dir)
                self.assertTrue(component["installed"])
                self.assertEqual(component["version"], "1.2.3")


def _take_path_option(flag: str) -> Path | None:
    """Pop `--flag <path>` out of argv; under CTest the script is not at __file__."""
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    try:
        value = Path(sys.argv[index + 1]).resolve()
    except IndexError as exc:
        raise SystemExit(f"{flag} requires a path") from exc
    del sys.argv[index : index + 2]
    return value


if __name__ == "__main__":
    NEAT = _take_path_option("--neat") or NEAT
    unittest.main()
