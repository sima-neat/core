"""Import the source-owned Kernel Library export into the full staging portal.

The exporter, not Core, owns operation discovery, translations and API generation.
Only verified chapter files are mounted; the standalone preview shell is never used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

LOCALES = ("en", "ko", "ja", "zh-Hant", "uk")
REQUIRED = ("index.md", "a65/index.md", "ev74/index.md", "reference/cpp/index.md")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_key(source: Path) -> str:
    # Numerical implementation changes do not invalidate documentation. Include
    # inventories, public headers, translations, snippets and generator checks.
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    inputs = {}
    for name in filter(None, files):
        path = PurePosixPath(name)
        if (path.suffix in (".md", ".mdx", ".h", ".hpp", ".cmake")
                or path.name == "CMakeLists.txt"
                or path.parts[0] in ("docs", "i18n", "tools", ".agents")
                or name.startswith("tests/docs/") or name == "sima-i18n.config.json"):
            inputs[name] = digest(source / name)
    inputs["importer"] = digest(Path(__file__))
    for tool in ("doxygen", "dot", "node"):
        command = [tool, "-V" if tool == "dot" else "--version"]
        inputs[tool] = subprocess.check_output(command, stderr=subprocess.STDOUT).decode().strip()
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def verify_export(export: Path, locale: str) -> dict:
    manifest = json.loads((export / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("locale") != locale:
        raise ValueError(f"Invalid Kernel Library manifest for {locale}")
    source = manifest.get("source", {})
    if source.get("dirty") or not source.get("git_revision") or not source.get("tree_sha256"):
        raise ValueError("Kernel documentation must come from a clean, identified source")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty Kernel Library export")
    actual = {}
    for path in export.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in Kernel Library export: {path}")
        if path.is_file() and path != export / "manifest.json":
            name = path.relative_to(export).as_posix()
            if not name.startswith("kernel-library/"):
                raise ValueError(f"File outside Kernel Library: {name}")
            actual[name] = digest(path)
    if actual != files:
        raise ValueError("Kernel Library export inventory/hash mismatch")
    for name in REQUIRED:
        if f"kernel-library/{name}" not in files:
            raise ValueError(f"Missing Kernel Library page: {name}")
    return manifest


def mount(source: Path, build: Path, docs: Path, i18n: Path) -> str:
    key = input_key(source)
    cache = build / "kernel-docs-cache" / key
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache.parent, prefix="export-") as temporary:
            stage = Path(temporary)
            for locale in LOCALES:
                subprocess.run([sys.executable, str(source / "tools/docs/build.py"),
                                "--source", str(source), "--locale", locale,
                                "--output", str(stage / locale)], check=True)
                verify_export(stage / locale, locale)
            stage.rename(cache)
    # Verify even cache hits: missing or corrupt exports must never silently
    # remove a chapter from a successful full-portal build.
    manifests = {locale: verify_export(cache / locale, locale) for locale in LOCALES}
    for locale in LOCALES:
        root = docs if locale == "en" else i18n / locale / "docusaurus-plugin-content-docs/current"
        destination = root / "kernel-library"
        if destination.is_symlink():
            raise ValueError(f"Refusing symlink mount: {destination}")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(cache / locale / "kernel-library", destination)
    # Retain both selected revision and cached export provenance. They can differ
    # when only numerical implementation files changed; never relabel a cache.
    provenance = {"input_key": key, "selected_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(), "exports": manifests}
    (build / "kernel-docs-import.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return f"verified Kernel Library ({', '.join(LOCALES)}), input key {key}"


def unmount(build: Path, docs: Path, i18n: Path | None) -> None:
    """Remove only generated chapter mounts when reusing a staging workspace."""
    roots = [docs]
    if i18n is not None:
        roots.extend(i18n / locale / "docusaurus-plugin-content-docs/current"
                     for locale in LOCALES if locale != "en")
    for root in roots:
        destination = root / "kernel-library"
        if destination.is_symlink():
            raise ValueError(f"Refusing symlink mount: {destination}")
        if destination.exists():
            shutil.rmtree(destination)
    (build / "kernel-docs-import.json").unlink(missing_ok=True)
