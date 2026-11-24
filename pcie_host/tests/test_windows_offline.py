#!/usr/bin/env python3
"""Contracts for the Windows PCIe-host offline bundle: producer, gate and publication."""

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "ci" / "pciehost"
WORKFLOW = ROOT / ".github" / "workflows" / "pcie-host-windows.yml"
MAIN_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-ci.yml"
sys.path.insert(0, str(SCRIPT_DIR))
import windows_offline as OFFLINE  # noqa: E402
import validate_windows_offline as GATE  # noqa: E402
from create_windows_metadata import gstreamer_installer_url  # noqa: E402
from qualify_windows_hardware import CARD_BUNDLE_ARTIFACT  # noqa: E402

COMMIT = "c" * 40
VERSION = "0.7.0"


def sha(data):
  return hashlib.sha256(data).hexdigest()


def python_with_pip():
  """pip resolves the closure; the contract job's python3 has it, a dev box may not."""
  for python in (sys.executable, shutil.which("python3")):
    if python and subprocess.run([python, "-m", "pip", "--version"], capture_output=True,
                                 check=False).returncode == 0:
      return python
  raise RuntimeError("these tests need a Python interpreter with pip")


def write_wheel(directory, name, version, requires=()):
  """A py3-none-any wheel with only what pip download reads."""
  stem = f"{name.replace('-', '_')}-{version}"
  path = directory / f"{stem}-py3-none-any.whl"
  metadata = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
              + "".join(f"Requires-Dist: {item}\n" for item in requires) + "\n")
  with zipfile.ZipFile(path, "w") as archive:
    archive.writestr(f"{stem}.dist-info/METADATA", metadata)
    archive.writestr(f"{stem}.dist-info/WHEEL",
                     "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    archive.writestr(f"{stem}.dist-info/RECORD", "")
  return path


def pin(use, path, url=None):
  return {"use": use, "name": path.name,
          "url": url or f"https://files.example/{path.name}", "sha256": sha(path.read_bytes())}


class Fixture:
  """A validated candidate, the build's wheelhouse and the card bundle of one commit."""

  def __init__(self, root):
    self.root = root
    self.candidate = root / "candidate"
    self.wheelhouse = root / "wheelhouse"
    self.card = root / "card"
    for directory in (self.candidate, self.wheelhouse, self.card):
      directory.mkdir()
    house = self.wheelhouse
    self.dependencies = [write_wheel(house, "numpy", "2.5.4"),
                         write_wheel(house, "opencv-python", "5.0.0.93", ["numpy"])]
    self.test_only = [write_wheel(house, "pytest", "9.2.0"),
                      write_wheel(house, "pillow", "13.0.0")]
    self.sima_cli = [write_wheel(house, "sima-cli", "2.1.23", ["click"]),
                     write_wheel(house, "click", "8.3.0")]
    self.inputs = root / "windows_build_inputs.json"
    self.write_inputs()
    wheel = write_wheel(self.candidate, "pyneatpcie", VERSION, ["numpy", "opencv-python"])
    scheduler = self.candidate / "simaai-scheduler-pcie-host-1.2.3-win64.msi"
    core = self.candidate / f"sima-pcie-host-{VERSION}-win64.msi"
    scheduler.write_bytes(b"scheduler msi")
    core.write_bytes(b"core msi")
    resources = {path.name: sha(path.read_bytes()) for path in (scheduler, core, wheel)}
    self.metadata = {
        "name": "gh:sima-neat/core/pciehost/windows/amd64",
        "version": VERSION, "release": "v0.7.0",
        "platforms": [{"type": "host", "os": ["windows"], "arch": ["amd64"]}],
        "resources": list(resources), "resources-checksum": resources,
        "selectable-resources": [],
        "installation": {
            "type": "windows-msi",
            "prerequisites": [
                {"name": "Windows OpenSSH client", "type": "executable", "value": "ssh.exe"},
                {"name": "SiMa Platform MLA driver", "type": "windows-pnp-driver",
                 "value": "SiMa.ai MLA Device", "version": "3.0.0.2"},
                {"name": "SiMa Platform PCIe service", "type": "windows-service",
                 "value": "SiMaMlaDaemon"},
                {"name": "GStreamer 1.28.6 MSVC x64 runtime", "type": "command-version",
                 "version": "1.28.6"},
            ],
            "native-packages": [
                {"owner": "scheduler", "msi": scheduler.name, "version": "1.2.3"},
                {"owner": "core", "msi": core.name, "version": VERSION}],
            "verification": {"command": {"environment": "ProgramFiles", "relative": "doctor"}},
            "skills": [{"id": "neat-pcie-application-builder"}],
            "setup": {"command": {"environment": "ProgramFiles", "relative": "setup"}},
            "python": {"wheel": wheel.name, "distribution": "pyneatpcie", "version": VERSION,
                       "import": "pyneatpcie", "requires-python": "3.14", "arch": "amd64"},
        },
        "artifact": {"type": "windows-msi", "repository": "core",
                     "package_path": "pciehost/windows/amd64", "distro": "windows",
                     "arches": ["amd64"]},
        "repository": "sima-neat/core", "branch": "v0.7.0", "commit": COMMIT,
        "published_at_utc": "2026-11-24T00:00:00+00:00",
        "size": {"download": "3MB", "install": "3MB"},
    }
    (self.candidate / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
    dll = [{"name": "gstreamer-1.0-0.dll", "sha256": "1" * 64}]
    (self.candidate / "windows-dependencies.json").write_text(json.dumps({
        "platform": {"version": "3.0.0.2"},
        "gstreamer": {"version_output": "gst-inspect-1.0 version 1.28.6",
                      "pkg_config_sha256": "2" * 64, "runtime_dlls": dll},
        "dependency_runtime": [{"files": [{"name": "opencv_world500.dll",
                                           "sha256": "3" * 64}]}]}))
    for name in ("msi-transaction-rollback.json", "msi-transaction-rollback.log",
                 "scheduler-platform-windows.json", "scheduler-windows-signatures.json",
                 "scheduler-windows-dependencies.json", "windows-signatures.json"):
      (self.candidate / name).write_text("{}\n")
    qualification = {path.name: sha(path.read_bytes()) for path in self.candidate.iterdir()
                     if path.suffix in (".json", ".log") and path.name != "metadata.json"}
    self.write_receipt(qualification)

    archive = self.card / "neat-core-offline-installer.tar.gz"
    archive.write_bytes(b"card bundle")
    (self.card / "metadata.json").write_text(json.dumps({
        "name": "gh:sima-neat/core/offline", "artifact": CARD_BUNDLE_ARTIFACT,
        "version": f"0.7.0.{COMMIT[:12]}", "installer_revision": COMMIT,
        "resources": [archive.name],
        "resources-checksum": {archive.name: sha(archive.read_bytes())}}))

  def write_receipt(self, qualification=None):
    path = self.candidate / "host-candidate.json"
    if qualification is None:
      qualification = json.loads(path.read_text())["qualification_resources"]
    path.write_text(json.dumps({
        "repository": "sima-neat/core", "profile": "pciehost/windows/amd64",
        "core_source_commit": COMMIT, "qualification_resources": qualification,
        "python": {"wheelhouse": [{"name": path.name, "sha256": sha(path.read_bytes())}
                                  for path in sorted(self.wheelhouse.iterdir())]}}))

  def write_inputs(self, change=None):
    manifest = {
        "schema_version": 1, "sima_cli_version": "2.1.23", "sima_cli_publication": "x",
        "files": [
            {"use": "gstreamer", "name": "gstreamer-1.0-msvc-x86_64-1.28.6.exe",
             "url": gstreamer_installer_url("1.28.6"), "sha256": "4" * 64, "size": 500},
            {"use": "python-installer", "name": "python-3.14.7-amd64.exe",
             "url": "https://www.python.org/ftp/python/3.14.7/python-3.14.7-amd64.exe",
             "sha256": "5" * 64, "size": 30},
            *(pin("wheelhouse", path) for path in self.dependencies + self.test_only),
            *(pin("sima-cli", path) for path in self.sima_cli),
        ]}
    if change:
      change(manifest)
    self.inputs.write_text(json.dumps(manifest, indent=2))

  def assemble(self, output, ref=f"v{VERSION}"):
    return OFFLINE.assemble_bundle(self.candidate, self.wheelhouse, self.card, self.inputs,
                                   "a" * 40, 42, ref, output)


class _Case(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.patch = mock.patch.object(OFFLINE, "PYTHON", python_with_pip())
    cls.patch.start()

  @classmethod
  def tearDownClass(cls):
    cls.patch.stop()

  def setUp(self):
    self.root = Path(tempfile.mkdtemp(prefix="neat offline "))
    self.addCleanup(shutil.rmtree, self.root)
    self.fixture = Fixture(self.root)

  def assemble(self):
    output = self.root / "out"
    archive = self.fixture.assemble(output)
    return output, archive


class AssembleContracts(_Case):
  def test_bundle_is_the_candidate_plus_pinned_closure_card_and_sima_cli(self):
    output, archive = self.assemble()
    tree = output / OFFLINE.TOP
    files = sorted(path.relative_to(tree).as_posix() for path in tree.rglob("*")
                   if path.is_file())
    self.assertEqual(files, sorted([
        "INSTALL.md", "metadata.json", "card/neat-core-offline-installer.tar.gz",
        *self.fixture.metadata["resources"],
        "wheelhouse/numpy-2.5.4-py3-none-any.whl",
        "wheelhouse/opencv_python-5.0.0.93-py3-none-any.whl",
        "sima-cli/requirements.txt", "sima-cli/click-8.3.0-py3-none-any.whl",
        "sima-cli/sima_cli-2.1.23-py3-none-any.whl"]))
    # The test-only wheels the build also pins never reach a customer.
    self.assertFalse(any("pytest" in name or "pillow" in name for name in files))
    for name in self.fixture.metadata["resources"]:
      self.assertEqual((tree / name).read_bytes(), (self.fixture.candidate / name).read_bytes())
    self.assertEqual((tree / "INSTALL.md").read_bytes(),
                     (OFFLINE.OFFLINE / "WINDOWS_INSTALL.md").read_bytes())

    with tarfile.open(archive) as bundle:
      members = bundle.getmembers()
    self.assertTrue(all(member.name == OFFLINE.TOP or member.name.startswith(OFFLINE.TOP + "/")
                        for member in members))
    self.assertEqual({member.name.split("/", 1)[1] for member in members if member.isfile()},
                     set(files))

  def test_install_metadata_adds_only_resources_checksums_and_wheelhouse(self):
    output, _ = self.assemble()
    metadata = json.loads((output / OFFLINE.TOP / "metadata.json").read_text())
    original = self.fixture.metadata
    self.assertEqual(metadata["resources"][:3], original["resources"])
    for name, digest in original["resources-checksum"].items():
      self.assertEqual(metadata["resources-checksum"][name], digest)
    self.assertTrue(all("\\" not in name for name in metadata["resources"]))
    self.assertEqual(metadata["installation"]["python"]["wheelhouse"], [
        {"wheel": "wheelhouse/numpy-2.5.4-py3-none-any.whl", "distribution": "numpy",
         "version": "2.5.4"},
        {"wheel": "wheelhouse/opencv_python-5.0.0.93-py3-none-any.whl",
         "distribution": "opencv-python", "version": "5.0.0.93"}])
    stripped = copy.deepcopy(metadata)
    stripped["resources"] = stripped["resources"][:3]
    stripped["resources-checksum"] = {name: stripped["resources-checksum"][name]
                                      for name in stripped["resources"]}
    del stripped["installation"]["python"]["wheelhouse"]
    self.assertEqual(stripped, original)

  def test_sima_cli_requirements_pin_every_file_by_hash(self):
    output, _ = self.assemble()
    text = (output / OFFLINE.TOP / "sima-cli" / "requirements.txt").read_text()
    self.assertEqual(text.splitlines(), [
        f"{name}=={version} --hash=sha256:{sha(path.read_bytes())}"
        for (name, version), path in zip((("click", "8.3.0"), ("sima-cli", "2.1.23")),
                                         sorted(self.fixture.sima_cli))])

  def test_download_metadata_follows_the_linux_offline_conventions(self):
    output, archive = self.assemble()
    offline = json.loads((output / OFFLINE.OFFLINE_METADATA).read_text())
    self.assertEqual(offline["name"], "gh:sima-neat/core/pciehost/windows/amd64/offline")
    self.assertEqual(offline["installation"]["script"], "")
    self.assertIn(f"follow neat-pcie-offline-installer-{sha(archive.read_bytes())[:8]}"
                  "\\neat-offline\\INSTALL.md.", offline["installation"]["post-message"])
    self.assertEqual(offline["platforms"], [])
    self.assertEqual(offline["artifact"]["type"], "offline")
    self.assertEqual(offline["resources"], [
        archive.name, gstreamer_installer_url("1.28.6"),
        "https://www.python.org/ftp/python/3.14.7/python-3.14.7-amd64.exe"])
    self.assertEqual(set(offline["resources-checksum"]), set(offline["resources"]))
    digest = sha(archive.read_bytes())
    self.assertEqual(archive.name, f"neat-pcie-offline-installer-{digest[:8]}.tar.gz")
    self.assertEqual(offline["resources-checksum"][archive.name], digest)
    self.assertEqual(offline["source_metadata_sha256"], {
        "host": sha((self.fixture.candidate / "metadata.json").read_bytes()),
        "card": sha((self.fixture.card / "metadata.json").read_bytes())})
    self.assertEqual((offline["installer_revision"], offline["source_run_id"]), ("a" * 40, 42))
    for key in ("version", "release", "repository", "branch", "commit", "published_at_utc"):
      self.assertEqual(offline[key], self.fixture.metadata[key])
    inventory = json.loads((output / OFFLINE.INVENTORY).read_text())
    self.assertEqual(inventory["online_resources"], self.fixture.metadata["resources-checksum"])
    self.assertEqual(inventory["archive"], {"name": archive.name, "sha256": digest})

  def test_unpinned_or_changed_dependency_is_refused(self):
    cases = {
        "not pinned": lambda manifest: manifest["files"].pop(2),
        "differs from its pinned SHA-256": lambda manifest: manifest["files"][2].update(
            sha256="0" * 64),
    }
    for message, change in cases.items():
      with self.subTest(message):
        self.fixture.write_inputs(change)
        with self.assertRaisesRegex(ValueError, message):
          self.fixture.assemble(self.root / message)
    self.fixture.write_inputs()
    receipt = json.loads((self.fixture.candidate / "host-candidate.json").read_text())
    receipt["python"]["wheelhouse"][0]["sha256"] = "0" * 64
    (self.fixture.candidate / "host-candidate.json").write_text(json.dumps(receipt))
    with self.assertRaisesRegex(ValueError, "differs from the build's wheelhouse record"):
      self.fixture.assemble(self.root / "record")

  @unittest.skipIf(sys.platform == "win32", "the target interpreter evaluates markers right")
  def test_closure_with_environment_markers_needs_the_target_interpreter(self):
    # pip on another OS would drop this dependency from the Windows closure.
    write_wheel(self.fixture.wheelhouse, "pywin32-ctypes", "0.2.3")
    wheel = write_wheel(self.root, "pyneatpcie", VERSION,
                        ["numpy", 'pywin32-ctypes; sys_platform == "win32"'])
    with self.assertRaisesRegex(ValueError, "environment markers.*CPython 3.14 x64 on Windows"):
      OFFLINE.resolve_closure(wheel, self.fixture.wheelhouse,
                              self.fixture.metadata["installation"]["python"])

  def test_card_bundle_version_follows_the_ref(self):
    # A branch build's card version ends in the commit; a release tag build's is the tag.
    self.fixture.assemble(self.root / "branch", ref="develop")
    path = self.fixture.card / "metadata.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), version=VERSION)))
    self.fixture.assemble(self.root / "tag", ref=f"v{VERSION}")
    with self.assertRaisesRegex(ValueError, "not tied to the selected Core commit"):
      self.fixture.assemble(self.root / "other branch", ref="develop")

  def test_vendor_pins_must_be_the_installers_the_package_requires(self):
    cases = {
        "GStreamer pin": lambda files: files[0].update(
            name="gstreamer-1.0-msvc-x86_64-1.28.5.exe", url=gstreamer_installer_url("1.28.5")),
        "CPython 3.14": lambda files: files[1].update(
            name="python-3.13.9-amd64.exe",
            url="https://www.python.org/ftp/python/3.13.9/python-3.13.9-amd64.exe"),
        "size": lambda files: files[1].pop("size"),
    }
    for message, change in cases.items():
      with self.subTest(message):
        self.fixture.write_inputs(lambda manifest: change(manifest["files"]))
        with self.assertRaisesRegex(ValueError, message):
          self.fixture.assemble(self.root / message)

  def test_verify_refuses_missing_extra_changed_and_linked_files(self):
    output, _ = self.assemble()
    pristine = output / OFFLINE.TOP
    OFFLINE.verify(pristine)
    cases = {
        "missing": lambda tree: (tree / "INSTALL.md").unlink(),
        "unexpected": lambda tree: (tree / "wheelhouse" / "extra.whl").write_bytes(b"x"),
        "checksum differs": lambda tree: (tree / "card" /
                                          "neat-core-offline-installer.tar.gz").write_bytes(b"y"),
        "symlinks": lambda tree: (tree / "link").symlink_to(tree / "INSTALL.md"),
    }
    for message, change in cases.items():
      with self.subTest(message):
        tree = self.root / message
        shutil.copytree(pristine, tree)
        change(tree)
        with self.assertRaisesRegex(ValueError, message):
          OFFLINE.verify(tree)


class MemoryStorage:
  def __init__(self, objects):
    self.objects, self.writes = dict(objects), []

  def read(self, key):
    try:
      return self.objects[key]
    except KeyError:
      raise FileNotFoundError(key) from None

  def has_object(self, key, digest):
    return key in self.objects and sha(self.objects[key]) == digest

  def put_resource(self, key, path):
    self.writes.append(key)
    self.objects[key] = path.read_bytes()

  put = put_resource

  def invalidate(self, path):
    self.writes.append("invalidate " + path)


class PublishContracts(_Case):
  PREFIX = f"core/v{VERSION}/{COMMIT[:12]}/pciehost/windows/amd64"

  def setUp(self):
    super().setUp()
    self.output, self.archive = self.assemble()
    self.live = {f"{self.PREFIX}/metadata.json":
                 (self.fixture.candidate / "metadata.json").read_bytes()}

  def publish(self, storage):
    return OFFLINE.publish(storage, f"v{VERSION}", COMMIT, self.output)

  def test_uploads_archive_then_metadata_then_invalidates(self):
    storage = MemoryStorage(self.live)
    self.publish(storage)
    self.assertEqual(storage.writes, [
        f"{self.PREFIX}/{self.archive.name}", f"{self.PREFIX}/metadata-offline.json",
        f"invalidate /{self.PREFIX}/metadata-offline.json"])
    # Nothing of the online release is written.
    self.assertEqual(storage.objects[f"{self.PREFIX}/metadata.json"],
                     self.live[f"{self.PREFIX}/metadata.json"])

  def test_same_bundle_is_a_no_op_and_another_is_refused(self):
    storage = MemoryStorage(self.live)
    self.publish(storage)
    again = MemoryStorage(storage.objects)
    self.assertEqual(self.publish(again), [])
    self.assertEqual(again.writes, [])
    changed = dict(storage.objects)
    changed[f"{self.PREFIX}/metadata-offline.json"] += b" "
    with self.assertRaisesRegex(ValueError, "never overwritten"):
      self.publish(MemoryStorage(changed))

  def test_refuses_another_online_release(self):
    metadata = dict(self.fixture.metadata, published_at_utc="2026-11-25T00:00:00+00:00")
    storage = MemoryStorage({f"{self.PREFIX}/metadata.json": json.dumps(metadata).encode()})
    with self.assertRaisesRegex(ValueError, "not the online release"):
      self.publish(storage)
    self.assertEqual(storage.writes, [])

  def test_bundled_sima_cli_must_be_a_release_or_a_main_build(self):
    path = self.output / OFFLINE.INVENTORY
    inventory = json.loads(path.read_text())
    for version in ("2.1.23", "2.1.23+main.0123456789ab"):
      with self.subTest(version):
        path.write_text(json.dumps(dict(inventory, sima_cli={"version": version, "files": 2})))
        self.publish(MemoryStorage(self.live))
    for version in ("2.1.23+feat.windows.offline.install.0123456789ab",
                    "2.1.23+codex.1074.windows.msi.install.91d5420c7db9", "2.1.23rc1"):
      with self.subTest(version):
        path.write_text(json.dumps(dict(inventory, sima_cli={"version": version, "files": 2})))
        storage = MemoryStorage(self.live)
        with self.assertRaisesRegex(ValueError, "never another branch"):
          self.publish(storage)
        self.assertEqual(storage.writes, [])

  def test_refuses_a_branch_or_other_commit(self):
    with self.assertRaisesRegex(ValueError, "release tag"):
      OFFLINE.publish(MemoryStorage(self.live), "develop", COMMIT, self.output)
    with self.assertRaisesRegex(ValueError, "names commit"):
      OFFLINE.publish(MemoryStorage(self.live), f"v{VERSION}", "d" * 40, self.output)


class GateContracts(unittest.TestCase):
  def test_hosted_metadata_drops_only_the_platform_steps(self):
    with tempfile.TemporaryDirectory() as directory:
      original = Fixture(Path(directory)).metadata
    derived, removed = GATE.hosted_gate_metadata(original)
    self.assertEqual(removed, ["prerequisite SiMa Platform MLA driver",
                               "prerequisite SiMa Platform PCIe service",
                               "verification", "setup"])
    self.assertEqual([item["type"] for item in derived["installation"]["prerequisites"]],
                     ["executable", "command-version"])
    restored = copy.deepcopy(derived)
    restored["installation"]["prerequisites"] = original["installation"]["prerequisites"]
    restored["installation"]["verification"] = original["installation"]["verification"]
    restored["installation"]["setup"] = original["installation"]["setup"]
    self.assertEqual(restored, original)
    self.assertIn("verification", original["installation"])

  def test_managed_environment_must_hold_exactly_the_bundle(self):
    inventory = {"python": {"distribution": "pyneatpcie", "version": VERSION, "wheelhouse": [
        {"distribution": "numpy", "version": "2.5.4"},
        {"distribution": "opencv-python", "version": "5.0.0.93"}]}}
    exact = [{"name": "pip", "version": "26.0"}, {"name": "pyneatpcie", "version": VERSION},
             {"name": "numpy", "version": "2.5.4"},
             {"name": "opencv_python", "version": "5.0.0.93"}]
    GATE.require_environment_packages(exact, inventory)
    for listed in (exact[1:], exact[:2] + [{"name": "numpy", "version": "2.6.0"}] + exact[3:],
                   exact + [{"name": "pillow", "version": "13.0.0"}]):
      with self.subTest(listed=listed), self.assertRaises(RuntimeError):
        GATE.require_environment_packages(listed, inventory)

  def test_local_channel_has_the_release_layout(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      offline = root / "offline"
      offline.mkdir()
      (offline / "neat-pcie-offline-installer-12345678.tar.gz").write_bytes(b"a")
      (offline / "metadata-offline.json").write_text(json.dumps({
          "version": VERSION, "commit": COMMIT,
          "resources": ["neat-pcie-offline-installer-12345678.tar.gz"]}))
      target = GATE.stage_channel(offline, root / "channel")
      self.assertEqual(target, f"core/pciehost/windows/amd64@v{VERSION}")
      self.assertEqual((root / "channel" / "core" / f"v{VERSION}" / "latest.tag").read_text(),
                       COMMIT[:12] + "\n")
      package = root / "channel" / PublishContracts.PREFIX
      self.assertEqual(sorted(path.name for path in package.iterdir()),
                       ["metadata-offline.json", "neat-pcie-offline-installer-12345678.tar.gz"])

  def test_only_the_blocked_profile_checks_for_a_newer_sima_cli(self):
    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory)
      ci = {"PIP_NO_INDEX": "1", "HTTPS_PROXY": "http://proxy.example:3128"}
      with mock.patch.dict(os.environ, ci):
        connected = GATE.customer_profile(work / "connected", work, blocked=False)
        target = GATE.customer_profile(work / "target", work, blocked=True)
    # With stdin closed, the update prompt aborts a connected run once PyPI has a newer
    # sima-cli than the pin; behind the block the check must fail and continue.
    self.assertEqual(connected["SIMA_CLI_CHECK_FOR_UPDATE"], "0")
    self.assertNotIn("SIMA_CLI_CHECK_FOR_UPDATE", target)
    for environment in (connected, target):
      self.assertFalse({"PIP_FIND_LINKS", "PIP_NO_INDEX"} & set(environment))
    self.assertNotIn("HTTPS_PROXY", connected)
    self.assertEqual(target["HTTPS_PROXY"], GATE.DEAD_PROXY)

  def test_block_covers_every_installer_process(self):
    programs = GATE.blocked_programs(Path("C:/py"), Path("C:/home"), Path("C:/Windows"))
    self.assertEqual(programs, [
        Path("C:/py/python.exe"), Path("C:/home/.sima-cli-env/Scripts/python.exe"),
        Path("C:/home/pyneatpcie/Scripts/python.exe"),
        Path("C:/Windows/System32/msiexec.exe"), Path("C:/Windows/SysWOW64/msiexec.exe")])
    source = (SCRIPT_DIR / "validate_windows_offline.py").read_text(encoding="utf-8")
    # The proof is a canary that must fail; a dead proxy alone is not a block.
    self.assertIn('raise RuntimeError("the firewall block is not effective', source)
    for name in ("PIP_FIND_LINKS", "PIP_NO_INDEX", "SIMA_CLI_CHECK_FOR_UPDATE"):
      self.assertIn(f'"{name}"', source)


class GateRun(unittest.TestCase):
  """validate() on Linux, with Windows, MSI, firewall and sima-cli replaced by fakes."""

  ARCHIVE = "neat-pcie-offline-installer-12345678.tar.gz"
  INSTALLERS = {"gstreamer-1.0-msvc-x86_64-1.28.6.exe": "4" * 64,
                "python-3.14.7-amd64.exe": "5" * 64}

  def setUp(self):
    self.root = Path(tempfile.mkdtemp(prefix="neat offline gate "))
    self.addCleanup(shutil.rmtree, self.root)
    self.offline = self.root / "offline"
    self.offline.mkdir()
    (self.offline / self.ARCHIVE).write_bytes(b"bundle")
    (self.offline / OFFLINE.OFFLINE_METADATA).write_text(json.dumps(
        {"version": VERSION, "commit": COMMIT, "resources": [self.ARCHIVE]}))
    (self.offline / OFFLINE.INVENTORY).write_text(json.dumps({
        "archive": {"name": self.ARCHIVE},
        "vendor_installers": [{"name": name, "sha256": digest}
                              for name, digest in self.INSTALLERS.items()]}))
    self.installed, self.canaries, self.unblocked = set(), [], set()
    self.blocked = []

  def run_logged(self, command, log, *, env=None, cwd=None, timeout=3600):
    argv = [str(value) for value in command]
    text = "ok\n"
    if "-t" in argv:  # the download: the extracted bundle and both installers
      folder = Path(argv[-1])
      tree = folder / self.ARCHIVE.removesuffix(".tar.gz") / OFFLINE.TOP
      tree.mkdir(parents=True)
      (tree / "metadata.json").write_text(json.dumps({"installation": {"prerequisites": []}}))
      for name in self.INSTALLERS:
        (folder / name).write_bytes(b"installer")
    elif argv[1:3] == ["install", "-m"]:
      if self.installed:
        text = "is already installed and verified\n"
      self.installed.update({"{core.msi}", "{scheduler.msi}"})
    elif argv[1:3] == ["packages", "remove"]:
      self.installed.clear()
    log.write_text(text)

  def powershell(self, script, log):
    if "New-NetFirewallRule" in script:
      self.blocked = [Path(path) for path in re.findall(r"-Program '([^']+)'", script)]
    return {"firewall-profiles-before.log": "Domain=True\nPrivate=True\nPublic=True\n",
            "python-installer-signature.log": "Valid\n"}.get(log.name, "")

  def reaches_internet(self, python, environment, log):
    self.canaries.append((Path(python), log.name))
    return log.name == "canary-before-block.log" or Path(python) in self.unblocked

  def receipts(self, home, name):
    return [{"state": "installed", "version": VERSION, "native_packages": [
        {"product_code": code} for code in sorted(self.installed)]}] if self.installed else []

  def validate(self):
    candidate = SimpleNamespace(metadata={"name": "pciehost", "version": VERSION},
                                core_msi="core.msi", scheduler_msi="scheduler.msi",
                                core_commit=COMMIT)
    args = SimpleNamespace(
        candidate=self.root / "candidate", upgrade_base_candidate=None,
        expected_upgrade_base_commit=None, offline=self.offline,
        evidence=self.root / "evidence", work=self.root / "work",
        sima_cli=self.root / "bootstrap" / "sima-cli.exe")
    fake_subprocess = SimpleNamespace(run=lambda *_, **__: SimpleNamespace(stdout="[]", returncode=0),
                                      SubprocessError=subprocess.SubprocessError)
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.dict(os.environ, {"SystemRoot": "C:/Windows",
                                                       "ProgramFiles": "C:/Program Files"}))
      for name, value in {
          "require_windows": lambda: None, "validate_candidate": lambda _: candidate,
          "load_upgrade_base": lambda *_: None, "msi_product_code": lambda msi: f"{{{msi}}}",
          "msi_product_state": lambda code: 5 if code in self.installed else -1,
          "run_logged": self.run_logged, "powershell": self.powershell,
          "reaches_internet": self.reaches_internet, "receipts": self.receipts,
          "sha256": lambda path: self.INSTALLERS[path.name],
          "require_environment_packages": lambda *_: None, "subprocess": fake_subprocess,
      }.items():
        stack.enter_context(mock.patch.object(GATE, name, value))
      stack.enter_context(mock.patch.object(GATE.windows_offline, "verify"))
      GATE.validate(args)

  def test_every_blocked_interpreter_is_proven_by_a_canary(self):
    self.validate()
    home = (self.root / "work" / "customer profile").resolve()
    managed = home / "pyneatpcie" / "Scripts" / "python.exe"
    self.assertEqual(self.canaries[-1], (managed, "canary-blocked-managed.log"))
    self.assertIn(managed, self.blocked)
    self.assertTrue((self.root / "evidence" / "windows-offline-gate.json").is_file())
    self.assertEqual(self.installed, set())

  def test_an_unblocked_managed_environment_fails_the_gate(self):
    # sima-cli created pyneatpcie from a CPython 3.14 that the rules do not cover.
    self.unblocked = {(self.root / "work" / "customer profile").resolve() / "pyneatpcie"
                      / "Scripts" / "python.exe"}
    with self.assertRaisesRegex(RuntimeError, "not effective for the managed environment"):
      self.validate()


def workflow_job(source, name):
  return re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", source)[1]


class WorkflowContracts(unittest.TestCase):
  def setUp(self):
    self.windows = WORKFLOW.read_text(encoding="utf-8")

  def test_offline_job_assembles_and_gates_beside_the_hardware_job(self):
    job = workflow_job(self.windows, "offline")
    self.assertIn("    runs-on: windows-2025\n", job)
    self.assertIn(
        "    needs: [native, linux_release, card_candidate, resolve-vulcan-config]\n", job)
    self.assertEqual(re.search(r"(?m)^    if: (.+)$", job)[1],
                     re.search(r"(?m)^    if: (.+)$", workflow_job(self.windows, "hardware"))[1])
    self.assertNotRegex(job, r"(?m)^    environment:")
    for text in ("windows_offline.py assemble", "validate_windows_offline.py",
                 "name: core-pciehost-windows-offline\n",
                 "name: core-pcie-card-qualification-candidate",
                 "name: core-pciehost-windows-card-inputs"):
      self.assertIn(text, job)
    # A release tag's card bundle carries the bare tag version, as in the hardware job.
    self.assertIn("            --ref $env:GITHUB_REF_NAME `\n", job)
    cleanup = job.split("      - name: Remove the network block\n", 1)[1]
    self.assertIn("        if: always()\n", cleanup)
    self.assertIn('Get-NetFirewallRule -DisplayName "neat-offline-gate"', cleanup)
    self.assertIn("firewall-profiles.json", cleanup)
    self.assertIn('RULE = "neat-offline-gate"',
                  (SCRIPT_DIR / "validate_windows_offline.py").read_text(encoding="utf-8"))

  def test_publish_offline_follows_publish_and_offline_only(self):
    job = workflow_job(self.windows, "publish-offline")
    self.assertIn("    needs: [publish, offline, resolve-vulcan-config]\n", job)
    self.assertIn("    if: ${{ needs.publish.result == 'success' && "
                  "needs.offline.result == 'success' }}\n", job)
    self.assertIn("cancel-in-progress: false", job)
    self.assertIn("windows_offline.py publish", job)
    self.assertIn("awscli==1.42.49", job)
    self.assertNotIn("publish_windows.sh", job)
    # The online publication is unchanged and does not wait for the offline bundle.
    self.assertIn("    needs: [hardware, linux_release, resolve-vulcan-config]\n",
                  workflow_job(self.windows, "publish"))

  def test_contract_job_runs_these_tests(self):
    self.assertIn("      - run: python3 pcie_host/tests/test_windows_offline.py\n",
                  workflow_job(self.windows, "contracts"))


class InputContracts(unittest.TestCase):
  def setUp(self):
    self.windows = WORKFLOW.read_text(encoding="utf-8")

  def test_cpython_installer_pin_is_the_version_ci_builds_with(self):
    manifest = json.loads((SCRIPT_DIR / "windows_build_inputs.json").read_text())
    pinned = [entry for entry in manifest["files"] if entry["use"] == "python-installer"]
    self.assertEqual(len(pinned), 1)
    version = re.fullmatch(r"python-([0-9.]+)-amd64[.]exe", pinned[0]["name"])[1]
    versions = re.findall(r"python-version: (\S+)", self.windows + MAIN_WORKFLOW.read_text(
        encoding="utf-8"))
    self.assertTrue(versions)
    self.assertEqual(set(versions), {version})

  def test_install_guide_uses_the_bundle_paths(self):
    guide = (OFFLINE.OFFLINE / "WINDOWS_INSTALL.md").read_text(encoding="utf-8")
    for text in (r"--find-links .\sima-cli --require-hashes -r .\sima-cli\requirements.txt",
                 r"install -m .\metadata.json", r".\card\neat-core-offline-installer.tar.gz",
                 r"Set-Location .\neat-pcie-offline-installer-*\neat-offline"):
      self.assertIn(text, guide)
    self.assertEqual(OFFLINE.SIMA_CLI_REQUIREMENTS, "sima-cli/requirements.txt")
    self.assertEqual(OFFLINE.CARD_RESOURCE, "card/neat-core-offline-installer.tar.gz")


if __name__ == "__main__":
  unittest.main()
