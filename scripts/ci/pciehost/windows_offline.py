#!/usr/bin/env python3
"""Assemble, verify and publish the Windows PCIe-host offline bundle.

The bundle is the qualified online release with what a host without internet
access needs to install it through sima-cli: the pinned Python dependency
wheels, the standalone Core card bundle of the same commit, and the pinned
sima-cli wheels. ``metadata-offline.json`` downloads it on any connected
computer, together with the exact GStreamer and CPython installers from their
vendors' URLs, as the Ubuntu offline bundles are downloaded.
"""

from __future__ import annotations

import argparse
import copy
from email.parser import BytesHeaderParser
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

from qualify_windows_hardware import CARD_BUNDLE_ARCHIVE, validate_card_bundle
from windows_candidate import sha256, validate_candidate, write_json
from windows_release import package_prefix, require_release_ref, s3_reader

ROOT = Path(__file__).resolve().parents[3]
OFFLINE = ROOT / "pcie_host" / "scripts" / "offline"
sys.path.insert(0, str(OFFLINE.parent))
sys.path.insert(0, str(OFFLINE))
import assemble  # noqa: E402
import ci  # noqa: E402
from create_windows_metadata import gstreamer_installer_url  # noqa: E402

TOP = "neat-offline"
OFFLINE_METADATA = assemble.METADATA_NAME
INVENTORY = "windows-offline-bundle.json"
CARD_RESOURCE = f"card/{CARD_BUNDLE_ARCHIVE}"
SIMA_CLI_REQUIREMENTS = "sima-cli/requirements.txt"
NAME = "gh:sima-neat/core/pciehost/windows/amd64/offline"
# pip resolves the closure for the Windows target, but evaluates environment markers for
# the interpreter that runs it: a closure with markers is assembled on the target.
PYTHON = sys.executable
INTERPRETER = ("import sys, sysconfig; print(sys.platform, sysconfig.get_platform(), "
               "sys.implementation.name, '%d.%d' % sys.version_info[:2])")
# A sima-cli release, or a build of sima-cli main: the channels WINDOWS_RELEASE.md pins.
PUBLISHABLE_SIMA_CLI = re.compile(r"[0-9]+[.][0-9]+[.][0-9]+(?:[+]main[.][0-9a-f]{12})?")


def _mebibytes(size: int) -> str:
  return f"{(size + 2**20 - 1) // 2**20}MB"


def _read_json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def pins(manifest: dict, use: str) -> dict[str, dict]:
  """The pinned files of one use, by file name."""
  return {entry["name"]: entry for entry in manifest["files"] if entry["use"] == use}


def vendor_installers(manifest: dict, metadata: dict) -> list[dict]:
  """The GStreamer and CPython installers the online metadata requires, as pinned."""
  installation = metadata["installation"]
  gstreamer = [item["version"] for item in installation.get("prerequisites", [])
               if item.get("type") == "command-version"
               and str(item.get("name", "")).startswith("GStreamer ")]
  selected = [*pins(manifest, "gstreamer").values(), *pins(manifest, "python-installer").values()]
  if len(gstreamer) != 1 or len(selected) != 2:
    raise ValueError("expected one GStreamer prerequisite and one GStreamer and one "
                     "python-installer pin")
  for entry in selected:
    if (not re.fullmatch(r"https://[^\s?#]+", entry["url"])
        or entry["url"].rsplit("/", 1)[1] != entry["name"]
        or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        or not isinstance(entry.get("size"), int) or entry["size"] <= 0):
      raise ValueError(
          f"vendor installer pin needs an HTTPS URL, SHA-256 and size: {entry['name']}")
  if selected[0]["url"] != gstreamer_installer_url(gstreamer[0]):
    raise ValueError(f"the GStreamer pin is not the GStreamer {gstreamer[0]} installer the "
                     "package requires")
  requires = installation["python"]["requires-python"]
  if not re.fullmatch(rf"python-{re.escape(requires)}[.][0-9]+-amd64[.]exe", selected[1]["name"]):
    raise ValueError(f"the python-installer pin is not a CPython {requires} x64 installer")
  return selected


def _wheel_metadata(wheel: Path):
  with zipfile.ZipFile(wheel) as archive:
    names = [name for name in archive.namelist()
             if re.fullmatch(r"[^/]+[.]dist-info/METADATA", name)]
    if len(names) != 1:
      raise ValueError(f"wheel has no unique METADATA: {wheel.name}")
    return BytesHeaderParser().parsebytes(archive.read(names[0]))


def wheel_identity(wheel: Path) -> tuple[str, str]:
  """Distribution name and version from the wheel's own METADATA."""
  headers = _wheel_metadata(wheel)
  name, version = headers.get("Name"), headers.get("Version")
  if not name or not version:
    raise ValueError(f"wheel METADATA lacks Name or Version: {wheel.name}")
  return name.strip(), version.strip()


def resolve_closure(wheel: Path, wheelhouse: Path, python_package: dict) -> list[Path]:
  """The dependency wheels pip selects for the target from the pinned files only."""
  python_version = python_package["requires-python"]
  if python_package.get("arch") != "amd64" or not re.fullmatch(r"3[.][0-9]+", python_version):
    raise ValueError("the wheel must target CPython 3.x amd64")
  with tempfile.TemporaryDirectory(prefix="neat-offline-closure-") as directory:
    # The target is the customer's interpreter, not the one assembling the bundle.
    subprocess.run([
        PYTHON, "-m", "pip", "download", "--isolated", "--disable-pip-version-check",
        "--no-index", "--no-cache-dir", "--only-binary", ":all:",
        "--platform", "win_amd64", "--implementation", "cp",
        "--python-version", python_version, "--abi", "cp" + python_version.replace(".", ""),
        "--find-links", str(wheelhouse), "--dest", directory, str(wheel)], check=True)
    names = sorted(path.name for path in Path(directory).iterdir() if path.name != wheel.name)
  closure = [wheelhouse / name for name in names]
  conditional = [f"{path.name}: {requirement}" for path in (wheel, *closure)
                 for requirement in _wheel_metadata(path).get_all("Requires-Dist", [])
                 if ";" in requirement]
  if conditional:
    host = subprocess.run([PYTHON, "-I", "-c", INTERPRETER], capture_output=True, text=True,
                          check=True).stdout.split()
    if host != ["win32", "win-amd64", "cpython", python_version]:
      raise ValueError(
          f"{conditional[0]} has environment markers, which pip evaluates for the assembling "
          f"interpreter ({' '.join(host)}); assemble with CPython {python_version} x64 on Windows")
  return closure


def require_pinned(paths: list[Path], pinned: dict[str, dict], recorded: dict[str, str],
                   description: str) -> None:
  """Every file must be pinned and recorded by the build with these bytes."""
  for path in paths:
    entry = pinned.get(path.name)
    if entry is None:
      raise ValueError(f"{description} {path.name} is not pinned in windows_build_inputs.json")
    digest = sha256(path)
    if digest != entry["sha256"]:
      raise ValueError(f"{description} {path.name} differs from its pinned SHA-256")
    if recorded.get(path.name) != digest:
      raise ValueError(f"{description} {path.name} differs from the build's wheelhouse record")


def verify(tree: Path, candidate=None) -> dict:
  """The tree holds exactly metadata.json and its resources, each with its checksum."""
  metadata = _read_json(tree / "metadata.json")
  resources, checksums = metadata.get("resources"), metadata.get("resources-checksum")
  if (not isinstance(resources, list) or len(set(resources)) != len(resources)
      or not isinstance(checksums, dict) or set(resources) != set(checksums)):
    raise ValueError("offline metadata resources and checksums differ")
  for name in resources:
    parts = name.split("/")
    if "\\" in name or name.startswith("/") or any(part in ("", ".", "..") for part in parts):
      raise ValueError(f"unsafe offline resource path: {name!r}")
  if any(path.is_symlink() for path in tree.rglob("*")):
    raise ValueError("offline bundle symlinks are forbidden")
  files = {path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_file()}
  if files != {"metadata.json", *resources}:
    raise ValueError(f"offline bundle files differ from metadata.json: missing "
                     f"{sorted({*resources} - files)}, unexpected "
                     f"{sorted(files - {'metadata.json', *resources})}")
  for name in resources:
    if sha256(tree / name) != checksums[name]:
      raise ValueError(f"offline bundle checksum differs: {name}")
  if candidate is not None:
    for name, digest in candidate.resources.items():
      if checksums.get(name) != digest:
        raise ValueError(f"offline bundle {name} is not the candidate's file")
  return metadata


def assemble_bundle(candidate_dir: Path, wheelhouse: Path, card_candidate: Path,
                    inputs: Path, installer_revision: str, run_id: int, ref: str,
                    output: Path) -> Path:
  if not re.fullmatch(r"[0-9a-f]{40}", installer_revision):
    raise ValueError("the installer revision must be a full lowercase commit")
  candidate = validate_candidate(candidate_dir)
  manifest = _read_json(inputs)
  vendors = vendor_installers(manifest, candidate.metadata)
  python_package = candidate.metadata["installation"]["python"]
  recorded = {item["name"]: item["sha256"] for item in candidate.receipt["python"]["wheelhouse"]}

  closure = resolve_closure(candidate.wheel, wheelhouse, python_package)
  require_pinned(closure, pins(manifest, "wheelhouse"), recorded, "Python dependency")
  sima_cli = [wheelhouse / name for name in sorted(pins(manifest, "sima-cli"))]
  for path in sima_cli:
    if not path.is_file():
      raise ValueError(f"the build's wheelhouse lacks pinned sima-cli file {path.name}")
  require_pinned(sima_cli, pins(manifest, "sima-cli"), recorded, "sima-cli")
  card = validate_card_bundle(card_candidate, candidate.core_commit, ref)

  output.mkdir(parents=True, exist_ok=False)
  tree = output / TOP
  (tree / "wheelhouse").mkdir(parents=True)
  (tree / "card").mkdir()
  (tree / "sima-cli").mkdir()
  for name in candidate.metadata["resources"]:
    shutil.copyfile(candidate.root / name, tree / name)
  for path in closure:
    shutil.copyfile(path, tree / "wheelhouse" / path.name)
  shutil.copyfile(card.archive, tree / CARD_RESOURCE)
  shutil.copyfile(OFFLINE / "WINDOWS_INSTALL.md", tree / "INSTALL.md")
  requirements, distributions = [], set()
  for path in sima_cli:
    shutil.copyfile(path, tree / "sima-cli" / path.name)
    name, version = wheel_identity(path)
    if name.lower() in distributions:
      raise ValueError(f"sima-cli pins two wheels of {name}")
    distributions.add(name.lower())
    requirements.append(f"{name}=={version} --hash=sha256:{sha256(path)}\n")
  (tree / SIMA_CLI_REQUIREMENTS).write_text("".join(requirements), encoding="utf-8")

  metadata = copy.deepcopy(candidate.metadata)
  added = [*(f"wheelhouse/{path.name}" for path in closure), CARD_RESOURCE, "INSTALL.md",
           SIMA_CLI_REQUIREMENTS, *(f"sima-cli/{path.name}" for path in sima_cli)]
  metadata["resources"] += added
  metadata["resources-checksum"].update({name: sha256(tree / name) for name in added})
  wheelhouse_inventory = []
  for path in closure:
    name, version = wheel_identity(path)
    wheelhouse_inventory.append(
        {"wheel": f"wheelhouse/{path.name}", "distribution": name, "version": version})
  metadata["installation"]["python"]["wheelhouse"] = wheelhouse_inventory
  write_json(tree / "metadata.json", metadata)
  verify(tree, candidate)

  built = assemble.archive_bundle(tree, assemble.ARCHIVE_NAME, TOP)
  digest = sha256(built)
  archive = built.rename(output / ci.resource_name(built.name, digest))
  tree_size = sum(path.stat().st_size for path in tree.rglob("*") if path.is_file())
  download_size = archive.stat().st_size + sum(entry["size"] for entry in vendors)
  source = candidate.metadata
  offline = {
      "name": NAME,
      "version": source["version"],
      "release": source.get("release", ""),
      "description": ("Offline Neat PCIe host installer for Windows 11 x64 hosts and "
                      "Platform 3.0 cards"),
      # Downloadable on any connected computer; metadata.json checks the Windows host.
      "platforms": [],
      "resources": [archive.name, *(entry["url"] for entry in vendors)],
      "resources-checksum": {archive.name: digest,
                             **{entry["url"]: entry["sha256"] for entry in vendors}},
      "selectable-resources": [],
      # sima-cli extracts the archive into a folder named after it.
      "installation": {"script": "", "post-message": (
          "Offline installer downloaded. Copy this folder to the Windows PCIe host and "
          f"follow {archive.name.removesuffix('.tar.gz')}\\{TOP}\\INSTALL.md.")},
      "artifact": {"type": "offline", "repository": "core",
                   "package_path": "pciehost/windows/amd64", "distro": "windows",
                   "arches": ["amd64"]},
      **{key: source[key] for key in ("repository", "branch", "commit", "published_at_utc")
         if key in source},
      # Everything the download command leaves: the archive, its extraction and both installers.
      "size": {"download": _mebibytes(download_size),
               "install": _mebibytes(download_size + tree_size)},
      "source_metadata_sha256": {"host": sha256(candidate.root / "metadata.json"),
                                 "card": sha256(card_candidate / "metadata.json")},
      "installer_revision": installer_revision,
      "source_run_id": run_id,
  }
  (output / OFFLINE_METADATA).write_text(json.dumps(offline, indent=2) + "\n", encoding="utf-8")
  write_json(output / INVENTORY, {
      "schema_version": 1,
      "core_commit": candidate.core_commit,
      "version": source["version"],
      "archive": {"name": archive.name, "sha256": digest},
      "online_resources": candidate.resources,
      "python": {"distribution": python_package["distribution"],
                 "version": python_package["version"],
                 "wheelhouse": [{**item, "sha256": metadata["resources-checksum"][item["wheel"]]}
                                for item in wheelhouse_inventory]},
      "card": {"resource": CARD_RESOURCE, "sha256": card.sha256},
      "sima_cli": {"version": manifest["sima_cli_version"], "files": len(sima_cli)},
      "vendor_installers": [{key: entry[key] for key in ("name", "url", "sha256", "size")}
                            for entry in vendors],
  })
  return archive


class S3Storage:
  """The release bucket, written create-only as the Ubuntu offline publisher does."""

  def __init__(self, bucket: str, kms: str, distribution: str):
    self.bucket, self.kms, self.distribution = bucket, kms, distribution
    self.read = s3_reader(bucket)

  def has_object(self, key: str, digest: str) -> bool:
    try:
      ci.verify_uploaded(self.bucket, key, digest)
    except (subprocess.CalledProcessError, ValueError):
      return False
    return True

  def put_resource(self, key: str, path: Path) -> None:
    ci.put_resource(self.bucket, key, path, self.kms)

  def put(self, key: str, path: Path) -> None:
    ci.put(self.bucket, key, path, self.kms)

  def invalidate(self, path: str) -> None:
    if self.distribution:
      ci.run("aws", "cloudfront", "create-invalidation", "--distribution-id",
             self.distribution, "--paths", path)


def publish(storage, ref: str, commit: str, offline_dir: Path) -> list[str]:
  """Publish beside the live online release with the same bytes; never overwrite.

  Returns the keys written: none when this bundle is already published.
  """
  document = (offline_dir / OFFLINE_METADATA).read_bytes()
  offline = json.loads(document)
  inventory = _read_json(offline_dir / INVENTORY)
  require_release_ref("tag", ref, offline["version"])
  sima_cli = inventory["sima_cli"]["version"]
  if not PUBLISHABLE_SIMA_CLI.fullmatch(sima_cli):
    raise ValueError(f"the bundle carries sima-cli {sima_cli}; a release bundles a sima-cli "
                     "release X.Y.Z or a main build X.Y.Z+main.<commit12>, never another branch")
  if offline.get("commit") != commit:
    raise ValueError(f"offline bundle names commit {offline.get('commit')}, not {commit}")
  archive_name = offline["resources"][0]
  archive = offline_dir / archive_name
  if sha256(archive) != offline["resources-checksum"][archive_name]:
    raise ValueError(f"{archive_name} differs from {OFFLINE_METADATA}")
  prefix = package_prefix(ref, commit)
  live = storage.read(f"{prefix}/metadata.json")
  if (hashlib.sha256(live).hexdigest() != offline["source_metadata_sha256"]["host"]
      or json.loads(live).get("resources-checksum") != inventory["online_resources"]):
    raise ValueError(f"{prefix}/metadata.json is not the online release this offline bundle "
                     "was assembled from")
  try:
    published = storage.read(f"{prefix}/{OFFLINE_METADATA}")
  except FileNotFoundError:
    published = None
  if published is not None:
    if (published != document
        or not storage.has_object(f"{prefix}/{archive_name}", sha256(archive))):
      raise ValueError(f"{prefix}/{OFFLINE_METADATA} is already published with other bytes; "
                       "a published release is never overwritten")
    return []
  storage.put_resource(f"{prefix}/{archive_name}", archive)
  storage.put(f"{prefix}/{OFFLINE_METADATA}", offline_dir / OFFLINE_METADATA)
  storage.invalidate(f"/{prefix}/{OFFLINE_METADATA}")
  return [f"{prefix}/{archive_name}", f"{prefix}/{OFFLINE_METADATA}"]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  build = commands.add_parser("assemble", help="assemble the bundle from the run's candidate")
  build.add_argument("--candidate", type=Path, required=True)
  build.add_argument("--wheelhouse", type=Path, required=True)
  build.add_argument("--card-candidate", type=Path, required=True)
  build.add_argument("--inputs", type=Path, required=True)
  build.add_argument("--installer-revision", required=True)
  build.add_argument("--run-id", type=int, required=True)
  build.add_argument("--ref", required=True, help="Git ref name of the selected Core commit")
  build.add_argument("--output", type=Path, required=True)
  check = commands.add_parser("verify", help="verify an extracted neat-offline folder")
  check.add_argument("tree", type=Path)
  check.add_argument("--candidate", type=Path)
  release = commands.add_parser("publish", help="publish beside the live online release")
  release.add_argument("--bucket", required=True)
  release.add_argument("--ref", required=True)
  release.add_argument("--commit", required=True)
  release.add_argument("--offline", type=Path, required=True)
  release.add_argument("--kms", default="")
  release.add_argument("--distribution", default="")
  args = parser.parse_args()
  if args.command == "assemble":
    archive = assemble_bundle(args.candidate, args.wheelhouse.resolve(strict=True),
                              args.card_candidate.resolve(strict=True), args.inputs,
                              args.installer_revision, args.run_id, args.ref,
                              args.output.resolve())
    print(f"Assembled {archive}")
  elif args.command == "verify":
    verify(args.tree.resolve(strict=True),
           validate_candidate(args.candidate) if args.candidate else None)
    print(f"Verified {args.tree}")
  else:
    keys = publish(S3Storage(args.bucket, args.kms, args.distribution), args.ref, args.commit,
                   args.offline)
    print("\n".join(f"Published s3://{args.bucket}/{key}" for key in keys)
          or "The offline bundle is already published with these bytes; nothing to publish")


if __name__ == "__main__":
  try:
    main()
  except (OSError, ValueError, subprocess.CalledProcessError) as error:
    print(f"error: {error}", file=sys.stderr)
    sys.exit(1)
