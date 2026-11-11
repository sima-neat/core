#!/usr/bin/env python3
"""Acquire the immutable Windows host dependencies selected by Core policy."""

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import zipfile


CORE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CORE / "tools" / "ci"))
from published_artifact import acquire  # noqa: E402
from scheduler_dependency import acquire_scheduler  # noqa: E402


def copy_candidate(source: Path, destination: Path) -> None:
  """Stage a trusted Actions artifact without accepting links or nested paths."""
  source = source.resolve(strict=True)
  files = sorted(source.iterdir())
  if not files:
    raise ValueError(f"dependency candidate is empty: {source}")
  destination.mkdir(parents=True, exist_ok=False)
  for path in files:
    if path.is_symlink() or not path.is_file():
      raise ValueError(f"dependency candidate contains a non-file: {path.name}")
    shutil.copy2(path, destination / path.name)


def extract_sima_cli_wheel(publication: Path, destination: Path) -> Path:
  """Extract the one CLI wheel after the publication transport verified its ZIP."""
  packages = sorted(publication.glob("sima-cli-package-*.zip"))
  if len(packages) != 1:
    raise ValueError(f"expected one published sima-cli package, found {len(packages)}")
  destination.mkdir(parents=True, exist_ok=False)
  with zipfile.ZipFile(packages[0]) as archive:
    if archive.testzip() is not None:
      raise ValueError("published sima-cli package failed CRC validation")
    wheels = [entry for entry in archive.infolist()
              if entry.filename.endswith(".whl") and not entry.is_dir()]
    if len(wheels) != 1:
      raise ValueError(f"expected one sima-cli wheel in package, found {len(wheels)}")
    wheel = wheels[0]
    if (Path(wheel.filename).name != wheel.filename
        or stat.S_IFMT(wheel.external_attr >> 16) == stat.S_IFLNK
        or wheel.file_size > 32 * 1024 * 1024):
      raise ValueError("published sima-cli package contains an unsafe wheel")
    output = destination / wheel.filename
    with archive.open(wheel) as source, output.open("xb") as target:
      shutil.copyfileobj(source, target)
  return output


def write_github_outputs(values: dict[str, str]) -> None:
  output = os.environ.get("GITHUB_OUTPUT")
  if not output:
    return
  with Path(output).open("a", encoding="utf-8") as stream:
    for name, value in values.items():
      stream.write(f"{name}={value}\n")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--core", type=Path, default=CORE)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--cli", default="sima-cli")
  parser.add_argument("--environment", default="production")
  parser.add_argument("--branch", required=True)
  parser.add_argument("--scheduler-candidate", type=Path)
  parser.add_argument("--scheduler-commit")
  parser.add_argument("--sima-cli-candidate", type=Path)
  parser.add_argument("--sima-cli-commit")
  parser.add_argument("--sima-cli-branch", default="main")
  args = parser.parse_args()

  for candidate, commit, label in (
      (args.scheduler_candidate, args.scheduler_commit, "Scheduler"),
      (args.sima_cli_candidate, args.sima_cli_commit, "sima-cli")):
    if bool(candidate) != bool(commit):
      parser.error(f"--{label.lower()}-candidate and --{label.lower()}-commit must be used together")
    if commit and (len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit)):
      parser.error(f"{label} commit must be a full lowercase Git commit")

  root = args.output.resolve()
  root.mkdir(parents=True, exist_ok=False)
  scheduler = root / "scheduler"
  if args.scheduler_candidate:
    copy_candidate(args.scheduler_candidate, scheduler)
    scheduler_record = {
        "branch": "actions-artifact",
        "commit": args.scheduler_commit,
        "selector": f"actions:{args.scheduler_commit}",
    }
  else:
    scheduler_record = acquire_scheduler(
        args.cli, args.core.resolve(), args.branch, None, args.environment, scheduler,
        package_path="pcie-host-windows-amd64")

  cli = root / "sima-cli"
  if args.sima_cli_candidate:
    copy_candidate(args.sima_cli_candidate, cli)
    cli_record = {
        "branch": "actions-artifact",
        "commit": args.sima_cli_commit,
        "selector": f"actions:{args.sima_cli_commit}",
    }
  else:
    cli_publication = root / "sima-cli-publication"
    cli_record = acquire(
        args.cli, "sima-cli", args.sima_cli_branch, "latest", args.environment,
        cli_publication)
    extract_sima_cli_wheel(cli_publication, cli)

  selection = {
      "schema_version": 1,
      "environment": args.environment,
      "scheduler": {
          "branch": scheduler_record["branch"],
          "commit": scheduler_record["commit"],
          "selector": scheduler_record["selector"],
      },
      "sima_cli": {
          "branch": cli_record["branch"],
          "commit": cli_record["commit"],
          "selector": cli_record["selector"],
      },
  }
  (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
  write_github_outputs({
      "scheduler_commit": selection["scheduler"]["commit"],
      "sima_cli_commit": selection["sima_cli"]["commit"],
  })
  print(json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
  main()
