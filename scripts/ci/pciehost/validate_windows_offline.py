#!/usr/bin/env python3
"""Install the Windows offline bundle as a customer does, with the network blocked.

Phase 1 is the connected computer: the documented download command fetches the
bundle from a local copy of the release channel and both installers from their
vendors. Phase 2 is the target: Windows Firewall blocks every installer
process, a canary proves the block, and the bundle's sima-cli installs, repairs
and removes the release from the copied folder. Linux proves the same with
network-less containers; Windows has no per-process network namespace.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from build_windows import msi_product_code, msi_product_state
from qualify_windows_hardware import (
    isolated_sima_cli_environment, load_upgrade_base, local_registry, require_fresh_directory,
    require_powershell, require_windows, run_logged)
from windows_candidate import sha256, validate_candidate, write_json
from windows_release import (
    customer_target, latest_tag_key, package_prefix, release_tag, short_commit)
import windows_offline


RULE = "neat-offline-gate"
INSTALLSTATE_DEFAULT = 5
CANARY = ("import urllib.request; "
          "urllib.request.urlopen('https://pypi.org/simple/pip/', timeout=15)")
DEAD_PROXY = "http://127.0.0.1:9"
# A hosted runner has no Platform package; these need it and touch no network.
PLATFORM_PREREQUISITES = {"windows-pnp-driver", "windows-service"}


def hosted_gate_metadata(metadata: dict) -> tuple[dict, list[str]]:
  """The install-side metadata without the steps that need the Platform package."""
  derived = copy.deepcopy(metadata)
  installation = derived["installation"]
  removed = [f"prerequisite {item['name']}" for item in installation["prerequisites"]
             if item.get("type") in PLATFORM_PREREQUISITES]
  installation["prerequisites"] = [item for item in installation["prerequisites"]
                                   if item.get("type") not in PLATFORM_PREREQUISITES]
  for step in ("verification", "setup"):
    if installation.pop(step, None) is not None:
      removed.append(step)
  return derived, removed


def _canonical(name: str) -> str:
  return re.sub(r"[-_.]+", "-", name).lower()


def require_environment_packages(listed: list[dict], inventory: dict) -> None:
  """The managed environment holds pip, the wheel and exactly the bundle's dependencies."""
  python = inventory["python"]
  expected = {_canonical(python["distribution"]): python["version"],
              **{_canonical(item["distribution"]): item["version"]
                 for item in python["wheelhouse"]}}
  found = {_canonical(item["name"]): item["version"] for item in listed}
  if found.pop("pip", None) is None or found != expected:
    raise RuntimeError(f"managed environment holds {found}, expected pip and {expected}")


def stage_channel(offline_dir: Path, root: Path) -> str:
  """Lay out metadata-offline.json and its archive as the release channel does."""
  offline = json.loads((offline_dir / windows_offline.OFFLINE_METADATA).read_text(
      encoding="utf-8"))
  tag = release_tag(offline["version"])
  package = root / package_prefix(tag, offline["commit"])
  package.mkdir(parents=True)
  shutil.copyfile(offline_dir / windows_offline.OFFLINE_METADATA,
                  package / windows_offline.OFFLINE_METADATA)
  shutil.copyfile(offline_dir / offline["resources"][0], package / offline["resources"][0])
  (root / latest_tag_key(tag)).write_text(short_commit(offline["commit"]) + "\n",
                                           encoding="ascii")
  return customer_target(tag)


def blocked_programs(base_prefix: Path, home: Path, system_root: Path) -> list[Path]:
  """The processes phase 2 starts that could open a connection."""
  return [base_prefix / "python.exe",
          home / ".sima-cli-env" / "Scripts" / "python.exe",
          home / "pyneatpcie" / "Scripts" / "python.exe",
          system_root / "System32" / "msiexec.exe",
          system_root / "SysWOW64" / "msiexec.exe"]


def customer_profile(home: Path, work: Path, *, blocked: bool) -> dict[str, str]:
  """A fresh customer profile: none of the CI pip or proxy settings.

  Behind the block sima-cli also looks for a newer release, as a customer's does, and
  must continue without the network. A connected run keeps the check off: with closed
  stdin its update prompt aborts once PyPI has a newer sima-cli than the pin.
  """
  home.mkdir(parents=True, exist_ok=True)
  # The wheelhouse argument only feeds PIP_FIND_LINKS, which is removed below.
  environment = isolated_sima_cli_environment(home, work, work)
  removed = {"PIP_FIND_LINKS", "PIP_NO_INDEX", "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
             "ALL_PROXY"} | ({"SIMA_CLI_CHECK_FOR_UPDATE"} if blocked else set())
  for name in list(environment):
    if name.upper() in removed:
      environment.pop(name)
  if blocked:
    environment.update({"HTTP_PROXY": DEAD_PROXY, "HTTPS_PROXY": DEAD_PROXY,
                        "ALL_PROXY": DEAD_PROXY})
  return environment


def powershell(script: str, log: Path) -> str:
  run_logged([require_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
              script], log, timeout=600)
  return log.read_text(encoding="utf-8", errors="replace").split("\n", 1)[1]


def _quoted(path: Path) -> str:
  return "'" + str(path).replace("'", "''") + "'"


def reaches_internet(python: Path, environment: dict[str, str], log: Path) -> bool:
  result = subprocess.run([python, "-c", CANARY], env=environment, capture_output=True,
                          text=True, timeout=120, check=False)
  log.write_text(f"+ {python} -c {CANARY}\nexit code: {result.returncode}\n"
                 f"{result.stdout}{result.stderr}", encoding="utf-8")
  return result.returncode == 0


def receipts(home: Path, name: str) -> list[dict]:
  path = home / ".sima-cli" / "registry.json"
  data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
  return [entry for entry in data if entry.get("package_name") == name]


def validate(args: argparse.Namespace) -> None:
  require_windows()
  candidate = validate_candidate(args.candidate)
  upgrade_base = load_upgrade_base(
      candidate, args.upgrade_base_candidate, args.expected_upgrade_base_commit)
  offline_dir = args.offline.resolve(strict=True)
  inventory = json.loads((offline_dir / windows_offline.INVENTORY).read_text(encoding="utf-8"))
  evidence = require_fresh_directory(args.evidence, "evidence directory")
  work = require_fresh_directory(args.work, "offline gate work directory")
  system_root = Path(os.environ["SystemRoot"])
  base_python = Path(sys.base_prefix) / "python.exe"
  name, version = candidate.metadata["name"], candidate.metadata["version"]
  products = [msi_product_code(candidate.core_msi), msi_product_code(candidate.scheduler_msi)]
  base_products = ([msi_product_code(upgrade_base.core_msi),
                    msi_product_code(upgrade_base.scheduler_msi)] if upgrade_base else [])
  if any(msi_product_state(code) == INSTALLSTATE_DEFAULT for code in products + base_products):
    raise RuntimeError("refusing to run on a host that has the release installed")
  home = work / "customer profile"
  result = {"candidate": candidate.core_commit, "inventory": inventory}

  profiles, failure = {}, None
  try:
    # Phase 1, the connected computer: the documented download command.
    channel = work / "release channel"
    target = stage_channel(offline_dir, channel)
    download = work / "connected computer with spaces" / "pcie-offline"
    connected = customer_profile(work / "connected computer profile", work, blocked=False)
    download.mkdir(parents=True)
    with local_registry(channel) as url:
      run_logged([args.sima_cli, "neat", "--base-url", url, "install", target, "-t", "offline",
                  "-d", download], evidence / "download.log", env=connected, cwd=work,
                 timeout=3600)
    archive = inventory["archive"]["name"]
    windows_offline.verify(download / archive.removesuffix(".tar.gz") / windows_offline.TOP,
                           candidate)
    installers = {item["name"]: download / item["name"] for item in inventory["vendor_installers"]}
    for item in inventory["vendor_installers"]:
      if sha256(installers[item["name"]]) != item["sha256"]:
        raise RuntimeError(f"downloaded {item['name']} differs from its pin")
    result["download"] = {"command": f"sima-cli neat install {target} -t offline -d <folder>",
                          "files": sorted(path.name for path in download.iterdir())}

    # A3: the customer copies the complete folder to the target.
    folder = work / "target with spaces" / "pcie-offline"
    shutil.copytree(download, folder)
    tree = folder / archive.removesuffix(".tar.gz") / windows_offline.TOP
    gstreamer, python_installer = (folder / item["name"] for item in inventory["vendor_installers"])
    powershell(
        f"$p = Start-Process -FilePath {_quoted(gstreamer)} -Wait -PassThru -ArgumentList "
        "@('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', '/ALLUSERS', '/TYPE=devel', "
        f"'/DIR=\"{Path(os.environ['ProgramFiles']) / 'gstreamer' / '1.0' / 'msvc_x86_64'}\"', "
        f"'/LOG=\"{evidence / 'gstreamer-setup.log'}\"'); "
        "if ($p.ExitCode -ne 0) { throw \"GStreamer installation failed: $($p.ExitCode)\" }",
        evidence / "gstreamer-install.log")
    # The runner already has this CPython from setup-python; its installer is checked, not run.
    status = powershell(
        f"(Get-AuthenticodeSignature -LiteralPath {_quoted(python_installer)}).Status",
        evidence / "python-installer-signature.log").strip()
    if status != "Valid":
      raise RuntimeError(f"the CPython installer's Authenticode status is {status!r}")

    if upgrade_base is not None:
      base_metadata, _ = hosted_gate_metadata(upgrade_base.metadata)
      write_json(upgrade_base.root / "metadata-hosted-gate.json", base_metadata)
      run_logged([args.sima_cli, "install", "-m", upgrade_base.root / "metadata-hosted-gate.json",
                  "--native-only"], evidence / "upgrade-base-install.log",
                 env=customer_profile(home, work, blocked=False), cwd=work, timeout=3600)

    offline = customer_profile(home, work, blocked=True)
    if not reaches_internet(base_python, customer_profile(home, work, blocked=False),
                            evidence / "canary-before-block.log"):
      raise RuntimeError("the positive control could not reach the internet; the block "
                         "cannot be proven")
    for line in powershell(
        "Get-NetFirewallProfile | ForEach-Object { \"$($_.Name)=$($_.Enabled)\" }",
        evidence / "firewall-profiles-before.log").split():
      profile, _, enabled = line.partition("=")
      profiles[profile] = enabled
    # The workflow's always() step restores this state if the job is cancelled.
    write_json(evidence / "firewall-profiles.json", profiles)
    rules = "; ".join(
        f"New-NetFirewallRule -DisplayName '{RULE}' -Direction Outbound -Action Block "
        f"-Profile Any -Program {_quoted(program)} | Out-Null"
        for program in blocked_programs(Path(sys.base_prefix), home, system_root))
    powershell("Set-NetFirewallProfile -Profile Domain,Private,Public -Enabled True; " + rules,
               evidence / "firewall-block.log")
    # A proxy alone is not a block: sima-cli's artifact downloads ignore it.
    if reaches_internet(base_python, customer_profile(home, work, blocked=False),
                        evidence / "canary-blocked.log"):
      raise RuntimeError("the firewall block is not effective: the canary reached the internet")

    # Phase 2, the target: B4, then B6 with the hosted-runner metadata.
    cli_python = home / ".sima-cli-env" / "Scripts" / "python.exe"
    cli = cli_python.with_name("sima-cli.exe")
    run_logged([sys.executable, "-m", "venv", cli_python.parents[1]],
               evidence / "sima-cli-venv.log", env=offline, cwd=tree)
    if reaches_internet(cli_python, customer_profile(home, work, blocked=False),
                        evidence / "canary-blocked-venv.log"):
      raise RuntimeError("the firewall block is not effective for a virtual environment")
    run_logged([cli_python, "-m", "pip", "install", "--no-index", "--find-links", r".\sima-cli",
                "--require-hashes", "-r", r".\sima-cli\requirements.txt"],
               evidence / "sima-cli-install.log", env=offline, cwd=tree, timeout=900)
    metadata, removed = hosted_gate_metadata(json.loads((tree / "metadata.json").read_text(
        encoding="utf-8")))
    write_json(tree / "metadata-hosted-gate.json", metadata)
    result["hosted_runner_removed_steps"] = removed
    install = [cli, "install", "-m", tree / "metadata-hosted-gate.json"]
    run_logged(install, evidence / "install.log", env=offline, cwd=tree, timeout=3600)
    installed = receipts(home, name)
    if ([entry.get("state") for entry in installed] != ["installed"]
        or installed[0].get("version") != version
        or sorted(item.get("product_code") for item in installed[0].get("native_packages", []))
        != sorted(products)):
      raise RuntimeError(f"expected one installed receipt of {name} {version} with both products")
    if any(msi_product_state(code) != INSTALLSTATE_DEFAULT for code in products):
      raise RuntimeError("both MSI products must be installed")
    managed = home / "pyneatpcie" / "Scripts" / "python.exe"
    # pip ran here; sima-cli may have created it from another CPython 3.14 than ours.
    if reaches_internet(managed, customer_profile(home, work, blocked=False),
                        evidence / "canary-blocked-managed.log"):
      raise RuntimeError("the firewall block is not effective for the managed environment")
    listing = subprocess.run(
        [managed, "-m", "pip", "list", "--format=json", "--isolated",
         "--disable-pip-version-check"], env=offline, capture_output=True, text=True,
        timeout=300, check=True).stdout
    (evidence / "pip-list.json").write_text(listing, encoding="utf-8")
    require_environment_packages(json.loads(listing), inventory)
    run_logged([managed, "-m", "pip", "check", "--isolated", "--disable-pip-version-check"],
               evidence / "pip-check.log", env=offline, timeout=300)
    run_logged([managed, "-c", "import pyneatpcie"], evidence / "import.log", env=offline,
               timeout=120)
    if upgrade_base is not None:
      if any(msi_product_state(code) == INSTALLSTATE_DEFAULT for code in base_products):
        raise RuntimeError("the upgrade left the baseline products installed")
      if "Replaced the registry receipt" not in (evidence / "install.log").read_text(
          encoding="utf-8", errors="replace"):
        raise RuntimeError("the upgrade did not replace the baseline's receipt")

    run_logged(install, evidence / "repeat-install.log", env=offline, cwd=tree, timeout=3600)
    if "is already installed and verified" not in (evidence / "repeat-install.log").read_text(
        encoding="utf-8", errors="replace"):
      raise RuntimeError("the repeated install did not keep the verified wheel")

    run_logged([cli, "packages", "remove", name, "--version", version, "--yes"],
               evidence / "remove.log", env=offline, cwd=tree, timeout=1800)
    if receipts(home, name) or any(
        msi_product_state(code) == INSTALLSTATE_DEFAULT for code in products):
      raise RuntimeError("removal left a receipt or an installed product")
    result.update(install="PASS", repeat="PASS", remove="PASS",
                  upgrade="PASS" if upgrade_base else "not run: no upgrade baseline")
    write_json(evidence / "windows-offline-gate.json", result)
  except BaseException as error:
    failure = error
    raise
  finally:
    errors = restore(profiles, products + base_products, evidence)
    # A cleanup error fails a passing gate; it never hides the gate's own failure.
    if errors and failure is None:
      raise RuntimeError("offline gate cleanup failed: " + "; ".join(errors))


def restore(profiles: dict[str, str], products: list[str], evidence: Path) -> list[str]:
  """Remove the block, restore the firewall profiles and uninstall leftover products."""
  errors = []
  cleanup = [f"Get-NetFirewallRule -DisplayName '{RULE}' -ErrorAction SilentlyContinue | "
             "Remove-NetFirewallRule"]
  cleanup += [f"Set-NetFirewallProfile -Name {profile} -Enabled {enabled}"
              for profile, enabled in profiles.items()]
  try:
    powershell("; ".join(cleanup), evidence / "firewall-restore.log")
  except (OSError, RuntimeError, subprocess.SubprocessError) as error:
    errors.append(str(error))
  for code in products:
    if msi_product_state(code) == INSTALLSTATE_DEFAULT:
      log = evidence / f"leftover-uninstall-{code.strip('{}')}.log"
      returncode = subprocess.run(["msiexec.exe", "/x", code, "/quiet", "/norestart", "/L*v",
                                   str(log)], check=False).returncode
      if returncode:
        errors.append(f"msiexec /x {code} exited with {returncode}; see {log.name}")
  return errors


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--offline", type=Path, required=True)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--sima-cli", type=Path, required=True)
  parser.add_argument("--work", type=Path, required=True)
  parser.add_argument("--evidence", type=Path, required=True)
  parser.add_argument("--upgrade-base-candidate", type=Path)
  parser.add_argument("--expected-upgrade-base-commit")
  args = parser.parse_args()
  if (args.upgrade_base_candidate is None) != (args.expected_upgrade_base_commit is None):
    parser.error("--upgrade-base-candidate and --expected-upgrade-base-commit go together")
  validate(args)


if __name__ == "__main__":
  main()
