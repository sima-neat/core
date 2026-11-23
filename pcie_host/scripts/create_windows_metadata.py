#!/usr/bin/env python3
"""Create sima-cli metadata for the Windows MSI and optional Python wheel."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def gstreamer_installer_url(version: str) -> str:
  return ("https://gstreamer.freedesktop.org/data/pkg/windows/"
          f"{version}/msvc/gstreamer-1.0-msvc-x86_64-{version}.exe")


def metadata(msi: Path, wheel: Path, version: str, branch: str, commit: str,
             scheduler_msi: Path = None, scheduler_version: str = None,
             platform_version: str = None, gstreamer_version: str = None) -> dict:
  if not platform_version or not gstreamer_version:
    raise ValueError("Windows metadata requires exact Platform and GStreamer versions")
  if (scheduler_msi is not None
      and (not isinstance(scheduler_version, str)
           or not re.fullmatch(r"[0-9]+(?:[.][0-9]+){2}", scheduler_version))):
    raise ValueError("Scheduler MSI requires its exact numeric package version")
  resources = []
  openssh = "Install the Windows OpenSSH Client optional feature and reopen the terminal."
  platform_driver = (
      f"Install the SiMa Platform 3.0 Windows host package whose drivers report version "
      f"{platform_version}, as described in the PCIe Driver Installation guide "
      "(https://developer.sima.ai/hardware/getting-started/pcie-mode/driver-installation). "
      "If no device is found although that package is installed, Windows did not load the "
      "test-signed driver or has not detected the card. The driver loads only with Secure Boot "
      "off and test-signing on: in PowerShell started as administrator, Confirm-SecureBootUEFI "
      "must print False and bcdedit must list testsigning Yes. If both are set, check that the "
      "card is seated and powered, then shut the host down and power it on."
  )
  installation = {
      "type": "windows-msi",
      "prerequisites": [
          *({"name": name, "type": "executable", "value": value, "guidance": openssh}
            for name, value in (("Windows OpenSSH client", "ssh.exe"),
                                ("Windows OpenSSH secure-copy client", "scp.exe"),
                                ("Windows OpenSSH key generator", "ssh-keygen.exe"))),
          *({"name": name, "type": "windows-pnp-driver", "value": value,
             "version": platform_version,
             "guidance": platform_driver}
            for name, value in (
                ("SiMa Platform MLA driver", "SiMa.ai MLA Device"),
                ("SiMa Platform virtual Ethernet driver", "SiMa.ai Virtual Ethernet Adapter"))),
          {
              "name": "SiMa Platform PCIe service",
              "type": "windows-service",
              "value": "SiMaMlaDaemon",
              "guidance": (
                  "Install the compatible SiMa Platform 3.0 Windows package and start "
                  "the SiMaMlaDaemon service."
              ),
          },
          {
              "name": f"GStreamer {gstreamer_version} MSVC x64 runtime",
              "type": "command-version",
              "version": gstreamer_version,
              "arguments": ["--version"],
              "version-pattern": (
                  r"(?m)^gst-inspect-1\.0 version ([0-9]+(?:\.[0-9]+){2})$"
              ),
              "candidates": [
                  {
                      # The documented override; searched first.
                      "environment": "GSTREAMER_ROOT_X86_64",
                      "relative": "bin\\gst-inspect-1.0.exe",
                  },
                  {
                      # Set by the GStreamer installer.
                      "environment": "GSTREAMER_1_0_ROOT_MSVC_X86_64",
                      "relative": "bin\\gst-inspect-1.0.exe",
                  },
                  {
                      "environment": "ProgramFiles",
                      "relative": "gstreamer\\1.0\\msvc_x86_64\\bin\\gst-inspect-1.0.exe",
                  },
              ],
              "guidance": (
                  f"Install GStreamer {gstreamer_version} MSVC x64 with the development "
                  f"files from {gstreamer_installer_url(gstreamer_version)}, then open a "
                  "new terminal, or set GSTREAMER_ROOT_X86_64 to its installation root."
              ),
          },
      ],
      "verification": {
          "command": {
              "environment": "ProgramFiles",
              "relative": "SimaPCIeHost\\bin\\sima-pcie-host-doctor.cmd",
          },
      },
      "skills": [
          {
              "id": "neat-pcie-application-builder",
              "environment": "ProgramFiles",
              "relative": (
                  "SimaPCIeHost\\share\\sima-pcie-host\\skills\\"
                  "neat-pcie-application-builder"
              ),
          },
      ],
      # Runs last, unelevated, as install_pciehost.sh runs pcie-setup.sh.
      "setup": {
          "command": {
              "environment": "ProgramFiles",
              "relative": "SimaPCIeHost\\bin\\pcie-setup.cmd",
          },
      },
      "post-message": (
          "[bold]Neat PCIe host installation completed.[/bold]\n"
          # Static text: it cannot know about --native-only or --python-executable.
          "If Python support was installed into the default managed environment, run "
          "Python applications with its interpreter:\n"
          "& \"$env:USERPROFILE\\pyneatpcie\\Scripts\\python.exe\" app.py\n"
          "If it was installed with --python-executable, use that interpreter instead.\n"
          "To show the host and card software, run:\n"
          "& \"$env:ProgramFiles\\SimaPCIeHost\\bin\\neat.cmd\"\n"
          "To set up card SSH again, for example after adding or replacing a card, run:\n"
          "& \"$env:ProgramFiles\\SimaPCIeHost\\bin\\pcie-setup.cmd\""
      ),
  }
  artifact_paths = {}
  if scheduler_msi:
    resources.append(scheduler_msi.name)
    artifact_paths[scheduler_msi.name] = scheduler_msi
    installation["native-packages"] = [
        {
            "owner": "scheduler",
            "msi": scheduler_msi.name,
            "version": scheduler_version,
        },
        {"owner": "core", "msi": msi.name, "version": version},
    ]
  else:
    installation["msi"] = msi.name
  resources.append(msi.name)
  artifact_paths[msi.name] = msi
  total_bytes = msi.stat().st_size
  if scheduler_msi:
    total_bytes += scheduler_msi.stat().st_size
  if wheel:
    resources.append(wheel.name)
    artifact_paths[wheel.name] = wheel
    total_bytes += wheel.stat().st_size
    installation["python"] = {
        "wheel": wheel.name,
        "distribution": "pyneatpcie",
        "version": version,
        "import": "pyneatpcie",
        "requires-python": "3.14",
        "arch": "amd64",
    }

  checksums = {resource: sha256(artifact_paths[resource]) for resource in resources}
  total_mebibytes = max(1, (total_bytes + 1024 * 1024 - 1) // (1024 * 1024))
  return {
      "name": "gh:sima-neat/core/pciehost/windows/amd64",
      "version": version,
      "release": branch,
      "description": "SiMa.ai Neat PCIe host for Windows x64",
      "platforms": [{
          "type": "host",
          "os": ["windows"],
          "versions": {"windows": [">=11"]},
          "arch": ["amd64"],
      }],
      "resources": resources,
      "resources-checksum": checksums,
      "selectable-resources": [],
      "installation": installation,
      "artifact": {
          "type": "windows-msi",
          "repository": "core",
          "package_path": "pciehost/windows/amd64",
          "distro": "windows",
          "arches": ["amd64"],
      },
      "repository": "sima-neat/core",
      "branch": branch,
      "commit": commit,
      "published_at_utc": datetime.now(timezone.utc).isoformat(),
      "size": {
          "download": f"{total_mebibytes}MB",
          "install": f"{total_mebibytes}MB",
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--msi", required=True, type=Path)
  parser.add_argument("--scheduler-msi", type=Path)
  parser.add_argument("--scheduler-version")
  parser.add_argument("--platform-version", required=True)
  parser.add_argument("--gstreamer-version", required=True)
  parser.add_argument("--wheel", type=Path)
  parser.add_argument("--version", required=True)
  parser.add_argument("--branch", required=True)
  parser.add_argument("--commit", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  for path in (args.msi, args.scheduler_msi, args.wheel):
    if path and not path.is_file():
      parser.error(f"artifact does not exist: {path}")
  args.output.write_text(
      json.dumps(
          metadata(args.msi.resolve(), args.wheel.resolve() if args.wheel else None,
                   args.version, args.branch, args.commit,
                   args.scheduler_msi.resolve() if args.scheduler_msi else None,
                   args.scheduler_version, args.platform_version,
                   args.gstreamer_version),
          indent=2,
      ) + "\n",
      encoding="utf-8",
  )


if __name__ == "__main__":
  main()
