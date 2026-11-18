#!/usr/bin/env python3
"""Name a published Windows PCIe-host release.

The publisher and the hardware qualifier's local registry use these helpers,
so they share the layout sima-cli resolves for
``core/pciehost/windows/amd64@<tag>``: ``core/<tag>/latest.tag`` names the
12-character commit and ``core/<tag>/<commit12>/pciehost/windows/amd64/`` holds
``metadata.json`` and its resources, as for the Ubuntu host packages.
"""

from __future__ import annotations

import re
from urllib.parse import quote

from windows_candidate import msi_version


PACKAGE_PATH = "pciehost/windows/amd64"


def release_tag(version: object) -> str:
  msi_version(version)
  return f"v{version}"


def short_commit(commit: str) -> str:
  if not re.fullmatch(r"[0-9a-f]{40}", commit):
    raise ValueError(f"expected a full lowercase Git commit: {commit!r}")
  return commit[:12]


def channel(ref: str) -> str:
  if not ref.strip():
    raise ValueError("release ref is empty")
  return f"core/{quote(ref.strip(), safe='')}"


def latest_tag_key(ref: str) -> str:
  return f"{channel(ref)}/latest.tag"


def package_prefix(ref: str, commit: str) -> str:
  return f"{channel(ref)}/{short_commit(commit)}/{PACKAGE_PATH}"


def customer_target(ref: str) -> str:
  return f"core/{PACKAGE_PATH}@{ref}"
