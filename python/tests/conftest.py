"""Shared pytest configuration for installed pyneat tests."""

import os
from pathlib import Path

import pytest


# The Python runtime tests exercise the actual runner outputs explicitly.  The
# hidden build-time startup preflight does an extra push/pull before each run,
# which can exhaust or time out on the DevKit's finite EV/MLA channels when the
# full installed suite is run as one pytest process.  Keep the functional checks
# deterministic by default while preserving an opt-in escape hatch for focused
# preflight debugging.
os.environ.setdefault("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "0")

# Kernels loads every SIMA_KERNEL_PATH provider on its registry's first use,
# before a test loads its own. Drop an ambient search path before any test can
# import pyneat, so provider tests see only the providers they load.
os.environ.pop("SIMA_KERNEL_PATH", None)

_KERNELS_REGISTRY = "libsimaai_kernels_providers.so"


def _kernels_provider_dir() -> Path:
  """Directory of the provider libraries shipped with the loaded Kernels registry.

  Kernels installs its providers in <libdir>/sima-kernels/providers, beside the
  registry in <libdir>, so this finds the providers built with the registry
  wherever this process loaded it from. Mirrors Runner's
  tests/runner/kernel_provider.h; move to a Kernels-owned runtime query once
  one exists.
  """
  import pyneat  # noqa: F401  The Scheduler frontend maps the registry.

  for line in Path("/proc/self/maps").read_text().splitlines():
    fields = line.split(maxsplit=5)
    if len(fields) == 6 and Path(fields[5]).name == _KERNELS_REGISTRY:
      return Path(fields[5]).parent / "sima-kernels" / "providers"
  raise RuntimeError(f"the Kernels provider registry {_KERNELS_REGISTRY} is not loaded")


@pytest.fixture
def kernels_provider():
  """Path of a provider library shipped with the loaded Kernels registry, by file name."""
  return lambda file: str(_kernels_provider_dir() / file)
