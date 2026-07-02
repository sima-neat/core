"""SiMa NEAT PCIe host Python bindings."""

from pathlib import Path as _Path
import os as _os
import sys as _sys


def _prepend_env_path(name, directories):
  existing = [token for token in _os.environ.get(name, "").split(":") if token]
  merged = []
  for directory in directories:
    value = str(directory)
    if value and value not in merged and _Path(value).is_dir():
      merged.append(value)
  for value in existing:
    if value not in merged:
      merged.append(value)
  if merged:
    _os.environ[name] = ":".join(merged)


def _configure_runtime_environment():
  if _os.name != "posix":
    return

  candidates = []
  package_dir = _Path(__file__).resolve().parent
  candidates.append(package_dir)

  for multiarch in ("x86_64-linux-gnu", "aarch64-linux-gnu"):
    candidates.extend([
        _Path(f"/usr/lib/{multiarch}/sima-pcie-host/gst-plugins"),
        _Path(f"/usr/lib/{multiarch}/gstreamer-1.0"),
    ])

  for token in _os.environ.get("SIMA_PCIE_HOST_LIBRARY_PATH", "").split(":"):
    if token:
      candidates.append(_Path(token))

  plugin_dirs = [
      path
      for path in candidates
      if (path / "libgstneatpciehost.so").exists() or "gstreamer-1.0" in str(path)
  ]

  _prepend_env_path("LD_LIBRARY_PATH", candidates)
  _prepend_env_path("GST_PLUGIN_PATH", plugin_dirs)
  _prepend_env_path("GST_PLUGIN_PATH_1_0", plugin_dirs)


_configure_runtime_environment()

if _os.name == "posix":
  _old_dl_flags = _sys.getdlopenflags()
  _sys.setdlopenflags(_os.RTLD_NOW | _os.RTLD_GLOBAL)
  from . import _pypciehost_core as _core
  _sys.setdlopenflags(_old_dl_flags)
  del _old_dl_flags
else:
  from . import _pypciehost_core as _core

from ._pypciehost_core import *

__version__ = _core.__version__

del _configure_runtime_environment
del _prepend_env_path
