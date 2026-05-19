"""SiMa Neat Python bindings."""

import ctypes as _ctypes
import os as _os
from pathlib import Path as _Path
import sys as _sys


def _existing_library_dirs():
  dirs = []

  def add(path):
    if path and path not in dirs and _Path(path).is_dir():
      dirs.append(path)

  for token in _os.environ.get("SIMA_NEAT_LIBRARY_PATH", "").split(":"):
    add(token)

  package_dir = _Path(__file__).resolve().parent
  add(package_dir)
  add(package_dir / "../neat-internals/gst-plugins")

  for path in (
      "/usr/lib/aarch64-linux-gnu/neat/gst-plugins",
      "/usr/lib/aarch64-linux-gnu/neat/runtime",
      "/usr/lib/sima-neat/gst-plugins",
      "/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
      "/lib/aarch64-linux-gnu/neat/gst-plugins",
      "/lib/aarch64-linux-gnu/neat/runtime",
      "/usr/local/lib",
  ):
    add(path)

  # Installed CI tests run from an extracted extras tree while pyneat itself is
  # installed in ~/pyneat.  If a future package carries runtime libs next to the
  # extras, find them without requiring LD_LIBRARY_PATH to be set before Python
  # starts.
  cwd = _Path.cwd()
  for root in (cwd, *cwd.parents[:4]):
    try:
      for candidate in root.glob("sima-neat-*-Linux-extras/lib/sima-neat/**/libgstneatallocator.so"):
        add(candidate.parent)
    except OSError:
      pass

  return dirs


def _prepend_env_path(name, directories):
  existing = [token for token in _os.environ.get(name, "").split(":") if token]
  merged = []
  for directory in directories:
    value = str(directory)
    if value and value not in merged:
      merged.append(value)
  for value in existing:
    if value not in merged:
      merged.append(value)
  if merged:
    _os.environ[name] = ":".join(merged)


def _configure_neat_runtime_environment():
  if _os.name != "posix":
    return

  directories = [_Path(path) for path in _existing_library_dirs()]
  runtime_dirs = [
      path
      for path in directories
      if (path / "libprocesscvu_testhooks.so").exists()
      or (path / "libneattensorbufferfast.so").exists()
      or (path / "libsimaaineatprofiler.so").exists()
      or (path / "libneatpreparedruntimebridge.so").exists()
      or (path / "libsimaaimem.so").exists()
  ]
  plugin_candidates = [
      path
      for path in directories
      if (path / "libgstneatprocesscvu.so").exists()
      or (path / "libgstneatallocator.so").exists()
      or (path / "libgstneattensorbuffer.so").exists()
  ]
  preferred_plugin_dirs = [
      path for path in plugin_candidates if "gst-plugins" in str(path) or "sima-neat" in str(path)
  ]
  plugin_dirs = preferred_plugin_dirs or plugin_candidates

  # GStreamer plugin discovery may spawn gst-plugin-scanner. That helper is a
  # separate process, so ctypes preloads in this process are not enough for
  # plugin-local runtime dependencies like libprocesscvu_testhooks.so. Export
  # the NEAT runtime/plugin paths before _pyneat_core can initialize GStreamer.
  _prepend_env_path("LD_LIBRARY_PATH", [*runtime_dirs, *plugin_dirs])
  _prepend_env_path("GST_PLUGIN_PATH", plugin_dirs)
  _prepend_env_path("GST_PLUGIN_PATH_1_0", plugin_dirs)


def _preload_neat_runtime_libraries():
  if _os.name != "posix":
    return

  flags = _os.RTLD_NOW | _os.RTLD_GLOBAL
  names = (
      "libgstsimaaimeta.so",
      "libsimaaimem.so",
      "libprocesscvu_testhooks.so",
      "libneattensorbufferfast.so",
      "libsimaaineatprofiler.so",
      "libgstneatallocator.so",
      "libgstneatbufferpool.so",
      "libgstneattensorbuffer.so",
      "libneatpreparedruntimebridge.so",
      "libneatprocessmlaruntime.so",
  )
  directories = _existing_library_dirs()
  for name in names:
    for directory in directories:
      path = _Path(directory) / name
      if not path.exists():
        continue
      try:
        _ctypes.CDLL(str(path), mode=flags)
        break
      except OSError:
        # Keep trying other candidate directories.  The actual extension import
        # below will still report the precise unresolved dependency if none of
        # the install locations is usable.
        pass


_configure_neat_runtime_environment()
_preload_neat_runtime_libraries()

# Load _pyneat_core with RTLD_GLOBAL so pyneat and NEAT GStreamer plugins share
# a single symbol namespace in this process. With RTLD_LOCAL (CPython default
# on POSIX), allocator/buffer-pool GObject types may be registered in separate
# loader scopes, which can cause plugin initialization failures and timeouts.
_old_dl_flags = _sys.getdlopenflags()
_sys.setdlopenflags(_os.RTLD_NOW | _os.RTLD_GLOBAL)

from . import _pyneat_core as _core

# Restore the previous dlopen flags for subsequent imports.
_sys.setdlopenflags(_old_dl_flags)
del _old_dl_flags
del _preload_neat_runtime_libraries
del _configure_neat_runtime_environment
del _prepend_env_path
del _existing_library_dirs

from ._pyneat_core import *
from ._wrappers import install_wrappers

def _set_doc(obj, doc: str) -> None:
  try:
    obj.__doc__ = doc
  except Exception:
    pass


install_wrappers(_core)

_set_doc(
  _core.Model,
  """Model loads a compiled MPK package and exposes reusable stage fragments.

  Use Model to assemble Sessions or run inference directly via build/run helpers.
  """,
)
_set_doc(
  _core.Session,
  """Session assembles Nodes/NodeGroups into a deterministic pipeline."""
)
_set_doc(
  _core.Run,
  """Run executes a built pipeline in sync or async mode."""
)
_set_doc(
  _core.Tensor,
  """Tensor represents typed payload data (shape, dtype, layout, device)."""
)
_set_doc(
  _core.Sample,
  """Sample wraps Tensor payloads with stream metadata."""
)

__version__ = _core.__version__
