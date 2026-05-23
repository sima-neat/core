"""SiMa Neat Python bindings."""

import ctypes as _ctypes
import os as _os
from pathlib import Path as _Path
import sys as _sys
import types as _types
import warnings as _warnings


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


class _DeprecatedModule(_types.ModuleType):
  def __init__(self, name, target, message):
    super().__init__(name)
    self.__dict__["_target"] = target
    self.__dict__["_message"] = message
    self.__doc__ = getattr(target, "__doc__", None)

  def __getattr__(self, attr):
    _warnings.warn(self.__dict__["_message"], DeprecationWarning, stacklevel=2)
    return getattr(self.__dict__["_target"], attr)

  def __dir__(self):
    return dir(self.__dict__["_target"])


_graph = _core._graph
_sys.modules[__name__ + "._graph"] = _graph
graph = _DeprecatedModule(
    __name__ + ".graph",
    _graph,
    "pyneat.graph is the old low-level runtime graph surface and is deprecated. "
    "Use public pyneat.Graph/pyneat.graphs for application composition; use "
    "pyneat._graph only for internal runtime tests.",
)
_sys.modules[__name__ + ".graph"] = graph

def _set_doc(obj, doc: str) -> None:
  try:
    obj.__doc__ = doc
  except Exception:
    pass


install_wrappers(_core)

_REMOVED_NAMES = {
  "Session": "Session was removed. Use pyneat.Graph.",
  "SessionError": "SessionError was removed. Catch pyneat.NeatError.",
  "SessionReport": "SessionReport was removed. Use pyneat.GraphReport.",
  "SessionOptions": "SessionOptions was removed. Use pyneat.GraphOptions.",
  "ModelSessionOptions": "ModelSessionOptions was removed. Use pyneat.ModelRouteOptions.",
  "NodeGroup": "NodeGroup was removed. Reusable fragments are pyneat.Graph objects.",
  "UdpOutputNodeGroupOptions": (
      "UdpOutputNodeGroupOptions was renamed to pyneat.UdpOutputGraphOptions."
  ),
  "OptiViewOutputNodeGroupOptions": (
      "OptiViewOutputNodeGroupOptions was renamed to pyneat.OptiViewOutputGraphOptions."
  ),
  "OptiViewOutputNodeGroup": (
      "OptiViewOutputNodeGroup was renamed to pyneat.OptiViewOutputGraph."
  ),
}


def __getattr__(name):
  if name in _REMOVED_NAMES:
    raise AttributeError(_REMOVED_NAMES[name])
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_set_doc(
  _core.Model,
  """Model loads a compiled `.tar.gz` model archive and exposes reusable stage fragments.

  Use Model to assemble Graphs or run inference directly via build/run helpers.
  """,
)
_set_doc(
  _core.Graph,
  """Graph assembles Nodes, Models, and reusable Graph fragments into a deterministic pipeline."""
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
