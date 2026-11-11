"""SiMa NEAT PCIe host Python bindings."""

from pathlib import Path as _Path
import os as _os
import sys as _sys


_dll_directory_handles = []


def _prepend_env_path(name, directories):
  existing = [token for token in _os.environ.get(name, "").split(_os.pathsep) if token]
  merged = []
  for directory in directories:
    value = str(directory)
    if value and value not in merged and _Path(value).is_dir():
      merged.append(value)
  for value in existing:
    if value not in merged:
      merged.append(value)
  if merged:
    _os.environ[name] = _os.pathsep.join(merged)


def _windows_platform_service_directory():
  try:
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"SYSTEM\CurrentControlSet\Services\SiMaMlaDaemon",
    ) as service:
      command, _ = winreg.QueryValueEx(service, "ImagePath")
  except (ImportError, OSError):
    return None

  command = _os.path.expandvars(str(command)).strip()
  if command.startswith('"'):
    end = command.find('"', 1)
    executable = command[1:end] if end > 1 else ""
  else:
    executable = command.split(maxsplit=1)[0] if command else ""
  return _Path(executable).parent if executable else None


def _windows_runtime_directories(package_dir):
  candidates = [package_dir, package_dir.parent / "bin"]
  for token in _os.environ.get("SIMA_PCIE_HOST_LIBRARY_PATH", "").split(_os.pathsep):
    if token:
      candidates.append(_Path(token))

  platform_runtime = _os.environ.get("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR")
  if platform_runtime:
    candidates.append(_Path(platform_runtime))
  service_runtime = _windows_platform_service_directory()
  if service_runtime:
    candidates.append(service_runtime)

  gstreamer_root = _os.environ.get("GSTREAMER_ROOT_X86_64")
  if gstreamer_root:
    candidates.append(_Path(gstreamer_root) / "bin")

  candidates.append(_Path(r"C:\simaai\build\x64\Bin\Release"))

  program_files = _os.environ.get("ProgramFiles")
  if program_files:
    root = _Path(program_files)
    candidates.extend([
        root / "SimaPCIeHost" / "bin",
        root / "gstreamer" / "1.0" / "msvc_x86_64" / "bin",
    ])
  return candidates


def _configure_runtime_environment():
  package_dir = _Path(__file__).resolve().parent
  if _os.name == "nt":
    candidates = _windows_runtime_directories(package_dir)
    _prepend_env_path("PATH", candidates)
    if hasattr(_os, "add_dll_directory"):
      for directory in candidates:
        if directory.is_dir():
          _dll_directory_handles.append(_os.add_dll_directory(str(directory)))

    plugin_dirs = [
        package_dir.parent / "bin" / "gstreamer-1.0",
        package_dir / "gstreamer-1.0",
    ]
    for directory in candidates:
      plugin_dirs.extend([
          directory / "gstreamer-1.0",
          directory.parent / "lib" / "gstreamer-1.0",
      ])
    _prepend_env_path("GST_PLUGIN_PATH", plugin_dirs)
    _prepend_env_path("GST_PLUGIN_PATH_1_0", plugin_dirs)
    return

  if _os.name != "posix":
    return

  candidates = []
  candidates.append(package_dir)

  for multiarch in ("x86_64-linux-gnu", "aarch64-linux-gnu"):
    candidates.extend([
        _Path(f"/usr/lib/{multiarch}/sima-pcie-host/gst-plugins"),
        _Path(f"/usr/lib/{multiarch}/gstreamer-1.0"),
    ])

  for token in _os.environ.get("SIMA_PCIE_HOST_LIBRARY_PATH", "").split(_os.pathsep):
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
  from . import _pyneatpcie_core as _core
  _sys.setdlopenflags(_old_dl_flags)
  del _old_dl_flags
else:
  from . import _pyneatpcie_core as _core

from ._pyneatpcie_core import *

__version__ = _core.__version__

genai = _core.genai
_sys.modules[__name__ + ".genai"] = genai

del _configure_runtime_environment
del _prepend_env_path
del _windows_platform_service_directory
del _windows_runtime_directories
