"""Windows runtime discovery and loaded-DLL admission for PyNeatPCIe."""

from pathlib import Path
import ctypes
import os


def _normalized_path(path):
  return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def platform_service_directory():
  try:
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"SYSTEM\CurrentControlSet\Services\SiMaMlaDaemon",
    ) as service:
      command, _ = winreg.QueryValueEx(service, "ImagePath")
  except (ImportError, OSError):
    return None

  command = os.path.expandvars(str(command)).strip()
  if command.startswith('"'):
    end = command.find('"', 1)
    executable = command[1:end] if end > 1 else ""
  else:
    executable = command.split(maxsplit=1)[0] if command else ""
  return Path(executable).parent if executable else None


def runtime_directories(package_dir, environ=None, service_directory=None):
  environ = os.environ if environ is None else environ
  candidates = [Path(package_dir), Path(package_dir).parent / "bin"]
  for token in environ.get("SIMA_PCIE_HOST_LIBRARY_PATH", "").split(os.pathsep):
    if token:
      candidates.append(Path(token))

  platform_runtime = environ.get("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR")
  if platform_runtime:
    candidates.append(Path(platform_runtime))
  service_runtime = platform_service_directory() if service_directory is None else service_directory
  if service_runtime:
    candidates.append(Path(service_runtime))

  gstreamer_root = environ.get("GSTREAMER_ROOT_X86_64")
  if gstreamer_root:
    candidates.append(Path(gstreamer_root) / "bin")

  program_files = environ.get("ProgramFiles")
  if program_files:
    root = Path(program_files)
    candidates.extend([
        root / "SimaPCIeHost" / "bin",
        root / "gstreamer" / "1.0" / "msvc_x86_64" / "bin",
    ])

  result = []
  seen = set()
  for directory in candidates:
    normalized = _normalized_path(directory)
    if normalized not in seen:
      seen.add(normalized)
      result.append(directory)
  return result


def find_loaded_dll_conflicts(directories, loaded_path):
  """Return approved DLL names that are already loaded from another location."""
  approved = {}
  for directory in directories:
    if not directory.is_dir():
      continue
    for path in directory.glob("*.dll"):
      approved.setdefault(path.name.casefold(), set()).add(
          _normalized_path(path))

  conflicts = []
  for name, paths in sorted(approved.items()):
    loaded = loaded_path(name)
    if loaded is None:
      continue
    normalized = _normalized_path(loaded)
    if normalized not in paths:
      conflicts.append((name, Path(loaded)))
  return conflicts


def _loaded_module_path(name):
  kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
  kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
  kernel32.GetModuleHandleW.restype = ctypes.c_void_p
  kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
  kernel32.GetModuleFileNameW.restype = ctypes.c_uint

  handle = kernel32.GetModuleHandleW(name)
  if not handle:
    return None
  buffer = ctypes.create_unicode_buffer(32768)
  length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
  if not length or length >= len(buffer):
    error = ctypes.get_last_error()
    raise OSError(error, f"cannot resolve loaded Windows module: {name}")
  return Path(buffer.value)


def admit_loaded_dlls(directories, loaded_path=_loaded_module_path):
  conflicts = find_loaded_dll_conflicts(directories, loaded_path)
  if conflicts:
    detail = ", ".join(f"{name} from {path}" for name, path in conflicts)
    raise ImportError(
        "PyNeatPCIe found runtime DLLs already loaded from outside the approved runtime "
        f"directories: {detail}. Start a fresh process with one compatible Platform, "
        "GStreamer, and Neat PCIe host installation.")


def configure(package_dir):
  directories = runtime_directories(package_dir)
  admit_loaded_dlls(directories)

  existing = [token for token in os.environ.get("PATH", "").split(os.pathsep) if token]
  selected = [str(path) for path in directories if path.is_dir()]
  selected_paths = {_normalized_path(path) for path in selected}
  os.environ["PATH"] = os.pathsep.join(selected + [
      value for value in existing if _normalized_path(value) not in selected_paths
  ])

  handles = []
  if hasattr(os, "add_dll_directory"):
    handles = [os.add_dll_directory(path) for path in selected]
  return directories, handles
