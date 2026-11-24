#!/usr/bin/env python3
"""Exercise the installed Windows GenAI default as a disposable standard user."""

from __future__ import annotations

import configparser
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


MEDIA_OVERRIDE_VARIABLES = (
    "SIMAPCIE_GENAI_MEDIA_DIRECTORY",
    "SIMAPCIE_GENAI_MEDIA_SERVE_ROOT",
)


def read_serve_mapping(configuration: Path, root: str) -> Path:
  """Read one Platform daemon [serve] mapping without interpolation."""
  parser = configparser.ConfigParser(interpolation=None)
  try:
    with configuration.open("r", encoding="utf-8-sig") as stream:
      # The Platform file sets daemon options before its first section.
      parser.read_string("[daemon]\n" + stream.read(), source=str(configuration))
  except (OSError, configparser.Error) as error:
    raise RuntimeError(
        f"standard user cannot read Platform daemon configuration: {configuration}: {error}"
    ) from error
  if not parser.has_section("serve") or not parser.has_option("serve", root):
    raise RuntimeError(
        f"Platform daemon configuration has no [serve] {root} mapping: {configuration}"
    )
  value = parser.get("serve", root).strip()
  if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
    value = value[1:-1]
  value = os.path.expandvars(value)
  if not value or "\0" in value:
    raise RuntimeError(f"Platform daemon [serve] {root} mapping is empty or invalid")
  return Path(value)


def common_application_data() -> Path:
  """Resolve FOLDERID_ProgramData independently of the process environment."""
  if sys.platform != "win32":
    raise RuntimeError("standard-user GenAI qualification requires Windows")

  class Guid(ctypes.Structure):
    _fields_ = [
        ("data1", wintypes.DWORD),
        ("data2", wintypes.WORD),
        ("data3", wintypes.WORD),
        ("data4", ctypes.c_ubyte * 8),
    ]

  folder_id = Guid.from_buffer_copy(
      uuid.UUID("62ab5d82-fdc1-4dc3-a9dd-070d1d495d97").bytes_le)
  allocated = ctypes.c_wchar_p()
  sh_get_known_folder_path = ctypes.windll.shell32.SHGetKnownFolderPath
  sh_get_known_folder_path.argtypes = [
      ctypes.POINTER(Guid), wintypes.DWORD, wintypes.HANDLE,
      ctypes.POINTER(ctypes.c_wchar_p),
  ]
  sh_get_known_folder_path.restype = ctypes.c_long
  co_task_mem_free = ctypes.windll.ole32.CoTaskMemFree
  co_task_mem_free.argtypes = [ctypes.c_void_p]
  co_task_mem_free.restype = None
  status = sh_get_known_folder_path(
      ctypes.byref(folder_id), 0, None, ctypes.byref(allocated))
  if status != 0 or not allocated.value:
    raise RuntimeError(
        "Windows CommonApplicationData (FOLDERID_ProgramData) is unavailable "
        f"to the standard user: HRESULT=0x{status & 0xffffffff:08x}")
  try:
    return Path(allocated.value)
  finally:
    co_task_mem_free(ctypes.cast(allocated, ctypes.c_void_p))


def is_user_an_administrator() -> bool:
  if sys.platform != "win32":
    return False
  is_admin = ctypes.windll.shell32.IsUserAnAdmin
  is_admin.argtypes = []
  is_admin.restype = wintypes.BOOL
  return bool(is_admin())


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def require_readable_file(path: Path, description: str) -> None:
  try:
    with path.open("rb") as stream:
      stream.read(1)
  except OSError as error:
    raise RuntimeError(
        f"standard user cannot read required customer {description}: {path}: {error}"
    ) from error


def same_directory(left: Path, right: Path) -> bool:
  try:
    return os.path.samefile(left, right)
  except OSError:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def prove_writable_sentinel(directory: Path, token: str | None = None) -> None:
  """Prove create/write/read/remove access without leaving a qualification file."""
  sentinel = directory / (".neat-standard-user-" + (token or uuid.uuid4().hex))
  try:
    with sentinel.open("x", encoding="utf-8") as stream:
      stream.write("standard-user-default-root\n")
    if sentinel.read_text(encoding="utf-8") != "standard-user-default-root\n":
      raise RuntimeError("standard-user default-root sentinel contents changed")
  finally:
    sentinel.unlink(missing_ok=True)
  if sentinel.exists():
    raise RuntimeError(f"standard-user default-root sentinel remained: {sentinel}")


def run(payload_path: Path) -> dict:
  payload = json.loads(payload_path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise RuntimeError("standard-user GenAI payload must be a JSON object")
  for name in MEDIA_OVERRIDE_VARIABLES:
    if name in os.environ:
      raise RuntimeError(f"standard-user default-root proof retained forbidden override: {name}")
  if is_user_an_administrator():
    raise RuntimeError("standard-user GenAI qualification process has administrator privileges")

  import pyneatpcie as p

  prefix = Path(sys.prefix).resolve()
  module = Path(p.__file__).resolve()
  if prefix not in module.parents:
    raise RuntimeError(f"pyneatpcie did not load from the disposable Python: {module}")
  if not (p.Model and p.Runtime and p.Tensor):
    raise RuntimeError("installed pyneatpcie base API is incomplete")
  receipt = {
      "process_is_administrator": False,
      "python": "PASS",
      "standard_user": "PASS",
  }
  if not payload.get("include_genai"):
    return receipt

  options = p.genai.ConnectionOptions()
  application_data = common_application_data().resolve(strict=True)
  expected_directory = (application_data / "SiMa.ai" / "data").resolve(strict=True)
  actual_directory = Path(options.media_directory).resolve(strict=True)
  if not same_directory(actual_directory, expected_directory):
    raise RuntimeError(
        "installed GenAI default is not Windows CommonApplicationData\\SiMa.ai\\data: "
        f"expected={expected_directory}, actual={actual_directory}")
  if options.media_serve_root != "data":
    raise RuntimeError(
        f"installed GenAI default service root is not 'data': {options.media_serve_root!r}")

  configuration = application_data / "SiMa.ai" / "simaai-mla-daemon.conf"
  configured_directory = read_serve_mapping(configuration, options.media_serve_root)
  try:
    configured_directory = configured_directory.resolve(strict=True)
  except OSError as error:
    raise RuntimeError(
        f"Platform daemon [serve] data directory is unavailable: {configured_directory}: {error}"
    ) from error
  if not same_directory(configured_directory, actual_directory):
    raise RuntimeError(
        "Platform daemon [serve] data does not map the installed GenAI default: "
        f"configured={configured_directory}, default={actual_directory}")

  environment_value = payload.get("environment")
  if not isinstance(environment_value, dict):
    raise RuntimeError("standard-user GenAI payload has no child environment")
  child_environment = {
      str(name): str(value) for name, value in environment_value.items()
      if isinstance(name, str) and isinstance(value, str)
  }
  for name in MEDIA_OVERRIDE_VARIABLES:
    child_environment.pop(name, None)
  # Windows resolves the ProgramData known folder through SystemDrive, and its OpenSSH
  # client exits without a message when ProgramData is unset.
  child_environment.update({
      name: value for name, value in os.environ.items()
      if name.upper() in {"COMSPEC", "PROGRAMDATA", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP",
                          "WINDIR"}
  })

  executable = Path(str(payload["executable"])).resolve(strict=True)
  model_root = Path(child_environment["SIMAPCIE_GENAI_MODELS_PATH"])
  model_name = child_environment.get(
      "SIMA_TEST_LLIMA_ASR_MODEL", "whisper-small-a16w8")
  require_readable_file(
      model_root / model_name / "devkit" / "whisper_config.json", "ASR model")
  assets = Path(child_environment["SIMAPCIE_GENAI_TEST_ASSETS"])
  for name in ("audio.wav", "audio_de.wav", "audio_16k_mono_f32le.raw"):
    require_readable_file(assets / name, f"ASR asset {name}")
  require_readable_file(Path(child_environment["SIMAPCIE_SSH_KEY"]), "SSH key copy")
  # The card host keys pcie-setup would have recorded for this user.
  known_hosts = Path.home() / ".ssh" / "known_hosts"
  known_hosts.parent.mkdir(exist_ok=True)
  shutil.copyfile(payload["known_hosts"], known_hosts)

  prove_writable_sentinel(actual_directory)

  completed = subprocess.run(
      [executable, "asr", assets], text=True, capture_output=True,
      env=child_environment, cwd=executable.parent,
      timeout=int(payload["timeout_seconds"]), check=False)
  print(completed.stdout, end="")
  print(completed.stderr, end="", file=sys.stderr)
  if completed.returncode != 0:
    raise RuntimeError(
        "standard-user installed C++ ASR hardware path failed with "
        f"exit code {completed.returncode}")

  receipt.update({
      "config_binding": "PASS",
      "common_application_data": str(application_data),
      "connection_options": "PASS",
      "default_root": "PASS",
      "default_media_directory": str(actual_directory),
      "default_media_serve_root": options.media_serve_root,
      "override_absence": "PASS",
      "platform_config": str(configuration),
      "platform_config_media_directory": str(configured_directory),
      "real_genai": "PASS",
      "real_genai_case": "test_genai_asr_run",
      "real_genai_executable_sha256": sha256(executable),
      "required_assets_readable": "PASS",
      "sentinel": "PASS",
      "sentinel_cleanup": "PASS",
  })
  return receipt


def main() -> None:
  if len(sys.argv) != 3:
    raise SystemExit("usage: standard_user_genai.py PAYLOAD RECEIPT")
  payload = Path(sys.argv[1])
  receipt = Path(sys.argv[2])
  value = run(payload)
  temporary = receipt.with_suffix(receipt.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  temporary.replace(receipt)


if __name__ == "__main__":
  main()
