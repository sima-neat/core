#!/usr/bin/env python3
"""Resolve the Windows card lane's inputs in-job, as the Linux card lane does.

Ports of scripts/ci/pciehost/resolve_assets.sh (with
pcie_host/scripts/resolve_hardware_test_assets.sh) and
pcie_host/scripts/prepare_pcie_genai_models.sh, for a Windows runner without
bash. The runtime paths come from the build's card-input artifact, the
installed GStreamer, and the installed Platform service, not runner variables.
Writes KEY=VALUE lines for GITHUB_ENV and the hardware asset manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windows_hardware_assets import create_asset_manifest  # noqa: E402

# The URLs and fixtures of resolve_assets.sh / resolve_hardware_test_assets.sh.
MLA_ONLY_MODELS = {
    "SIMAPCIE_MLA_ONLY_MODEL": "https://docs.sima.ai/pkg_downloads/SDK2.1.3/models/modalix/"
                               "yolo26-detection/yolo26n-det-int8-b1.tar.gz",
    "SIMAPCIE_MLA_ONLY_BF16_MODEL": "https://docs.sima.ai/pkg_downloads/SDK2.1.3/models/modalix/"
                                    "yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz",
}
YOLOV8_MODEL_NAME = "yolo_v8s"
TEST_IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
# The fixtures of prepare_pcie_genai_models.sh: (directory, repository, config, revision).
GENAI_MODELS = (
    ("Qwen2.5-0.5B-Instruct-Autoround-a16w4", "simaai/Qwen2.5-0.5B-Instruct-Autoround-a16w4",
     "devkit/vlm_config.json", None),
    ("LFM2.5-VL-450M-Autoround-a16w4", "simaai/LFM2.5-VL-450M-Autoround-a16w4",
     "devkit/vlm_config.json", None),
    ("whisper-small-a16w8", "simaai/whisper-small-a16w8", "devkit/whisper_config.json",
     "b34d5a300e74fcd1e9bc46bd70bd23cfc2b6b8d3"),
)
GENAI_TEST_ASSETS = ("tests/assets/genai/audio.wav", "tests/assets/genai/audio_de.wav",
                     "tests/assets/genai/audio_16k_mono_f32le.raw", "tests/images/people.jpg")


def run(command: list[object], cwd: Path) -> None:
  print("+", subprocess.list2cmdline([str(value) for value in command]), flush=True)
  subprocess.run([str(value) for value in command], cwd=cwd, check=True,
                 env={**os.environ, "SIMA_CLI_CHECK_FOR_UPDATE": "0"})


def cached(cache: Path, pattern: str) -> Path | None:
  roots = (cache, Path.home() / ".simaai" / "modelzoo", Path.home() / ".sima" / "modelzoo")
  found = sorted(path for root in roots if root.is_dir() for path in root.rglob(pattern)
                 if path.is_file())
  return found[0] if found else None


def resolve_yolov8(cache: Path, sima_cli: Path, workspace: Path) -> Path:
  found = cached(cache, "*yolo*v8*s*.tar.gz")
  if not found:
    manifest = json.loads((workspace / "deps" / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("modelzoo-version") or manifest["platform-version"]
    run([sima_cli, "modelzoo", "-v", version, "get", YOLOV8_MODEL_NAME], cache)
    found = cached(cache, "*yolo*v8*s*.tar.gz")
  if not found:
    raise RuntimeError("failed to resolve the YOLOv8 model from the Model Zoo")
  return found


def resolve_download(cache: Path, sima_cli: Path, url: str) -> Path:
  path = cache / url.rsplit("/", 1)[1]
  if not path.is_file():
    run([sima_cli, "download", url], cache)
  if not path.is_file():
    raise RuntimeError(f"sima-cli did not download {url}")
  return path


def resolve_test_image(cache: Path) -> Path:
  path = cache / "test_image.jpg"
  if not path.is_file():
    with urlopen(TEST_IMAGE_URL, timeout=120) as response:
      path.write_bytes(response.read())
  return path


def prepare_genai_models(models: Path) -> Path:
  from huggingface_hub import snapshot_download  # The wheelhouse's sima-cli closure.
  models.mkdir(parents=True, exist_ok=True)
  for name, repository, config, revision in GENAI_MODELS:
    snapshot_download(repo_id=repository, local_dir=models / name, revision=revision)
    if not (models / name / config).is_file():
      raise RuntimeError(f"missing {config} in {models / name}")
  return models


def prepare_genai_test_assets(workspace: Path, cache: Path) -> Path:
  # Linux's tests read these from the checkout; the Windows tests take one directory.
  root = cache / "genai-test-assets"
  root.mkdir(parents=True, exist_ok=True)
  for name in GENAI_TEST_ASSETS:
    shutil.copy2(workspace / name, root / Path(name).name)
  return root


def installed_platform_runtime() -> Path:
  """Find the Platform runtime as the installed sima-pcie-host-env.cmd does."""
  output = subprocess.run(["sc.exe", "qc", "SiMaMlaDaemon"], capture_output=True, text=True,
                          check=False).stdout
  match = re.search(r"BINARY_PATH_NAME\s*:\s*(.+)", output)
  if not match:
    raise RuntimeError("the SiMa Platform service SiMaMlaDaemon is not installed")
  image = match[1].strip()
  executable = image[1:].split('"', 1)[0] if image.startswith('"') else image.split(" ", 1)[0]
  runtime = Path(executable).parent
  if not (runtime / "SiMaPCIe.dll").is_file():
    raise RuntimeError(f"the SiMa Platform runtime was not found at {runtime}")
  return runtime


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workspace", type=Path, required=True)
  parser.add_argument("--card-inputs", type=Path, required=True)
  parser.add_argument("--cache-dir", type=Path, required=True)
  parser.add_argument("--genai-models", type=Path, required=True)
  parser.add_argument("--sima-cli", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--revision", required=True)
  parser.add_argument("--include-genai", action="store_true")
  parser.add_argument("--env-file", type=Path, required=True)
  args = parser.parse_args()

  workspace = args.workspace.resolve(strict=True)
  inputs = args.card_inputs.resolve(strict=True)
  cache = args.cache_dir
  cache.mkdir(parents=True, exist_ok=True)
  cache = cache.resolve()
  gstreamer = Path(os.environ.get("GSTREAMER_ROOT_X86_64")
                   or Path(os.environ["ProgramFiles"]) / "gstreamer" / "1.0" / "msvc_x86_64")
  values = {
      "GSTREAMER_ROOT_X86_64": gstreamer,
      "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR": installed_platform_runtime(),
      "SIMA_WINDOWS_PYTHON_WHEELHOUSE": inputs / "wheelhouse",
      "SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH": inputs / "opencv" / "opencv" / "build" / "x64" /
                                              "vc16" / "bin",
      "SIMAPCIE_YOLOV8_MODEL": resolve_yolov8(cache, args.sima_cli, workspace),
      **{name: resolve_download(cache, args.sima_cli, url)
         for name, url in MLA_ONLY_MODELS.items()},
      "SIMAPCIE_TEST_IMAGE": resolve_test_image(cache),
      "SIMAPCIE_BOXDECODE_IMAGE": workspace / "tests" / "images" / "people.jpg",
  }
  if args.include_genai:
    values["SIMAPCIE_GENAI_MODELS_PATH"] = prepare_genai_models(args.genai_models)
    values["SIMAPCIE_GENAI_TEST_ASSETS"] = prepare_genai_test_assets(workspace, cache)
  for name in ("GSTREAMER_ROOT_X86_64", "SIMA_WINDOWS_PYTHON_WHEELHOUSE",
               "SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH"):
    if not values[name].is_dir():
      raise RuntimeError(f"{name} is not a directory: {values[name]}")
  environment = {name: str(path) for name, path in values.items()}
  manifest = create_asset_manifest(environment, args.include_genai, args.revision)
  args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
  environment["SIMA_WINDOWS_PCIE_ASSET_MANIFEST"] = str(args.manifest.resolve())
  with args.env_file.open("a", encoding="utf-8") as output:
    for name, value in environment.items():
      output.write(f"{name}={value}\n")
      print(f"{name}={value}")


if __name__ == "__main__":
  main()
