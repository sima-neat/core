from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_tutorial_module():
  root = Path(__file__).resolve().parents[2]
  module_path = root / "tutorials" / "019_mipi_camera_input" / "mipi_camera_input.py"
  spec = importlib.util.spec_from_file_location("tutorial019_mipi_camera_input", module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load tutorial module from {module_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _load_python_utils_module():
  root = Path(__file__).resolve().parents[2]
  module_path = root / "tutorials" / "common" / "python_utils.py"
  spec = importlib.util.spec_from_file_location("tutorial_python_utils", module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load helper module from {module_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_pull_v4l2_frames_uses_explicit_neat_runtime_and_closes_run():
  module = _load_tutorial_module()

  fake_run = SimpleNamespace(
      pulled=0,
      stopped=False,
      closed=False,
  )

  def pull(*, timeout_ms):
    assert timeout_ms == 5000
    if fake_run.pulled == 0:
      fake_run.pulled += 1
      return object()
    return None

  def stop():
    fake_run.stopped = True

  def close():
    fake_run.closed = True

  fake_run.pull = pull
  fake_run.stop = stop
  fake_run.close = close

  class FakeSession:
    def build_source(self, options):
      assert options.queue_depth == 4
      assert options.overflow_policy == "keep-latest"
      assert options.output_memory == "owned"
      return fake_run

  class FakeRunOptions:
    def __init__(self):
      self.queue_depth = 0
      self.overflow_policy = None
      self.output_memory = None

  fake_neat = SimpleNamespace(
      RunOptions=FakeRunOptions,
      OverflowPolicy=SimpleNamespace(KeepLatest="keep-latest"),
      OutputMemory=SimpleNamespace(Owned="owned"),
  )

  pulled = module.pull_v4l2_frames(fake_neat, FakeSession(), 2)

  assert pulled == 1
  assert fake_run.stopped is True
  assert fake_run.closed is True


def test_find_pyneat_core_extension_prefers_build_elxr_codex(tmp_path, monkeypatch):
  module = _load_python_utils_module()

  build_ext = tmp_path / "build" / "python" / "_pyneat_core_test.so"
  build_ext.parent.mkdir(parents=True)
  build_ext.write_text("", encoding="utf-8")

  elxr_ext = tmp_path / "build-elxr-codex" / "python" / "_pyneat_core_test.so"
  elxr_ext.parent.mkdir(parents=True)
  elxr_ext.write_text("", encoding="utf-8")

  monkeypatch.setattr(module, "repo_root", lambda: tmp_path)
  monkeypatch.setattr(module.sysconfig, "get_config_var", lambda name: ".so" if name == "EXT_SUFFIX" else None)

  selected = module._find_pyneat_core_extension()

  assert selected == elxr_ext
