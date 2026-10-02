import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import pyneat
import pytest


_PERIPHERAL_CPP_PROBE_ENV = "SIMA_NEAT_PERIPHERAL_CPP_PROBE"
_INSTALLED_PYTHON_TEST_SUFFIX = ("share", "sima-neat", "python", "tests")
_SOURCE_CPP_PROBE_RELATIVE_PATH = Path(
    "build/tests/peripheral_catalog_cpp_probe"
)


def _catalog(**overrides):
  value = {
      "schema_version": 1,
      "instance_id": "python-daemon-instance",
      "state": "ready",
      "ready": True,
      "stale": False,
      "revision": 3,
      "sequence": 5,
      "scan_sequence": 8,
      "last_success_at": "2026-10-01T01:02:03.004Z",
      "last_attempt_at": "2026-10-01T01:02:03.004Z",
      "error": None,
      "issues": [],
      "devices": [
          {
              "id": "camera:imx477 5-001a",
              "type": "camera",
              "provider": "daemon.camera.libcamera",
              "camera": {
                  "camera_name": "imx477 5-001a",
                  "model": "imx477",
                  "backend": "libcamera",
                  "modes": [
                      {
                          "format": "NV12",
                          "width": 1920,
                          "height": 1080,
                          "framerate_num": 30,
                          "framerate_den": 1,
                          "supported": True,
                          "reason": "",
                      },
                      {
                          "format": "NV12",
                          "size_range": {
                              "min_width": 640,
                              "min_height": 480,
                              "max_width": 1920,
                              "max_height": 1080,
                              "step_width": 16,
                              "step_height": 8,
                          },
                          "framerate_num": 30,
                          "framerate_den": 1,
                          "supported": False,
                          "reason": "range is advisory",
                      },
                  ],
              },
          },
          {
              "id": "lidar:future",
              "type": "lidar",
              "provider": "daemon.lidar.future",
              "lidar": {"future": True},
          },
      ],
  }
  value.update(overrides)
  return value


class _FakeServer:
  def __init__(self, path: Path, body: str, *, gate=None):
    self.path = path
    self.body = body.encode()
    self.gate = gate
    self.gate_observed = None
    self.error = None
    self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.listener.bind(str(path))
    self.listener.listen(1)
    self.thread = threading.Thread(target=self._serve, daemon=True)

  def __enter__(self):
    self.thread.start()
    return self

  def __exit__(self, *_):
    self.thread.join(timeout=2)
    self.listener.close()
    self.path.unlink(missing_ok=True)
    assert not self.thread.is_alive()
    if self.error:
      raise self.error

  def _serve(self):
    try:
      client, _ = self.listener.accept()
      with client:
        request = bytearray()
        while b"\r\n\r\n" not in request:
          chunk = client.recv(1)
          if not chunk:
            raise AssertionError("client closed before sending a complete request")
          request.extend(chunk)
        assert request.startswith(b"GET /v1/peripherals HTTP/1.1\r\n")
        if self.gate is not None:
          self.gate_observed = self.gate.wait(timeout=0.5)
        response = (
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(self.body)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + self.body
        )
        for offset in range(0, len(response), 3):
          client.sendall(response[offset : offset + 3])
    except Exception as error:  # surfaced by __exit__ on the test thread
      self.error = error


def _list(path: Path, timeout_ms=1000):
  return pyneat.peripherals._list_from_socket_for_testing(str(path), timeout_ms)


def _resolve_peripheral_cpp_probe(test_file=Path(__file__), environ=None):
  environ = os.environ if environ is None else environ
  test_path = Path(test_file).resolve()
  suffix_size = len(_INSTALLED_PYTHON_TEST_SUFFIX)
  installed = (
      test_path.parent.parts[-suffix_size:] == _INSTALLED_PYTHON_TEST_SUFFIX
  )

  override = environ.get(_PERIPHERAL_CPP_PROBE_ENV)
  if override:
    return Path(override), installed

  if installed:
    extras_prefix = test_path.parents[suffix_size]
    return extras_prefix / "lib/sima-neat/tests/peripheral_catalog_cpp_probe", True

  source_root = test_path.parents[2]
  return source_root / _SOURCE_CPP_PROBE_RELATIVE_PATH, False


def _require_peripheral_cpp_probe(test_file=Path(__file__), environ=None):
  probe, installed = _resolve_peripheral_cpp_probe(test_file, environ)
  if probe.is_file() and os.access(probe, os.X_OK):
    return probe, installed

  reason = "missing" if not probe.is_file() else "not executable"
  message = f"peripheral catalog C++ probe is {reason}: {probe}"
  if not installed:
    message += (
        f"; build the peripheral_catalog_cpp_probe target or set "
        f"{_PERIPHERAL_CPP_PROBE_ENV}"
    )
  pytest.fail(message, pytrace=False)


def _native_catalog(catalog):
  def camera_details(camera):
    if camera is None:
      return None
    modes = []
    for mode in camera.modes:
      value = {
          "format": mode.format,
          "framerate_num": mode.framerate_num,
          "framerate_den": mode.framerate_den,
          "supported": mode.supported,
          "reason": mode.reason,
      }
      if mode.size_range is None:
        value.update(width=mode.width, height=mode.height)
      else:
        value["size_range"] = {
            "min_width": mode.size_range.min_width,
            "min_height": mode.size_range.min_height,
            "max_width": mode.size_range.max_width,
            "max_height": mode.size_range.max_height,
            "step_width": mode.size_range.step_width,
            "step_height": mode.size_range.step_height,
        }
      modes.append(value)
    return {
        "camera_name": camera.camera_name,
        "model": camera.model,
        "backend": camera.backend,
        "modes": modes,
    }

  error = None if catalog.error is None else {
      "code": catalog.error.code,
      "reason": catalog.error.reason,
  }
  return {
      "instance_id": catalog.instance_id,
      "state": catalog.state,
      "stale": catalog.stale,
      "revision": catalog.revision,
      "sequence": catalog.sequence,
      "scan_sequence": catalog.scan_sequence,
      "last_success_at": catalog.last_success_at,
      "last_attempt_at": catalog.last_attempt_at,
      "error": error,
      "issues": [
          {
              "provider": issue.provider,
              "code": issue.code,
              "reason": issue.reason,
              "retained_last_good": issue.retained_last_good,
          }
          for issue in catalog.issues
      ],
      "devices": [
          {
              "id": device.id,
              "type": device.type,
              "provider": device.provider,
              "camera": camera_details(device.camera),
          }
          for device in catalog.devices
      ],
  }


def test_public_peripheral_catalog_maps_native_types(tmp_path):
  path = tmp_path / "catalog.sock"
  with _FakeServer(path, json.dumps(_catalog())):
    catalog = _list(path)

  assert callable(pyneat.peripherals.list)
  assert catalog.instance_id == "python-daemon-instance"
  assert (catalog.revision, catalog.sequence, catalog.scan_sequence) == (3, 5, 8)
  assert len(catalog) == 2
  assert list(catalog)[0].id == catalog[0].id
  assert catalog[-1].id == "lidar:future"
  assert catalog[-1].camera is None
  camera = catalog[0].camera
  assert camera.camera_name == "imx477 5-001a"
  assert camera.model == "imx477"
  assert camera.modes[0].width == 1920
  assert camera.modes[0].supported is True
  assert camera.modes[1].is_range is True
  assert camera.modes[1].size_range.step_width == 16
  assert camera.modes[1].reason == "range is advisory"
  with pytest.raises(IndexError):
    _ = catalog[2]


def test_peripheral_catalog_preserves_empty_and_stale_snapshots(tmp_path):
  path = tmp_path / "empty.sock"
  with _FakeServer(path, json.dumps(_catalog(devices=[]))):
    empty = _list(path)

  assert not empty
  assert list(empty) == []

  path = tmp_path / "stale.sock"
  body = _catalog(
      state="degraded",
      stale=True,
      error={"code": "peripherals.discovery_failed", "reason": "camera scan failed"},
      issues=[
          {
              "provider": "daemon.camera.libcamera",
              "code": "io.permission_denied",
              "reason": "permission denied",
              "retained_last_good": True,
          }
      ],
      changes=[
          {
              "sequence": 5,
              "revision": 3,
              "kind": "error",
              "error": {"code": "io.permission_denied", "reason": "permission denied"},
          }
      ],
  )
  with _FakeServer(path, json.dumps(body)):
    stale = _list(path)

  assert stale.stale is True
  assert stale.error.code == "peripherals.discovery_failed"
  assert stale.issues[0].retained_last_good is True
  assert len(stale) == 2


def test_peripheral_catalog_releases_gil_while_waiting(tmp_path):
  path = tmp_path / "gil.sock"
  progress = threading.Event()
  worker = threading.Thread(target=lambda: (time.sleep(0.02), progress.set()))
  with _FakeServer(path, json.dumps(_catalog()), gate=progress) as server:
    worker.start()
    catalog = _list(path)
  worker.join(timeout=1)

  assert catalog.instance_id == "python-daemon-instance"
  assert server.gate_observed is True


def test_peripheral_catalog_errors_keep_structured_code(tmp_path):
  missing = tmp_path / "missing.sock"
  with pytest.raises(pyneat.NeatError) as unavailable:
    _list(missing, 100)
  assert unavailable.value.error_code == pyneat.ERROR_PERIPHERAL_DAEMON_UNAVAILABLE
  assert "sima-cli neat install sentinel" in str(unavailable.value)
  assert "simaai-sentinel.service" in str(unavailable.value)

  path = tmp_path / "not-ready.sock"
  body = _catalog(
      state="starting",
      ready=False,
      revision=0,
      last_success_at=None,
      devices=[],
  )
  with _FakeServer(path, json.dumps(body)):
    with pytest.raises(pyneat.NeatError) as not_ready:
      _list(path)
  assert not_ready.value.error_code == pyneat.ERROR_PERIPHERAL_DAEMON_NOT_READY

  path = tmp_path / "malformed.sock"
  with _FakeServer(path, "{"):
    with pytest.raises(pyneat.NeatError) as malformed:
      _list(path)
  assert malformed.value.error_code == pyneat.ERROR_IO_PARSE


def test_peripheral_cpp_probe_resolution_is_layout_aware(tmp_path):
  source_test = tmp_path / "checkout/python/tests/test_peripherals.py"
  source_probe = tmp_path / "checkout/build/tests/peripheral_catalog_cpp_probe"
  assert _resolve_peripheral_cpp_probe(source_test, {}) == (source_probe, False)

  installed_test = (
      tmp_path / "extras/share/sima-neat/python/tests/test_peripherals.py"
  )
  installed_probe = (
      tmp_path / "extras/lib/sima-neat/tests/peripheral_catalog_cpp_probe"
  )
  assert _resolve_peripheral_cpp_probe(installed_test, {}) == (
      installed_probe,
      True,
  )

  override = tmp_path / "custom-probe"
  assert _resolve_peripheral_cpp_probe(
      source_test, {_PERIPHERAL_CPP_PROBE_ENV: str(override)}
  ) == (override, False)

  with pytest.raises(pytest.fail.Exception, match="build the .* target"):
    _require_peripheral_cpp_probe(source_test, {})

  with pytest.raises(pytest.fail.Exception, match="is missing"):
    _require_peripheral_cpp_probe(installed_test, {})

  source_probe.parent.mkdir(parents=True)
  source_probe.write_text("probe")
  source_probe.chmod(0o644)
  with pytest.raises(pytest.fail.Exception, match="not executable"):
    _require_peripheral_cpp_probe(source_test, {})

  source_probe.chmod(0o755)
  assert _require_peripheral_cpp_probe(source_test, {}) == (source_probe, False)

  installed_probe.parent.mkdir(parents=True)
  installed_probe.write_text("probe")
  installed_probe.chmod(0o644)
  with pytest.raises(pytest.fail.Exception, match="not executable"):
    _require_peripheral_cpp_probe(installed_test, {})

  installed_probe.chmod(0o755)
  assert _require_peripheral_cpp_probe(installed_test, {}) == (
      installed_probe,
      True,
  )


def test_cpp_and_python_peripheral_catalogs_match(tmp_path):
  probe, installed = _require_peripheral_cpp_probe()

  if not installed:
    body = json.dumps(_catalog())
    path = tmp_path / "parity.sock"
    with _FakeServer(path, body):
      python_catalog = _native_catalog(_list(path))
    with _FakeServer(path, body):
      cpp_catalog = json.loads(
          subprocess.check_output([probe, "--socket", path], text=True)
      )
    assert cpp_catalog == python_catalog
    return

  for _ in range(3):
    python_before = _native_catalog(pyneat.peripherals.list())
    cpp = json.loads(subprocess.check_output([probe], text=True))
    python_after = _native_catalog(pyneat.peripherals.list())
    if cpp == python_before or cpp == python_after:
      return
  raise AssertionError("the installed C++ and Python catalog mappings differ")
