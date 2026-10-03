import json
import socket
import threading
import time

import pyneat
import pytest


_CATALOG = {
    "schema_version": 1, "instance_id": "daemon-a", "state": "ready", "ready": True,
    "stale": False, "revision": 3, "sequence": 5, "scan_sequence": 8,
    "last_success_at": "2026-10-01T01:02:03Z", "last_attempt_at": "2026-10-01T01:02:04Z",
    "error": None, "issues": [],
    "devices": [
        {"id": "camera:imx477 5-001a", "type": "camera", "provider": "daemon.camera.mipi",
         "camera": {"camera_name": "imx477 5-001a", "model": "imx477", "backend": "mipi",
                    "modes": [
                        {"format": "NV12", "width": 1920, "height": 1080, "framerate_num": 30,
                         "framerate_den": 1, "supported": True, "reason": ""},
                        {"format": "NV12", "framerate_num": 30, "framerate_den": 1,
                         "supported": False, "reason": "range is advisory",
                         "size_range": {"min_width": 640, "min_height": 480, "max_width": 1920,
                                        "max_height": 1080, "step_width": 16, "step_height": 8}},
                    ]}},
        {"id": "mic:1", "type": "microphone", "provider": "daemon.audio.alsa",
         "microphone": {"channels": 2, "nested": {"a": [1, 2]}}},
        {"id": "lidar:1", "type": "lidar", "provider": "lidar", "lidar": [1]},
    ],
}


def _list(tmp_path, body, status=200, before_reply=lambda: None):
  """Serve one HTTP response on a fake Sentinel socket and list it."""
  path = tmp_path / "api.sock"
  listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  listener.bind(str(path))
  listener.listen(1)

  def serve():
    client, _ = listener.accept()
    with client:
      request = b""
      while b"\r\n\r\n" not in request and (chunk := client.recv(4096)):
        request += chunk
      before_reply()
      payload = body.encode()
      client.sendall(
          f"HTTP/1.1 {status} Status\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
          + payload
      )

  server = threading.Thread(target=serve, daemon=True)
  server.start()
  try:
    return pyneat.peripherals._list_from_socket_for_testing(str(path), 1000)
  finally:
    server.join(timeout=2)
    listener.close()


def _assert_matches(native, expected):
  """Every field of the daemon document is bound under the same name."""
  if isinstance(expected, dict):
    for key, value in expected.items():
      _assert_matches(getattr(native, key), value)
  elif isinstance(expected, list):
    assert len(native) == len(expected)
    for item, value in zip(native, expected):
      _assert_matches(item, value)
  else:
    assert native == expected


def test_catalog_binds_every_field(tmp_path):
  catalog = _list(tmp_path, json.dumps(_CATALOG))

  assert callable(pyneat.peripherals.list)
  root = {k: v for k, v in _CATALOG.items() if k not in ("schema_version", "ready", "devices")}
  _assert_matches(catalog, root)
  _assert_matches(catalog[0].camera, _CATALOG["devices"][0]["camera"])
  assert [mode.is_range for mode in catalog[0].camera.modes] == [False, True]
  assert [device.id for device in catalog] == [d["id"] for d in _CATALOG["devices"]]
  assert len(catalog) == 3 and catalog[-1].id == "lidar:1"
  with pytest.raises(IndexError, match="peripheral catalog index out of range"):
    catalog[3]
  assert type(iter(catalog)).__name__ == "PeripheralCatalogIterator"


def test_stale_and_empty_snapshots(tmp_path):
  stale = dict(
      _CATALOG, state="degraded", stale=True,
      error={"code": "peripherals.discovery_failed", "reason": "camera scan failed"},
      issues=[{"provider": "daemon.camera.mipi", "code": "io.permission_denied",
               "reason": "permission denied", "retained_last_good": True}])
  catalog = _list(tmp_path, json.dumps(stale))
  _assert_matches(catalog, {k: stale[k] for k in ("state", "stale", "error", "issues")})
  assert catalog and len(catalog) == 3

  (tmp_path / "api.sock").unlink()
  empty = _list(tmp_path, json.dumps(dict(_CATALOG, devices=[])))
  assert not empty and len(empty) == 0 and list(empty) == []


def test_list_releases_the_gil_while_waiting(tmp_path):
  progress = threading.Event()
  observed = []
  worker = threading.Thread(target=lambda: (time.sleep(0.02), progress.set()))
  worker.start()
  _list(tmp_path, json.dumps(_CATALOG), before_reply=lambda: observed.append(progress.wait(0.5)))
  worker.join(timeout=1)
  assert observed == [True]


def test_details_cover_any_type(tmp_path):
  camera, microphone, lidar = _list(tmp_path, json.dumps(_CATALOG))

  assert camera.details == _CATALOG["devices"][0]["camera"]
  assert microphone.camera is None
  assert microphone.details == json.loads(microphone.details_json)
  assert microphone.details == {"channels": 2, "nested": {"a": [1, 2]}}
  microphone.details["channels"] = 1
  assert microphone.details["channels"] == 2
  assert lidar.details_json == "{}" and lidar.details == {}


@pytest.mark.parametrize(
    ("status", "body", "code", "fragment"),
    [
        (200, "{", "ERROR_IO_PARSE", "JSON parsing failed"),
        (
            200,
            json.dumps(dict(_CATALOG, state="starting", ready=False, devices=[])),
            "ERROR_PERIPHERAL_DAEMON_NOT_READY",
            "simaai-sentinel.service",
        ),
        (
            404,
            json.dumps({"error": "unknown endpoint"}),
            "ERROR_PERIPHERAL_DAEMON_UNAVAILABLE",
            "sima-cli neat install sentinel",
        ),
        (
            503,
            json.dumps({"error": "too_many_clients"}),
            "ERROR_PERIPHERAL_DAEMON_UNAVAILABLE",
            "Sentinel reported: too_many_clients.",
        ),
    ],
)
def test_errors_keep_structured_code(tmp_path, status, body, code, fragment):
  with pytest.raises(pyneat.NeatError) as error:
    _list(tmp_path, body, status)
  assert error.value.error_code == getattr(pyneat, code)
  assert fragment in str(error.value)


def test_missing_sentinel_asks_for_install(tmp_path):
  with pytest.raises(pyneat.NeatError) as error:
    pyneat.peripherals._list_from_socket_for_testing(str(tmp_path / "missing.sock"), 100)
  assert error.value.error_code == pyneat.ERROR_PERIPHERAL_DAEMON_UNAVAILABLE
  assert "sima-cli neat install sentinel" in str(error.value)
