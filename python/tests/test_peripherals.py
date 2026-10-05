import copy
import json
import socket
import threading
import time
from pathlib import Path

import pyneat
import pytest


# Sentinel's published contract fixture, docs/peripherals/catalog-example.json at
# sima-neat/sentinel commit 7efb980: a DevKit capture (IMX477, Logitech C920 camera and
# microphone) with the USB camera trimmed to one mode per format. The byte-for-byte
# copy lives in tests/assets/peripherals and is installed next to this file.
def _load_example():
  here = Path(__file__).resolve()
  for path in (here.with_name("catalog-example.json"),
               here.parents[2] / "tests" / "assets" / "peripherals" / "catalog-example.json"):
    if path.is_file():
      return json.loads(path.read_text())
  raise FileNotFoundError("catalog-example.json")


_CATALOG = _load_example()
_MIPI, _USB, _MIC = _CATALOG["devices"]
_USB_REASON = (
    "CameraInput's default libcamera profile accepts MIPI cameras only. These rules do not "
    "classify raw V4L2 profiles such as MetoakSimor (RAW8 1920x360, selected with "
    "CameraInputOptions.profile and device)."
)


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


def test_catalog_binds_every_field(tmp_path):
  catalog = _list(tmp_path, json.dumps(_CATALOG))

  assert callable(pyneat.peripherals.list)
  assert catalog.revision == _CATALOG["revision"]
  assert catalog.observed_at == _CATALOG["observed_at"]
  assert catalog.errors == []
  assert [device.id for device in catalog] == [d["id"] for d in _CATALOG["devices"]]
  assert [device.type for device in catalog.devices] == ["camera", "camera", "microphone"]
  assert len(catalog) == 3 and catalog[-1].id == _MIC["id"]
  with pytest.raises(IndexError, match="peripheral catalog index out of range"):
    catalog[3]
  assert type(iter(catalog)).__name__ == "PeripheralCatalogIterator"

  mipi, usb = catalog[0].camera, catalog[1].camera
  assert (mipi.camera_name, mipi.model, mipi.backend) == ("imx477 5-001a", "imx477", "mipi")
  assert (usb.camera_name, usb.model, usb.backend) == (None, "HD Pro Webcam C920", "v4l2")
  format_reason = "CameraInput's default libcamera profile supports NV12 output only."
  assert [
      (m.format, m.width, m.height, m.is_range, m.framerate_num, m.framerate_den, m.supported,
       m.reason)
      for m in mipi.modes
  ] == [
      (mode["format"], mode["width"], mode["height"], False, 0, 1, mode["format"] == "NV12",
       "" if mode["format"] == "NV12" else format_reason)
      for mode in _MIPI["modes"]
  ]
  assert [
      (m.format, m.width, m.height, m.is_range, m.framerate_num, m.framerate_den, m.supported,
       m.reason)
      for m in usb.modes
  ] == [
      ("MJPG", 1920, 1080, False, 30, 1, False, _USB_REASON),
      ("YUYV", 2560, 1472, False, 2, 1, False, _USB_REASON),
  ]


def test_size_range_and_interval_range(tmp_path):
  catalog = copy.deepcopy(_CATALOG)
  mode = catalog["devices"][1]["modes"][1]
  del mode["width"], mode["height"]
  mode["size_range"] = {"type": "stepwise", "min_width": 640, "min_height": 480,
                        "max_width": 1920, "max_height": 1080, "step_width": 16,
                        "step_height": 8}
  mode["frame_intervals"] = [{"width": 640, "height": 480, "intervals": [
      {"type": "stepwise", "minimum": {"numerator": 1, "denominator": 60},
       "maximum": {"numerator": 1, "denominator": 5},
       "step": {"numerator": 1, "denominator": 1000}}]}]
  ranged = _list(tmp_path, json.dumps(catalog))[1].camera.modes[1]
  assert ranged.is_range and (ranged.width, ranged.height) == (0, 0)
  assert (ranged.framerate_num, ranged.framerate_den) == (60, 1)
  size_range = ranged.size_range
  assert (size_range.min_width, size_range.min_height, size_range.max_width,
          size_range.max_height, size_range.step_width, size_range.step_height) == (
              640, 480, 1920, 1080, 16, 8)


def test_provider_errors_and_first_scan(tmp_path):
  error = {"provider": "camera.v4l2", "code": "io.permission_denied",
           "reason": "permission denied"}
  catalog = _list(tmp_path, json.dumps(dict(_CATALOG, errors=[error])))
  assert [(e.provider, e.code, e.reason) for e in catalog.errors] == [tuple(error.values())]
  assert len(catalog) == 3

  (tmp_path / "api.sock").unlink()
  first = _list(tmp_path, json.dumps(dict(_CATALOG, observed_at=None, devices=[])))
  assert first.observed_at is None
  assert not first and len(first) == 0 and list(first) == []


def test_list_releases_the_gil_while_waiting(tmp_path):
  progress = threading.Event()
  observed = []
  worker = threading.Thread(target=lambda: (time.sleep(0.02), progress.set()))
  worker.start()
  _list(tmp_path, json.dumps(_CATALOG), before_reply=lambda: observed.append(progress.wait(0.5)))
  worker.join(timeout=1)
  assert observed == [True]


def test_details_cover_any_type(tmp_path):
  mipi, _, microphone = _list(tmp_path, json.dumps(_CATALOG))

  assert mipi.details == _MIPI
  assert microphone.camera is None
  assert microphone.details == json.loads(microphone.details_json) == _MIC
  microphone.details["name"] = "changed"
  assert microphone.details["name"] == _MIC["name"]


@pytest.mark.parametrize(
    ("status", "body", "code", "fragment"),
    [
        (200, "{", "ERROR_IO_PARSE", "JSON parsing failed"),
        (
            404,
            json.dumps({"error": "unknown Sentinel API endpoint"}),
            "ERROR_PERIPHERAL_DAEMON_UNAVAILABLE",
            "sima-cli neat install sentinel",
        ),
        (
            503,
            json.dumps({"error": "peripheral discovery is not running"}),
            "ERROR_PERIPHERAL_DAEMON_UNAVAILABLE",
            "Sentinel reported: peripheral discovery is not running.",
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
