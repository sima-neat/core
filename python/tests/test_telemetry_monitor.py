"""pyneat.TelemetryMonitor: board telemetry from Sentinel over a window the caller chooses.

The tests marked `needs_sentinel` run on a board with Sentinel; the rest run anywhere.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

import pyneat

SENTINEL_SOCKET = Path("/run/simaai-sentinel/api.sock")
needs_sentinel = pytest.mark.skipif(not SENTINEL_SOCKET.exists(), reason="needs Sentinel")

STATS_FIELDS = ("key", "label", "unit", "samples", "mean", "minimum", "maximum", "last")
ONE_MEASUREMENT = ("power_som_watts", "power_mla_watts", "die_temperature_max_c", "cpu_usage_pct",
                   "mla_memory_mib", "cma_memory_mib", "linux_memory_used_mib")
MEASUREMENT_LISTS = ("power_rails_watts", "die_temperatures_c", "board_temperatures_c",
                     "cpu_cores_usage_pct")


def as_dict(telemetry):
  """The fields of `telemetry`, as `save()` writes them."""

  def stats(value):
    return {field: getattr(value, field) for field in STATS_FIELDS}

  out = {field: getattr(telemetry, field)
         for field in ("status", "sentinel_version", "sample_interval_ms", "samples", "duration_ms")}
  out.update({field: stats(getattr(telemetry, field)) for field in ONE_MEASUREMENT})
  out.update({field: [stats(value) for value in getattr(telemetry, field)]
              for field in MEASUREMENT_LISTS})
  return out


def run_python(code: str, timeout: float = 60) -> subprocess.CompletedProcess:
  return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                        timeout=timeout, check=False)


def test_api_surface():
  assert isinstance(pyneat.BenchmarkReport().telemetry, pyneat.Telemetry)
  assert isinstance(pyneat.Telemetry().power_som_watts, pyneat.TelemetryStats)
  for name in ("start", "read", "stop", "running"):
    assert callable(getattr(pyneat.TelemetryMonitor, name)), name
  assert not hasattr(pyneat, "BenchmarkTelemetry")
  assert not hasattr(pyneat, "BenchmarkMetric")


def test_read_before_start():
  monitor = pyneat.TelemetryMonitor()
  telemetry = monitor.read()
  assert monitor.running() is False
  assert telemetry.status == "not started"
  assert (telemetry.samples, telemetry.duration_ms) == (0, 0.0)


def test_print_shows_what_save_writes(tmp_path):
  telemetry = pyneat.TelemetryMonitor().read()

  yaml_path = telemetry.save(tmp_path / "nested" / "telemetry.yaml")
  assert Path(yaml_path) == tmp_path / "nested" / "telemetry.yaml"
  assert str(telemetry) == Path(yaml_path).read_text()
  assert str(telemetry).startswith("# NEAT board telemetry from Sentinel.")
  assert yaml.safe_load(str(telemetry)) == as_dict(telemetry)

  json_path = telemetry.save(str(tmp_path / "telemetry.json"))
  assert json.loads(Path(json_path).read_text()) == as_dict(telemetry)

  assert repr(telemetry) == ("<Telemetry status='not started' samples=0 duration_ms=0.0 "
                             "power_som_watts.mean=0.00>")


def test_telemetry_is_read_only():
  telemetry = pyneat.TelemetryMonitor().read()
  with pytest.raises(AttributeError):
    telemetry.samples = 3
  with pytest.raises(AttributeError):
    telemetry.power_som_watts.mean = 1.0


def test_without_sentinel_the_status_says_how_to_install_it(capfd, tmp_path):
  # Exercise absence independently of the board's live daemon, without stopping it.
  socket_path = tmp_path / "missing-sentinel.sock"
  assert not socket_path.exists()
  monitor = pyneat.TelemetryMonitor(socket_path=str(socket_path))
  monitor.start()
  assert monitor.running() is True
  telemetry = monitor.stop()
  assert monitor.running() is False
  assert telemetry.samples == 0
  assert "sima-cli neat install sentinel" in telemetry.status
  assert capfd.readouterr().err.count("[WARN] TelemetryMonitor:") == 1


@needs_sentinel
def test_monitor_records_board_telemetry():
  monitor = pyneat.TelemetryMonitor()
  monitor.start()
  time.sleep(5)  # Sentinel samples every 2 s, so the window holds at least two new samples.
  partial = monitor.read()
  telemetry = monitor.stop()

  assert telemetry.status == "ok", telemetry.status
  assert telemetry.samples >= 2
  assert partial.samples <= telemetry.samples
  assert telemetry.duration_ms >= 5000
  assert telemetry.sentinel_version
  assert 1000 <= telemetry.sample_interval_ms <= 3000

  power = telemetry.power_som_watts
  assert (power.key, power.unit) == ("power_current_watts", "W")
  assert 1 <= power.samples <= telemetry.samples
  assert 0 < power.minimum <= power.mean <= power.maximum
  assert power.minimum <= power.last <= power.maximum
  assert telemetry.die_temperature_max_c.unit == "C"  # Units come from Sentinel.
  assert telemetry.cpu_cores_usage_pct
  assert yaml.safe_load(str(telemetry)) == as_dict(telemetry)

  again = monitor.stop()
  assert (again.samples, again.duration_ms) == (telemetry.samples, telemetry.duration_ms)


# --- The GIL --------------------------------------------------------------------------------


@needs_sentinel
@pytest.mark.parametrize("call", ["start", "stop"])
def test_waiting_for_sentinel_releases_the_gil(call):
  """A Python thread keeps running while the monitor waits for a busy Sentinel."""
  monitor = pyneat.TelemetryMonitor()
  if call == "stop":
    monitor.start()

  counted = 0
  done = threading.Event()

  def count():
    nonlocal counted
    while not done.is_set():
      counted += 1

  counter = threading.Thread(target=count)
  counter.start()
  # Sentinel answers one request at a time. A request it has not fully received keeps it
  # busy, so the monitor's request waits for its 1 s timeout.
  with socket.socket(socket.AF_UNIX) as busy:
    busy.connect(str(SENTINEL_SOCKET))
    busy.sendall(b"GET /v1/cache HTTP/1.1\r\n")
    time.sleep(0.2)
    before = counted
    started = time.monotonic()
    telemetry = getattr(monitor, call)()
    waited = time.monotonic() - started
    during = counted - before
    status = monitor.read().status
  done.set()
  counter.join()
  monitor.stop()

  assert waited >= 0.9, f"Sentinel answered while busy, so nothing blocked ({waited:.2f} s)"
  assert waited < 5, f"{call}() waited {waited:.2f} s for a busy Sentinel"
  assert during > 10_000, f"the counting thread ran {during} times while {call}() waited"
  if call == "start":
    assert "Sentinel did not answer" in status, status
  else:
    assert telemetry.duration_ms > 0


def test_calls_from_many_threads_at_once():
  code = """
import threading
import pyneat

monitor = pyneat.TelemetryMonitor()
errors = []

def use():
  try:
    for _ in range(25):
      monitor.start()
      monitor.read()
      monitor.running()
      monitor.stop()
  except BaseException as error:
    errors.append(error)

threads = [threading.Thread(target=use) for _ in range(4)]
for thread in threads:
  thread.start()
for thread in threads:
  thread.join()
assert not errors, errors
assert not monitor.running()
print("finished")
"""
  result = run_python(code)
  assert result.returncode == 0, result.stderr
  assert result.stdout.strip() == "finished"


@pytest.mark.parametrize("holder", ["global", "daemon thread"])
def test_interpreter_exits_with_a_running_monitor(holder):
  """Exiting without stop() neither hangs nor crashes, wherever the monitor is held."""
  if holder == "global":
    code = """
import pyneat
monitor = pyneat.TelemetryMonitor()
monitor.start()
print("started")
"""
  else:
    code = """
import threading
import pyneat

def hold():
  monitor = pyneat.TelemetryMonitor()
  monitor.start()
  ready.set()
  threading.Event().wait()

ready = threading.Event()
threading.Thread(target=hold, daemon=True).start()
ready.wait()
print("started")
"""
  result = run_python(code, timeout=30)
  assert result.returncode == 0, result.stderr
  assert result.stdout.strip() == "started"
