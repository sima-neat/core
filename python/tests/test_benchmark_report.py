"""Benchmark reports round trip through independent parsers; no hardware is required."""

import json
from pathlib import Path

import pytest
import yaml

import pyneat


@pytest.mark.parametrize("filename", ["yolo_benchmark.yaml", "yolo_benchmark.json",
                                      "yolo_benchmark", "yolo_benchmark.txt", ".json"])
def test_report_round_trip(tmp_path, filename):
  report = pyneat.BenchmarkReport()
  report.latency_ms = 0.12345678901234567
  report.latency_p50_ms = 0.1
  report.latency_p95_ms = 0.2
  report.latency_p99_ms = 0.3
  report.peak_power_watts = 8.5
  report.throughput_samples = 123
  report.throughput_duration_ms = 1200.5
  report.fps = 100.5
  report.avg_power_watts = 7.25
  report.energy_joules = 0.00125
  path = tmp_path / "model with spaces" / filename
  written = report.save(path)
  assert Path(written) == path
  load = json.loads if path.name.endswith(".json") else yaml.safe_load
  parsed = load(path.read_text())
  telemetry = parsed.pop("telemetry")

  def metric(value):
    return {field: getattr(value, field)
            for field in ("key", "label", "unit", "samples", "mean", "minimum", "maximum", "last")}

  expected_telemetry = {field: getattr(report.telemetry, field)
                        for field in ("status", "sentinel_version", "sample_interval_ms",
                                      "samples", "duration_ms")}
  for field in ("power_som_watts", "power_mla_watts", "die_temperature_max_c", "cpu_usage_pct",
                "mla_memory_mib", "cma_memory_mib", "linux_memory_used_mib"):
    expected_telemetry[field] = metric(getattr(report.telemetry, field))
  for field in ("power_rails_watts", "die_temperatures_c", "board_temperatures_c", "cpu_cores_usage_pct"):
    expected_telemetry[field] = [metric(value) for value in getattr(report.telemetry, field)]
  assert telemetry == expected_telemetry
  assert parsed == {
      "latency_ms": report.latency_ms,
      "latency_p50_ms": report.latency_p50_ms,
      "latency_p95_ms": report.latency_p95_ms,
      "latency_p99_ms": report.latency_p99_ms,
      "peak_power_watts": report.peak_power_watts,
      "throughput_samples": report.throughput_samples,
      "throughput_duration_ms": report.throughput_duration_ms,
      "fps": report.fps,
      "avg_power_watts": report.avg_power_watts,
      "energy_joules": report.energy_joules,
      "mla_layer_stats": {"status": "not requested", "layers": []},
      "mla_stage_stats": {"status": "not requested", "stages": []},
  }
  assert Path(report.save(str(path))) == path


@pytest.mark.parametrize("extension", ["yaml", "json"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
@pytest.mark.parametrize("field", ["latency_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "fps",
                                  "avg_power_watts", "peak_power_watts", "energy_joules",
                                  "throughput_duration_ms"])
def test_invalid_headline_preserves_existing_file(tmp_path, value, field, extension):
  path = tmp_path / f"existing.{extension}"
  path.write_text("previous report\n")
  report = pyneat.BenchmarkReport()
  setattr(report, field, value)
  with pytest.raises(ValueError, match=field):
    report.save(path)
  assert path.read_text() == "previous report\n"


def test_mla_stats_are_read_only_results():
  for stats in (pyneat.MlaLayer, pyneat.MlaLayerStats, pyneat.MlaStage, pyneat.MlaStageStats):
    with pytest.raises(TypeError):
      stats()
  report = pyneat.BenchmarkReport()
  for owner, field, value in ((report, "mla_layer_stats", None), (report, "mla_stage_stats", None),
                              (report.mla_layer_stats, "status", "ok"),
                              (report.mla_layer_stats, "layers", []),
                              (report.mla_stage_stats, "status", "ok"),
                              (report.mla_stage_stats, "stages", [])):
    # Without this, a misspelled field would also raise AttributeError.
    assert hasattr(owner, field)
    with pytest.raises(AttributeError):
      setattr(owner, field, value)
  assert report.mla_layer_stats.status == "not requested" and report.mla_layer_stats.layers == []
  assert report.mla_stage_stats.status == "not requested" and report.mla_stage_stats.stages == []


def test_invalid_destination(tmp_path):
  report = pyneat.BenchmarkReport()
  with pytest.raises((RuntimeError, ValueError)):
    report.save(tmp_path)
  parent = tmp_path / "regular_file"
  parent.write_text("keep")
  with pytest.raises(RuntimeError):
    report.save(parent / "child.yaml")
  assert parent.read_text() == "keep"
