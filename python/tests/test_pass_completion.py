"""Completion pass: Phase 4 leftovers + Phase 2 metrics + Phase 1 class patches."""

from __future__ import annotations

import model_fixture_helpers as model_fixtures
import pyneat


# ── Phase 4 leftovers ────────────────────────────────────────────────────────────────────────


def test_phase4_leftover_nodes():
  for name in ("h264_caps_fixup", "h264_encode_sw", "pcie_src", "pcie_sink"):
    assert hasattr(pyneat.nodes, name), name
  assert isinstance(pyneat.nodes.h264_caps_fixup(), pyneat.Node)
  assert isinstance(pyneat.nodes.h264_encode_sw(bitrate_kbps=2000), pyneat.Node)
  src_opt = pyneat.PCIeSrcOptions()
  src_opt.format = "NV12"
  src_opt.width = 1920
  assert src_opt.width == 1920
  assert isinstance(pyneat.nodes.pcie_src(src_opt), pyneat.Node)
  sink_opt = pyneat.PCIeSinkOptions()
  sink_opt.data_buf_name = "overlay"
  sink_opt.use_multi_buffers = True
  assert sink_opt.use_multi_buffers is True
  assert isinstance(pyneat.nodes.pcie_sink(sink_opt), pyneat.Node)


# ── Phase 2 metrics ──────────────────────────────────────────────────────────────────────────


def test_measure_options_trace_fields():
  opt = pyneat.MeasureOptions()
  opt.plugin_latency_source = pyneat.advanced.MetricsTraceSource.Lttng
  opt.include_edge_latency = True
  opt.include_message_latency = True
  opt.message_latency_source = pyneat.advanced.MetricsTraceSource.Auto
  opt.retain_metrics_trace = True
  opt.metrics_trace_dir = "/tmp/traces"
  assert opt.plugin_latency_source == pyneat.advanced.MetricsTraceSource.Lttng
  assert opt.include_message_latency is True
  assert opt.metrics_trace_dir == "/tmp/traces"


def test_measure_plugin_latency_attribution_fields():
  row = pyneat.MeasurePluginLatency()
  for field in ("stream_id", "plugin_instance_id", "source", "attribution_source",
                "mapping_error", "reliable"):
    assert hasattr(row, field), field
  row.reliable = False
  assert row.reliable is False


def test_measure_report_diagnostic_fields():
  report = pyneat.MeasureReport()
  for field in ("plugin_latency_unattributed", "edge_latency", "edge_latency_unattributed",
                "plugin_latency_status", "message_latency_status", "metrics_trace_dir", "warnings",
                "trace_loss_detected", "graph_sample_timing_unkeyed", "graph_sample_timing_misses"):
    assert hasattr(report, field), field
  assert list(report.edge_latency) == []
  # MeasurePath* per-node structs are deferred — path_timing is not exposed.
  assert not hasattr(report, "path_timing")


def test_metrics_advanced_types():
  assert hasattr(pyneat.advanced, "MetricsTraceSource")
  assert hasattr(pyneat.advanced, "MeasureEdgeLatency")
  for member in ("Auto", "Off", "Lttng"):
    assert hasattr(pyneat.advanced.MetricsTraceSource, member), member


# ── Phase 1 class patches ────────────────────────────────────────────────────────────────────


def test_run_advanced_options_new_fields():
  opt = pyneat.RunAdvancedOptions()
  opt.sync_num_buffers_override = 8
  opt.prepare_output_cpu_visible = True
  assert opt.sync_num_buffers_override == 8
  assert opt.prepare_output_cpu_visible is True


def test_input_options_preprocess_meta():
  template = pyneat.PreprocessMetaTemplate()
  template.enabled = True
  template.target_width = 640
  template.target_height = 640
  template.resize_mode = "fit"
  assert template.target_width == 640

  opts = pyneat.InputOptions()
  assert opts.preprocess_meta is None  # std::optional default
  opts.preprocess_meta = template
  assert opts.preprocess_meta is not None
  assert opts.preprocess_meta.resize_mode == "fit"


def test_model_summary_text():
  # S11: text-only summary() alongside structured info().
  path = model_fixtures.strict_model_tar_path("SIMA_RESNET50_TAR")
  model = pyneat.Model(str(path))
  text = model.summary()
  assert isinstance(text, str)
  assert "Model:" in text
  assert "outputs:" in text
