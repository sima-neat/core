#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/InputPublicationMeasurement.h"
#include "pipeline/runtime/LttngMetricsCollector.h"
#include <gstsimaaitensorbuffer.h>

#include <gst/gst.h>

#include <cstdint>
#include <memory>
#include <thread>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using namespace simaai::neat::pipeline_internal;

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

GstBuffer* make_input(std::int64_t sequence, const char* stream = "0") {
  GstBuffer* buffer = gst_buffer_new();
  auto* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  require(meta != nullptr, "metadata registration");
  gst_structure_set(gst_custom_meta_get_structure(meta), "stream-id", G_TYPE_STRING, stream,
                    "orig-input-seq", G_TYPE_INT64, sequence, "input-seq", G_TYPE_INT64,
                    sequence + 10000, "frame-id", G_TYPE_INT64, sequence + 20000, nullptr);
  return buffer;
}

PluginTraceSelection select(const std::vector<InputPublicationIdentity>& entries) {
  PluginTraceSelection selection;
  selection.require_session_identity = true;
  selection.measurement_epoch = 1;
  selection.plugin_instance_ids.insert("fixture.m1");
  selection.require_sample_identity = true;
  for (const auto& entry : entries) {
    auto kind = PluginSampleKeyKind::OrigInputSeq;
    auto value = entry.original_sequence;
    if (value < 0) {
      kind = PluginSampleKeyKind::InputSeq;
      value = entry.input_sequence;
    }
    if (value < 0) {
      kind = PluginSampleKeyKind::FrameId;
      value = entry.frame_id;
    }
    if (value >= 0)
      add_measured_plugin_sample(&selection, {entry.stream_id, kind, value});
  }
  return selection;
}

void lifetime_contract() {
  InputPublicationMeasurement measurement;
  // Invalid address proves disabled observation does not even read metadata.
  auto off = measurement.observe(reinterpret_cast<GstBuffer*>(std::uintptr_t{1}));
  off.finish(GST_FLOW_OK);
  GstBuffer* buffer = make_input(0);
  const auto references = GST_MINI_OBJECT_REFCOUNT_VALUE(buffer);
  measurement.start(1);
  auto inline_push = measurement.observe(buffer);
  // An inline downstream output may let the caller stop before push returns.
  auto first = measurement.stop(1).entries;
  require(first.size() == 1 && first.front().original_sequence == 0 &&
              first.front().input_sequence == 10000 && first.front().frame_id == 20000 &&
              first.front().stream_id == "0",
          "final identity must exist before inline output/push return");
  inline_push.finish(GST_FLOW_OK);
  require(GST_MINI_OBJECT_REFCOUNT_VALUE(buffer) == references,
          "observer must not retain or make the buffer writable");

  measurement.start(2);
  auto failed = measurement.observe(buffer);
  failed.finish(GST_FLOW_ERROR);
  require(measurement.stop(2).entries.empty(), "failed publication reservation must roll back");

  measurement.start(3);
  auto old = measurement.observe(buffer);
  measurement.discard(3);
  measurement.start(4);
  auto current = measurement.observe(buffer);
  old.finish(GST_FLOW_ERROR);
  require(measurement.stop(3).entries.empty(), "stale stop must not close current epoch");
  current.finish(GST_FLOW_OK);
  require(measurement.stop(4).entries.size() == 1, "stale rollback must not erase new epoch");

  measurement.start(5);
  { auto exception = measurement.observe(buffer); }
  require(measurement.stop(5).entries.empty(), "exception must roll back reservation");

  // Already-published appsrc warmup is not retrospectively added when it later
  // completes. Work still outside appsrc belongs to its actual admission epoch.
  auto warmup = measurement.observe(buffer);
  measurement.start(6);
  warmup.finish(GST_FLOW_OK);
  require(measurement.stop(6).entries.empty(), "pre-scope appsrc warmup must stay excluded");

  measurement.start(7);
  auto pending_failure = measurement.observe(buffer);
  auto attempted = measurement.stop(7).entries;
  pending_failure.finish(GST_FLOW_FLUSHING);
  require(attempted.size() == 1, "stop must not wait on an in-progress handoff");
  const auto no_span = parse_lttng_trace_text("", 31, 32, false, select(attempted));
  require(no_span.plugin_metrics.empty() && no_span.plugin_metrics_unattributed.empty(),
          "an attempted/failed input is not an accepted latency without an actual span");
  gst_buffer_unref(buffer);
}

GstBuffer* make_compact_input() {
  using namespace simaai::gst;
  TensorBufferPublishContract contract;
  contract.stage_key = "linear-ingress-test";
  contract.physical_outputs.push_back({0, 4, "input"});
  TensorBufferPublishLogicalOutput logical;
  logical.logical_index = logical.physical_index = logical.memory_index = 0;
  logical.backend_output_index = logical.route_slot = 0;
  logical.logical_name = logical.backend_name = logical.segment_name = "input";
  logical.byte_offset = 0;
  logical.size_bytes = 4;
  logical.dtype = SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  logical.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
  logical.shape = {2, 2};
  logical.stride_bytes = {2, 1};
  contract.logical_outputs.push_back(logical);
  TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  require(
      tensor_buffer_prepare_meta_template_from_contract(contract, {{0, 0, 4}}, &prepared, &error),
      error.c_str());
  auto publication = tensor_buffer_freeze_publication(prepared, &error);
  require(bool(publication), error.c_str());
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 4, nullptr);
  require(tensor_buffer_attach_publication(buffer, publication, &error), error.c_str());
  return buffer;
}

void compact_identity_contract() {
  GstBuffer* buffer = make_compact_input();
  simaai::gst::TensorBufferFrameMetadata frame;
  frame.stream_id = "held-stream";
  frame.frame_id = 9;
  frame.buffer_id = frame.buffer_offset = 0;
  frame.original_sequence = 51;
  frame.input_sequence = 71;
  std::string error;
  require(simaai::gst::tensor_buffer_attach_frame_metadata(buffer, frame, &error),
          "compact metadata attach");
  InputPublicationMeasurement measurement;
  measurement.start(1);
  auto admission = measurement.observe(buffer);
  admission.finish(GST_FLOW_OK);
  const auto entries = measurement.stop(1).entries;
  require(entries.size() == 1 && entries[0].stream_id == "held-stream" &&
              entries[0].original_sequence == 51 && entries[0].input_sequence == 71 &&
              entries[0].frame_id == 9,
          "compact transport identity must remain authoritative");
  // Competing authority is a genuine production metadata-reader failure, not
  // a mocked exception. Observation must report failure and leave ownership alone.
  gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  int finalized = 0;
  gst_mini_object_weak_ref(
      GST_MINI_OBJECT(buffer), [](gpointer p, GstMiniObject*) { ++*static_cast<int*>(p); },
      &finalized);
  measurement.start(2);
  auto invalid = measurement.observe(buffer);
  invalid.finish(GST_FLOW_OK);
  const auto rejected = measurement.stop(2);
  require(rejected.failed && rejected.entries.empty(),
          "metadata failure must not qualify partial metrics");

  std::jthread reader([&](std::stop_token stop) {
    while (!stop.stop_requested()) {
      auto concurrent = measurement.observe(buffer);
      concurrent.finish(GST_FLOW_OK);
    }
  });
  for (std::uint64_t epoch = 3; epoch < 103; ++epoch) {
    measurement.start(epoch);
    auto invalid_current = measurement.observe(buffer);
    invalid_current.finish(GST_FLOW_OK);
    require(measurement.stop(epoch).failed, "concurrent old failure must not mask current epoch");
  }
  reader.request_stop();
  reader.join();
  gst_buffer_unref(buffer);
  require(finalized == 1, "failed observation must not leak or retain the input buffer");
}

void verify_ctf_regression(const std::string& text, int frames) {
  const auto old = parse_lttng_trace_text(text, 31, 32, false, select({}));
  require(old.parsed && old.plugin_metrics.empty(), "old linear empty selection regression");
  InputPublicationMeasurement measurement;
  measurement.start(1);
  for (std::int64_t i = 0; i < frames; ++i) {
    GstBuffer* buffer = make_input(i, "fixture-stream");
    auto admission = measurement.observe(buffer);
    admission.finish(GST_FLOW_OK);
    gst_buffer_unref(buffer);
  }
  const auto fixed =
      parse_lttng_trace_text(text, 31, 32, false, select(measurement.stop(1).entries));
  require(fixed.parsed && fixed.plugin_metrics.size() == 2, "real CTF components not selected");
  for (const auto& metric : fixed.plugin_metrics)
    require(metric.calls == static_cast<std::uint64_t>(frames) && metric.reliable,
            "real CTF count/reliability regression");
}
void captured_ctf_contract() {
  // Unmodified Babeltrace output from the real production adapter/provider and
  // LTTng userspace collector, not handcrafted parser events (2026-10-09).
  const std::string text =
      R"ctf([1791522730.387758487] aiagent-neat-sdk sima_neat_plugin:plugin_span: { cpu_id = 6 }, { vpid = 1414411, vtid = 1414411 }, { event_type = 0, run_id_hash = 31, graph_id_hash = 32, pipeline_segment_id = 4, runtime_node_id = 5, public_node_id = 6, plugin_instance_id = "fixture.m1", element_name = "neatkernelrunner0", stream_id = "fixture-stream", backend = "A65", phase = "Exec", kernel_name = "stage_wall_v2", frame_id = 0, request_id = 0x1, message_id = 0 }
[1791522730.387795388] aiagent-neat-sdk sima_neat_plugin:plugin_span: { cpu_id = 6 }, { vpid = 1414411, vtid = 1414411 }, { event_type = 0, run_id_hash = 31, graph_id_hash = 32, pipeline_segment_id = 4, runtime_node_id = 5, public_node_id = 6, plugin_instance_id = "fixture.m1", element_name = "neatkernelrunner0", stream_id = "fixture-stream", backend = "A65", phase = "Run", kernel_name = "backend_wall_v2", frame_id = 0, request_id = 0x1, message_id = 0 }
[1791522730.387800493] aiagent-neat-sdk sima_neat_plugin:plugin_span: { cpu_id = 6 }, { vpid = 1414411, vtid = 1414411 }, { event_type = 1, run_id_hash = 31, graph_id_hash = 32, pipeline_segment_id = 4, runtime_node_id = 5, public_node_id = 6, plugin_instance_id = "fixture.m1", element_name = "neatkernelrunner0", stream_id = "fixture-stream", backend = "A65", phase = "Run", kernel_name = "backend_wall_v2", frame_id = 0, request_id = 0x1, message_id = 0 }
[1791522730.387842378] aiagent-neat-sdk sima_neat_plugin:plugin_span: { cpu_id = 15 }, { vpid = 1414411, vtid = 1414446 }, { event_type = 1, run_id_hash = 31, graph_id_hash = 32, pipeline_segment_id = 4, runtime_node_id = 5, public_node_id = 6, plugin_instance_id = "fixture.m1", element_name = "neatkernelrunner0", stream_id = "fixture-stream", backend = "A65", phase = "Exec", kernel_name = "stage_wall_v2", frame_id = 0, request_id = 0x1, message_id = 0 }
)ctf";
  verify_ctf_regression(text, 1);
}

void real_ctf_regression(const std::string& path) {
  std::ifstream input(path);
  require(input.good(), "real CTF text fixture missing");
  const std::string text{std::istreambuf_iterator<char>(input), {}};
  verify_ctf_regression(text, 1000);
}

} // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  require(simaai::gst::tensor_buffer_register_meta(), "Runtime metadata registration");
  if (!gst_meta_get_info("GstSimaMeta")) {
    static const gchar* tags[] = {nullptr};
    gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr);
  }
  try {
    lifetime_contract();
    compact_identity_contract();
    captured_ctf_contract();
    if (argc > 1)
      real_ctf_regression(argv[1]);
    std::cout << "linear input measurement contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
