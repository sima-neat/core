#include "neat/graph.h"
#include "neat/nodes.h"
#include "neat/session.h"
#include "common/cpp_utils.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<int64_t> contiguous_strides_bytes(const std::vector<int64_t>& shape,
                                              int64_t elem_bytes) {
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = elem_bytes;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

simaai::neat::Sample make_sample() {
  const int w = 8;
  const int h = 8;
  const int c = 3;
  const std::size_t bytes = static_cast<std::size_t>(w) * h * c;

  simaai::neat::Tensor t;
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, c};
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::RGB, ""};
  t.storage = simaai::neat::make_cpu_owned_storage(bytes);
  t.strides_bytes = contiguous_strides_bytes(t.shape, 1);
  t.read_only = false;

  {
    auto map = t.map(simaai::neat::MapMode::Write);
    auto* p = static_cast<std::uint8_t*>(map.data);
    for (std::size_t i = 0; i < bytes; ++i) {
      p[i] = static_cast<std::uint8_t>(i % 255);
    }
  }
  t.read_only = true;

  simaai::neat::Sample s;
  s.kind = simaai::neat::SampleKind::Tensor;
  s.tensor = std::move(t);
  s.stream_id = "graph";
  s.frame_id = -1;
  return s;
}

std::pair<std::string, simaai::neat::Sample> run_pipeline_plus_stage() {
  // Preferred path: pipeline node negotiates media contract, stage stamps frame id.
  using namespace simaai::neat::graph;
  Graph g;

  const NodeId pipe =
      g.add(std::make_shared<nodes::PipelineNode>(simaai::neat::nodes::VideoConvert(), "convert"));
  const NodeId stamp = g.add(nodes::StampFrameIdNode("stamp"));
  g.connect(pipe, stamp);

  std::cout << GraphPrinter::to_text(g) << "\n";

  GraphRun run = GraphSession(std::move(g)).build();
  tutorial_v2::check("graph_push", run.push(pipe, make_sample()), "sample reached pipeline node");
  auto out = run.pull(stamp, 2000);
  tutorial_v2::check("graph_pull", out.has_value(), "stage sink produced output");
  run.stop();

  return {"pipeline_plus_stage", *out};
}

std::pair<std::string, simaai::neat::Sample> run_stage_only_fallback() {
  // Fallback still teaches GraphSession/GraphRun when pipeline plugins are unavailable.
  using namespace simaai::neat::graph;
  Graph g;

  const NodeId stamp = g.add(nodes::StampFrameIdNode("stamp"));
  std::cout << GraphPrinter::to_text(g) << "\n";

  GraphRun run = GraphSession(std::move(g)).build();
  tutorial_v2::check("graph_push", run.push(stamp, make_sample()), "sample reached stage node");
  auto out = run.pull(stamp, 2000);
  tutorial_v2::check("graph_pull", out.has_value(), "stage sink produced output");
  run.stop();

  return {"stage_only_fallback", *out};
}

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << "\n";
  tutorial_v2::print_common_flags(std::cout);
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (tutorial_v2::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    tutorial_v2::step("input_contract", "build graph and push one deterministic tensor sample");
    tutorial_v2::step("run_mode_choice", "prefer pipeline+stage hybrid and fallback to stage-only");
    tutorial_v2::why("understand the contract first: inputs, run mode, and outputs");
    tutorial_v2::tradeoff(
        "prefer deterministic samples and stable contracts over production realism");
    tutorial_v2::failure_mode(
        "runtime/plugin issues should degrade to runtime_fallback without losing observability");
    tutorial_v2::interpret_output(
        "use CHECK markers plus SIGNATURE fields to validate behavior and parity");

    std::string flow = "pipeline_plus_stage";
    simaai::neat::Sample out;
    try {
      std::tie(flow, out) = run_pipeline_plus_stage();
    } catch (const std::exception& e) {
      std::cout << "fallback reason: " << e.what() << "\n";
      std::tie(flow, out) = run_stage_only_fallback();
    }

    tutorial_v2::step("output_interpretation", "validate stream/frame metadata after traversal");
    tutorial_v2::check("stream_id_present", !out.stream_id.empty(),
                       "stream id should not be empty");
    tutorial_v2::check("frame_id_stamped", out.frame_id >= 0, "stamp stage should assign id");

    tutorial_v2::print_signature({
        {"tutorial", "014"},
        {"lang", "cpp"},
        {"flow", flow},
        {"run_mode", "graph_sync_pull"},
        {"output_kind", std::to_string(static_cast<int>(out.kind))},
        {"tensor_rank", out.tensor.has_value() ? std::to_string(out.tensor->shape.size()) : "-1"},
        {"field_count", std::to_string(out.fields.size())},
    });

    std::cout << "Output stream=" << out.stream_id << " frame=" << out.frame_id << "\n";
    std::cout << "[OK] 014_graph_basics\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
