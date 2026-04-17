#include "neat/graph.h"
#include "neat/nodes.h"
#include "neat/session.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

bool wants_help(int argc, char** argv) {
  return has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

void step(const std::string& name, const std::string& detail = {}) {
  if (detail.empty()) {
    std::cout << "STEP " << name << "\n";
  } else {
    std::cout << "STEP " << name << ": " << detail << "\n";
  }
}

void check(const std::string& name, bool cond, const std::string& detail = {}) {
  std::cout << "CHECK " << name << ": " << (cond ? "PASS" : "FAIL");
  if (!detail.empty())
    std::cout << " (" << detail << ")";
  std::cout << "\n";
  if (!cond)
    throw std::runtime_error("check failed: " + name);
}

void why(const std::string& detail) {
  std::cout << "WHY " << detail << "\n";
}

void tradeoff(const std::string& detail) {
  std::cout << "TRADEOFF " << detail << "\n";
}

void failure_mode(const std::string& detail) {
  std::cout << "FAILURE_MODE " << detail << "\n";
}

void interpret_output(const std::string& detail) {
  std::cout << "INTERPRET " << detail << "\n";
}

void print_signature(std::initializer_list<std::pair<std::string, std::string>> values) {
  static constexpr std::array<const char*, 7> kRequired = {
      "tutorial", "lang", "flow", "run_mode", "output_kind", "tensor_rank", "field_count",
  };
  for (const char* key : kRequired) {
    bool found = false;
    for (const auto& kv : values) {
      if (kv.first == key) {
        found = true;
        break;
      }
    }
    if (!found)
      throw std::invalid_argument(std::string("missing signature key: ") + key);
  }
  std::cout << "SIGNATURE {";
  bool first = true;
  for (const auto& kv : values) {
    if (!first)
      std::cout << ",";
    std::cout << kv.first << "=" << kv.second;
    first = false;
  }
  std::cout << "}\n";
}

} // namespace

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
  // CORE LOGIC
  // Preferred path: pipeline node negotiates media contract, stage stamps frame id.
  using namespace simaai::neat::graph;
  Graph g;

  const NodeId pipe =
      g.add(std::make_shared<nodes::PipelineNode>(simaai::neat::nodes::VideoConvert(), "convert"));
  const NodeId stamp = g.add(nodes::StampFrameIdNode("stamp"));
  g.connect(pipe, stamp);

  std::cout << GraphPrinter::to_text(g) << "\n";

  GraphRun run = GraphSession(std::move(g)).build();
  check("graph_push", run.push(pipe, make_sample()), "sample reached pipeline node");
  auto out = run.pull(stamp, 2000);
  check("graph_pull", out.has_value(), "stage sink produced output");
  run.stop();
  // END CORE LOGIC
  return {"pipeline_plus_stage", *out};
}

std::pair<std::string, simaai::neat::Sample> run_stage_only_fallback() {
  // CORE LOGIC
  // Fallback still teaches GraphSession/GraphRun when pipeline plugins are unavailable.
  using namespace simaai::neat::graph;
  Graph g;

  const NodeId stamp = g.add(nodes::StampFrameIdNode("stamp"));
  std::cout << GraphPrinter::to_text(g) << "\n";

  GraphRun run = GraphSession(std::move(g)).build();
  check("graph_push", run.push(stamp, make_sample()), "sample reached stage node");
  auto out = run.pull(stamp, 2000);
  check("graph_pull", out.has_value(), "stage sink produced output");
  run.stop();
  // END CORE LOGIC
  return {"stage_only_fallback", *out};
}

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << "\n";
  print_common_flags(std::cout);
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    step("input_contract", "build graph and push one deterministic tensor sample");
    step("run_mode_choice", "prefer pipeline+stage hybrid and fallback to stage-only");
    why("understand the contract first: inputs, run mode, and outputs");
    tradeoff("prefer deterministic samples and stable contracts over production realism");
    failure_mode(
        "runtime/plugin issues should degrade to runtime_fallback without losing observability");
    interpret_output("use CHECK markers plus SIGNATURE fields to validate behavior and parity");

    std::string flow = "pipeline_plus_stage";
    simaai::neat::Sample out;
    try {
      std::tie(flow, out) = run_pipeline_plus_stage();
    } catch (const std::exception& e) {
      std::cout << "fallback reason: " << e.what() << "\n";
      std::tie(flow, out) = run_stage_only_fallback();
    }

    step("output_interpretation", "validate stream/frame metadata after traversal");
    check("stream_id_present", !out.stream_id.empty(), "stream id should not be empty");
    check("frame_id_stamped", out.frame_id >= 0, "stamp stage should assign id");

    print_signature({
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
