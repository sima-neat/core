// Compose a minimal NEAT Graph: PipelineNode -> StampFrameIdNode, push one sample, pull output.
//
// Usage:
//   tutorial_v2_012_build_a_custom_data_graph

#include "neat/graph.h"
#include "neat/nodes.h"
#include "neat/session.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
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
    for (std::size_t i = 0; i < bytes; ++i)
      p[i] = static_cast<std::uint8_t>(i % 255);
  }
  t.read_only = true;

  simaai::neat::Sample s;
  s.kind = simaai::neat::SampleKind::Tensor;
  s.tensor = std::move(t);
  s.stream_id = "graph";
  s.frame_id = -1;
  return s;
}

} // namespace

int main() {
  try {
    using namespace simaai::neat::graph;

    // CORE LOGIC
    // A Graph is a DAG of nodes. Here: a PipelineNode (wraps a classic gst node)
    // feeds a StampFrameIdNode, which assigns a monotonic frame id.
    Graph g;

    const NodeId pipe = g.add(
        std::make_shared<nodes::PipelineNode>(simaai::neat::nodes::VideoConvert(), "convert"));
    const NodeId stamp = g.add(nodes::StampFrameIdNode("stamp"));
    g.connect(pipe, stamp);

    std::cout << GraphPrinter::to_text(g) << "\n";

    GraphRun run = GraphSession(std::move(g)).build();
    run.push(pipe, make_sample());
    auto out = run.pull(stamp, /*timeout_ms=*/2000);
    run.stop();
    // END CORE LOGIC

    if (!out.has_value())
      throw std::runtime_error("graph produced no output");
    std::cout << "stream=" << out->stream_id << " frame=" << out->frame_id << "\n";
    std::cout << "[OK] 012_build_a_custom_data_graph\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
