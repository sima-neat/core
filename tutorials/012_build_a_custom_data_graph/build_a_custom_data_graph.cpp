// Compose a minimal public Neat Graph: named Input -> named Output.
//
// Usage:
//   tutorial_012_build_a_custom_data_graph

#include "neat.h"

#include <cstdint>
#include <iostream>
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
  s.frame_id = 42;
  s.pts_ns = 123456789;
  return s;
}

} // namespace

int main() {
  try {
    // CORE LOGIC
    // STEP compose-graph
    // `Graph` is the public composition type. Input("image") declares the name
    // used by Run::push("image", ...). Output("out") declares the name used by
    // Run::pull("out", ...).
    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::Input("image"));
    graph.add(simaai::neat::nodes::Output("out"));
    // END STEP
    // STEP connect-endpoints
    graph.connect("image", "out");

    std::cout << graph.describe() << "\n";
    // END STEP

    // STEP build-and-push
    simaai::neat::Run run = graph.build();
    if (!run.push("image", make_sample())) {
      throw std::runtime_error("push failed: " + run.last_error());
    }
    // END STEP
    // STEP pull-and-verify
    auto out = run.pull("out", /*timeout_ms=*/2000);
    run.close();
    // END STEP
    // END CORE LOGIC

    if (!out.has_value())
      throw std::runtime_error("graph produced no output");
    std::cout << "stream=" << out->stream_id << " frame=" << out->frame_id
              << " pts_ns=" << out->pts_ns << "\n";
    std::cout << "[OK] 012_build_a_custom_data_graph\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
