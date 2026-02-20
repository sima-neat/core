// tutorial_0024_hybrid_graph.cpp
// Story: build a small hybrid graph (pipeline + stage) and run it.
// What you learn:
// - Graph nodes can be pipeline-backed or stage-backed.
// - GraphSession compiles a hybrid runtime from a single DAG.
// - Stage executors run in-process and can stamp metadata.

#include "neat/graph.h"
#include "neat/nodes.h"
#include "neat/session.h"

#include "tutorial_common.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << "\n";
  sima_tutorial::print_common_flags(std::cout);
}

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

simaai::neat::Tensor make_rgb_tensor(int w, int h) {
  const int depth = 3;
  const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * depth;

  simaai::neat::Tensor t;
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, depth};
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::RGB, ""};
  t.storage = simaai::neat::make_cpu_owned_storage(bytes);
  t.strides_bytes = contiguous_strides_bytes(t.shape, 1);
  t.read_only = false;

  {
    auto mapping = t.map(simaai::neat::MapMode::Write);
    auto* ptr = static_cast<std::uint8_t*>(mapping.data);
    for (std::size_t i = 0; i < bytes; ++i) {
      ptr[i] = static_cast<std::uint8_t>(i % 255);
    }
  }

  t.read_only = true;
  return t;
}

simaai::neat::Sample make_sample() {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_rgb_tensor(4, 4);
  sample.frame_id = -1;  // StampFrameId will fill this in.
  sample.stream_id = ""; // StampFrameId will fill this in.
  return sample;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    // Build a tiny hybrid graph:
    // Pipeline (VideoConvert) -> Stage (StampFrameId) -> graph sink.
    simaai::neat::graph::Graph g;

    auto pipe = g.add(std::make_shared<simaai::neat::graph::nodes::PipelineNode>(
        simaai::neat::nodes::VideoConvert(), "convert"));
    auto stamp = g.add(simaai::neat::graph::nodes::StampFrameIdNode("stamp"));
    g.connect(pipe, stamp);

    std::cout << "[GraphPrinter] text\n";
    std::cout << simaai::neat::graph::GraphPrinter::to_text(g) << "\n\n";

    simaai::neat::graph::GraphSession session(std::move(g));
    simaai::neat::graph::GraphRun run = session.build();

    std::cout << "[GraphRun::describe]\n";
    std::cout << run.describe() << "\n";

    simaai::neat::Sample input = make_sample();
    sima_tutorial::require(run.push(pipe, input), "GraphRun::push failed");

    auto out = run.pull(stamp, 2000);
    if (!out.has_value()) {
      const std::string err = run.last_error();
      if (!err.empty()) {
        throw std::runtime_error(err);
      }
      throw std::runtime_error("GraphRun::pull timed out");
    }

    std::cout << "[Result] stream_id=" << out->stream_id << " frame_id=" << out->frame_id
              << " kind=" << static_cast<int>(out->kind) << "\n";

    run.stop();
    std::cout << "[OK] tutorial_0024 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
