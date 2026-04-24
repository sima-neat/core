// Hybrid graph: a Model lives inside a Graph via StageModelExecutorNode.
//
// Usage:
//   tutorial_v2_015_graph_model_hybrid --mpk /path/to/model.tar.gz

#include "neat/graph.h"
#include "neat/models.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
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

simaai::neat::Tensor make_fp32_tensor(int w, int h, int d) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * d * sizeof(float);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  std::memset(map.data, 0, map.size_bytes);

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, d};
  t.strides_bytes = contiguous_strides_bytes(t.shape, static_cast<int64_t>(sizeof(float)));
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  return t;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string mpk;
    if (!get_arg(argc, argv, "--mpk", mpk)) {
      std::cerr << "Usage: tutorial_v2_015_graph_model_hybrid --mpk <path>\n";
      return 1;
    }

    // CORE LOGIC
    // Wrap a Model as a StageModelExecutorNode so it participates in a Graph.
    auto model = std::make_shared<simaai::neat::Model>(mpk);

    simaai::neat::graph::nodes::StageModelExecutorOptions opt;
    opt.model = model;
    opt.do_preproc = false;
    opt.do_mla = false;
    opt.do_boxdecode = false;

    simaai::neat::graph::Graph g;
    const auto node_id =
        g.add(simaai::neat::graph::nodes::StageModelExecutorNode(opt, "stage_model"));

    simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(g)).build();

    // Size the input tensor from the model's declared input appsrc options.
    simaai::neat::InputOptions tensor_opt = model->input_appsrc_options(true);
    int w = (tensor_opt.width > 0) ? tensor_opt.width : tensor_opt.max_width;
    int h = (tensor_opt.height > 0) ? tensor_opt.height : tensor_opt.max_height;
    int d = (tensor_opt.depth > 0) ? tensor_opt.depth : tensor_opt.max_depth;

    simaai::neat::Sample in;
    in.kind = simaai::neat::SampleKind::Tensor;
    in.tensor = make_fp32_tensor(w, h, d);
    in.frame_id = 1;
    in.stream_id = "model";

    run.push(node_id, in);
    auto out = run.pull(node_id, /*timeout_ms=*/2000);
    run.stop();
    // END CORE LOGIC

    if (!out.has_value())
      throw std::runtime_error("graph produced no output");
    if (!out->tensor.has_value())
      throw std::runtime_error("graph output missing tensor");
    std::cout << "output_rank=" << out->tensor->shape.size() << "\n";
    std::cout << "[OK] 015_embed_model_inside_graph\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
