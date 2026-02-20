#include "neat/graph.h"
#include "neat/models.h"
#include "common/cpp_utils.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

namespace fs = std::filesystem;

namespace {

// Why: this chapter contrasts model-backed graph stages with deterministic fallback flow.
// Why: learners should see identical checkpoint output even when model assets are unavailable.

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
  const std::size_t bytes =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(d) *
      sizeof(float);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0, map.size_bytes);
  }

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

std::pair<std::string, simaai::neat::Sample> run_model_stage(const fs::path& mpk_path) {
  auto model = std::make_shared<simaai::neat::Model>(mpk_path.string());

  simaai::neat::graph::nodes::StageModelExecutorOptions opt;
  opt.model = model;
  opt.do_preproc = false;
  opt.do_mla = false;
  opt.do_boxdecode = false;

  simaai::neat::graph::Graph g;
  const auto node_id = g.add(simaai::neat::graph::nodes::StageModelExecutorNode(opt, "stage_model"));

  simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(g)).build();

  simaai::neat::InputOptions tensor_opt = model->input_appsrc_options(true);
  int w = (tensor_opt.width > 0) ? tensor_opt.width : tensor_opt.max_width;
  int h = (tensor_opt.height > 0) ? tensor_opt.height : tensor_opt.max_height;
  int d = (tensor_opt.depth > 0) ? tensor_opt.depth : tensor_opt.max_depth;
  if (w <= 0 || h <= 0 || d <= 0) {
    w = 8;
    h = 8;
    d = 3;
  }

  simaai::neat::Sample in;
  in.kind = simaai::neat::SampleKind::Tensor;
  in.tensor = make_fp32_tensor(w, h, d);
  in.frame_id = 1;
  in.stream_id = "model";

  tutorial_v2::check("graph_push", run.push(node_id, in), "sample accepted by stage-model node");
  auto out = run.pull(node_id, 2000);
  tutorial_v2::check("graph_pull", out.has_value(), "stage-model node produced output");
  run.stop();
  return {"model_stage", *out};
}

std::pair<std::string, simaai::neat::Sample> run_stage_fallback() {
  simaai::neat::graph::Graph g;
  const auto node_id = g.add(simaai::neat::graph::nodes::StampFrameIdNode("stamp"));

  simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(g)).build();

  simaai::neat::Sample in;
  in.kind = simaai::neat::SampleKind::Tensor;
  in.tensor = make_fp32_tensor(8, 8, 3);
  in.frame_id = 1;
  in.stream_id = "model";

  tutorial_v2::check("graph_push", run.push(node_id, in), "sample accepted by fallback stage");
  auto out = run.pull(node_id, 2000);
  tutorial_v2::check("graph_pull", out.has_value(), "fallback stage produced output");
  run.stop();
  return {"stage_fallback", *out};
}

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>]\n";
  tutorial_v2::print_common_flags(std::cout);
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (tutorial_v2::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const fs::path root = tutorial_v2::find_repo_root();
    std::string mpk_arg;
    const fs::path mpk_path = tutorial_v2::get_arg(argc, argv, "--mpk", mpk_arg)
                                  ? fs::path(mpk_arg)
                                  : tutorial_v2::first_existing({
                                        tutorial_v2::default_yolo_mpk(root),
                                        tutorial_v2::default_resnet_mpk(root),
                                    });

    tutorial_v2::step("input_contract", "model-hybrid stage consumes tensor-shaped samples");
    tutorial_v2::step("run_mode_choice", "use stage-model node when MPK exists, else fallback stage");
    tutorial_v2::why("understand the contract first: inputs, run mode, and outputs");
    tutorial_v2::tradeoff("prefer deterministic samples and stable contracts over production realism");
    tutorial_v2::failure_mode("runtime/plugin issues should degrade to runtime_fallback without losing observability");
    tutorial_v2::interpret_output("use CHECK markers plus SIGNATURE fields to validate behavior and parity");

    std::string flow = "stage_fallback";
    simaai::neat::Sample out;
    if (!mpk_path.empty() && fs::exists(mpk_path)) {
      try {
        std::tie(flow, out) = run_model_stage(mpk_path);
      } catch (const std::exception& e) {
        std::cout << "model branch fallback reason: " << e.what() << "\n";
        std::tie(flow, out) = run_stage_fallback();
      }
    } else {
      std::tie(flow, out) = run_stage_fallback();
    }

    tutorial_v2::step("output_interpretation", "inspect output rank to reason about stage boundaries");
    tutorial_v2::check("output_kind_tensor", out.kind == simaai::neat::SampleKind::Tensor,
                       "hybrid stage should emit tensor sample");
    tutorial_v2::check("output_tensor_present", out.tensor.has_value(), "tensor payload must exist");

    tutorial_v2::print_signature({
        {"tutorial", "015"},
        {"lang", "cpp"},
        {"flow", flow},
        {"run_mode", "graph_sync_pull"},
        {"output_kind", std::to_string(static_cast<int>(out.kind))},
        {"tensor_rank", out.tensor.has_value() ? std::to_string(out.tensor->shape.size()) : "-1"},
        {"field_count", std::to_string(out.fields.size())},
    });

    std::cout << "Output rank: " << (out.tensor.has_value() ? out.tensor->shape.size() : 0) << "\n";
    std::cout << "[OK] 015_graph_model_hybrid\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
