#include "neat/graph.h"
#include "neat/models.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <array>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
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
  if (!detail.empty()) std::cout << " (" << detail << ")";
  std::cout << "\n";
  if (!cond) throw std::runtime_error("check failed: " + name);
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
      if (kv.first == key) { found = true; break; }
    }
    if (!found) throw std::invalid_argument(std::string("missing signature key: ") + key);
  }
  std::cout << "SIGNATURE {";
  bool first = true;
  for (const auto& kv : values) {
    if (!first) std::cout << ",";
    std::cout << kv.first << "=" << kv.second;
    first = false;
  }
  std::cout << "}\n";
}

std::filesystem::path find_repo_root() {
  namespace fs = std::filesystem;
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "include") &&
        fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path()) break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

std::filesystem::path first_existing(std::initializer_list<std::filesystem::path> candidates) {
  for (const auto& c : candidates) {
    if (std::filesystem::exists(c)) return c;
  }
  return {};
}

std::filesystem::path default_yolo_mpk() {
  namespace fs = std::filesystem;
  const fs::path root = find_repo_root();
  return first_existing({
      root / "tmp" / "yolo_v8s_mpk.tar.gz",
      root / "tmp" / "yolov8s_mpk.tar.gz",
      root / "tmp" / "yolo_mpk.tar.gz",
  });
}

std::filesystem::path default_resnet_mpk() {
  namespace fs = std::filesystem;
  const fs::path root = find_repo_root();
  return first_existing({
      root / "tmp" / "resnet_50_mpk.tar.gz",
      root / "tmp" / "resnet50_mpk.tar.gz",
  });
}

} // namespace

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
  const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                            static_cast<std::size_t>(d) * sizeof(float);
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
  // CORE LOGIC
  auto model = std::make_shared<simaai::neat::Model>(mpk_path.string());

  simaai::neat::graph::nodes::StageModelExecutorOptions opt;
  opt.model = model;
  opt.do_preproc = false;
  opt.do_mla = false;
  opt.do_boxdecode = false;

  simaai::neat::graph::Graph g;
  const auto node_id =
      g.add(simaai::neat::graph::nodes::StageModelExecutorNode(opt, "stage_model"));

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

  check("graph_push", run.push(node_id, in), "sample accepted by stage-model node");
  auto out = run.pull(node_id, 2000);
  check("graph_pull", out.has_value(), "stage-model node produced output");
  run.stop();
  // END CORE LOGIC
  return {"model_stage", *out};
}

std::pair<std::string, simaai::neat::Sample> run_stage_fallback() {
  // CORE LOGIC
  simaai::neat::graph::Graph g;
  const auto node_id = g.add(simaai::neat::graph::nodes::StampFrameIdNode("stamp"));

  simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(g)).build();

  simaai::neat::Sample in;
  in.kind = simaai::neat::SampleKind::Tensor;
  in.tensor = make_fp32_tensor(8, 8, 3);
  in.frame_id = 1;
  in.stream_id = "model";

  check("graph_push", run.push(node_id, in), "sample accepted by fallback stage");
  auto out = run.pull(node_id, 2000);
  check("graph_pull", out.has_value(), "fallback stage produced output");
  run.stop();
  // END CORE LOGIC
  return {"stage_fallback", *out};
}

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>]\n";
  print_common_flags(std::cout);
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const fs::path root = find_repo_root();
    std::string mpk_arg;
    const fs::path mpk_path = get_arg(argc, argv, "--mpk", mpk_arg)
                                  ? fs::path(mpk_arg)
                                  : first_existing({
                                        default_yolo_mpk(),
                                        default_resnet_mpk(),
                                    });

    step("input_contract", "model-hybrid stage consumes tensor-shaped samples");
    step("run_mode_choice",
                      "use stage-model node when MPK exists, else fallback stage");
    why("understand the contract first: inputs, run mode, and outputs");
    tradeoff(
        "prefer deterministic samples and stable contracts over production realism");
    failure_mode(
        "runtime/plugin issues should degrade to runtime_fallback without losing observability");
    interpret_output(
        "use CHECK markers plus SIGNATURE fields to validate behavior and parity");

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

    step("output_interpretation",
                      "inspect output rank to reason about stage boundaries");
    check("output_kind_tensor", out.kind == simaai::neat::SampleKind::Tensor,
                       "hybrid stage should emit tensor sample");
    check("output_tensor_present", out.tensor.has_value(),
                       "tensor payload must exist");

    print_signature({
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
