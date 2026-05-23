#include "pipeline/Graph.h"
#include "model/Model.h"
#include "nodes/common/Output.h"
#include "gst/GstHelpers.h"

#include "asset_utils.h"
#include "cli_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* kGoldfishUrl =
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
    "n01443537_goldfish.JPEG";
constexpr int kGoldfishId = 1;
constexpr int kInferWidth = 224;
constexpr int kInferHeight = 224;

struct BenchResult {
  int top1 = -1;
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double fps = 0.0;
};

static void usage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " --goldfish [--model <model.tar.gz>]\n"
            << "  " << prog << " --image <path> [--model <model.tar.gz>]\n"
            << "\n"
            << "Options:\n"
            << "  --goldfish             Download ImageNet goldfish sample and run comparison.\n"
            << "  --goldfish-url <url>   Override goldfish image URL.\n"
            << "  --image <path>         Input image path.\n"
            << "  --model <path>         Model pack tar.gz path.\n"
            << "  --expect-id <int>      Expected top-1 class id.\n"
            << "  --warmup <int>         Warmup iterations per route (default: 10).\n"
            << "  --iters <int>          Measured iterations per route (default: 80).\n"
            << "  --print-pipeline       Print baseline/argmax pipeline strings.\n";
}

static cv::Mat load_rgb_resized(const std::string& image_path, int w, int h) {
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }
  if (w > 0 && h > 0 && (bgr.cols != w || bgr.rows != h)) {
    cv::resize(bgr, bgr, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

static simaai::neat::Model make_resnet_model(const std::string& tar_gz, bool mla_only) {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  if (mla_only) {
    opt.inference_terminal.last_plugin_id = "processmla";
  }
  return simaai::neat::Model(tar_gz, opt);
}

static bool pipeline_has_fused_classifier_post(const std::string& pipeline) {
  if (pipeline.find("detessdequant") != std::string::npos ||
      pipeline.find("detess_dequant") != std::string::npos) {
    return true;
  }
  const bool has_fused_processcvu = pipeline.find("! neatprocesscvu ") != std::string::npos;
  const bool has_fused_stage_id = pipeline.find("stage-id=dequantize_") != std::string::npos ||
                                  pipeline.find(" name=dequantize_") != std::string::npos ||
                                  pipeline.find("stage-id=detessdequant_") != std::string::npos ||
                                  pipeline.find(" name=detessdequant_") != std::string::npos;
  return has_fused_processcvu && has_fused_stage_id;
}

static void add_mla_terminal_argmax_route(simaai::neat::Graph& graph,
                                          const simaai::neat::Model& model) {
  simaai::neat::InputOptions in_opt = model.input_appsrc_options(/*tensor_mode=*/false);
  graph.add(simaai::neat::nodes::Input(in_opt));
  const auto pre = model.preprocess();
  if (!pre.describe_backend(false).empty()) {
    graph.add(pre);
  }
  const auto infer = model.inference();
  require(!infer.describe_backend(false).empty(),
          "argmax: model.inference() returned an empty Graph");
  graph.add(infer);
  graph.custom("neatargmax name=neatargmax_1 axis=-1 keepdims=false num-threads=0 silent=true");
  graph.add(simaai::neat::nodes::Output());
}

static simaai::neat::Tensor require_tensor(const simaai::neat::TensorList& out,
                                           const std::string& label) {
  if (out.empty()) {
    throw std::runtime_error(label + ": expected tensor output");
  }
  return out.front();
}

static std::vector<float> scores_from_tensor(const simaai::neat::Tensor& t,
                                             const std::string& label) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error(label + ": expected Float32 output tensor");
  }
  if (!t.is_dense()) {
    throw std::runtime_error(label + ": expected dense output tensor");
  }
  simaai::neat::Tensor cpu = t.clone();
  simaai::neat::Mapping map = cpu.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes < sizeof(float)) {
    throw std::runtime_error(label + ": empty output tensor");
  }
  if ((map.size_bytes % sizeof(float)) != 0U) {
    throw std::runtime_error(label + ": output tensor size is not float-aligned");
  }
  const std::size_t elems = map.size_bytes / sizeof(float);
  std::vector<float> out(elems, 0.0f);
  std::memcpy(out.data(), map.data, map.size_bytes);
  if (out.size() > 1000U) {
    out.resize(1000U);
  }
  return out;
}

static int top1_from_scores(const std::vector<float>& scores, const std::string& label) {
  if (scores.empty()) {
    throw std::runtime_error(label + ": scores are empty");
  }
  // Deliberately model the baseline as a simple CPU top-k postprocess rather
  // than the best possible top-1 reduction.  Real application code commonly
  // materializes class indices for top-k display/thresholding, so use a stable
  // index sort here.  The argmax route replaces this CPU postprocess with the
  // pipeline neatargmax stage, so include this work in the measured baseline.
  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
    const float av = scores[static_cast<std::size_t>(a)];
    const float bv = scores[static_cast<std::size_t>(b)];
    if (av == bv) {
      return a < b;
    }
    return av > bv;
  });
  return indices.front();
}

static int top1_from_int32_tensor(const simaai::neat::Tensor& t, const std::string& label) {
  if (t.dtype != simaai::neat::TensorDType::Int32) {
    throw std::runtime_error(label + ": expected Int32 output tensor");
  }
  if (!t.is_dense()) {
    throw std::runtime_error(label + ": expected dense output tensor");
  }
  simaai::neat::Tensor cpu = t.clone();
  simaai::neat::Mapping map = cpu.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes < sizeof(std::int32_t)) {
    throw std::runtime_error(label + ": empty output tensor");
  }
  if ((map.size_bytes % sizeof(std::int32_t)) != 0U) {
    throw std::runtime_error(label + ": output tensor size is not int32-aligned");
  }
  const auto* vals = static_cast<const std::int32_t*>(map.data);
  return static_cast<int>(vals[0]);
}

static double percentile_ms(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  p = std::clamp(p, 0.0, 1.0);
  const std::size_t idx =
      static_cast<std::size_t>(std::llround(p * static_cast<double>(values.size() - 1U)));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(idx), values.end());
  return values[idx];
}

template <typename ExtractTop1>
static BenchResult run_bench(simaai::neat::Graph& graph, const cv::Mat& rgb, int warmup, int iters,
                             const std::string& label, ExtractTop1 extract_top1) {
  for (int i = 0; i < warmup; ++i) {
    (void)graph.run(std::vector<cv::Mat>{rgb});
  }

  std::vector<double> per_iter_ms;
  per_iter_ms.reserve(static_cast<std::size_t>(iters));
  int fixed_top1 = -1;

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    const auto s0 = std::chrono::steady_clock::now();
    auto out = graph.run(std::vector<cv::Mat>{rgb});
    const int top1 = extract_top1(require_tensor(out, label));
    const auto s1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(s1 - s0).count();
    per_iter_ms.push_back(ms);

    if (i == 0) {
      fixed_top1 = top1;
    } else {
      require(top1 == fixed_top1,
              label + ": unstable top1 across iterations first=" + std::to_string(fixed_top1) +
                  " got=" + std::to_string(top1) + " at_iter=" + std::to_string(i));
    }
  }
  const auto t1 = std::chrono::steady_clock::now();

  const double total_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
  double sum_ms = std::accumulate(per_iter_ms.begin(), per_iter_ms.end(), 0.0);
  BenchResult out;
  out.top1 = fixed_top1;
  out.avg_ms = (iters > 0) ? (sum_ms / static_cast<double>(iters)) : 0.0;
  out.p50_ms = percentile_ms(per_iter_ms, 0.50);
  out.p95_ms = percentile_ms(per_iter_ms, 0.95);
  out.fps = (total_ms > 0.0) ? (1000.0 * static_cast<double>(iters) / total_ms) : 0.0;
  return out;
}

} // namespace

int main(int argc, char** argv) {
  try {
    try {
      if (!simaai::neat::element_exists("neatargmax")) {
        return skip_long_test("missing neatargmax plugin");
      }
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      if (msg.find("Allocator symbol already loaded from unexpected path") != std::string::npos) {
        return skip_long_test(msg);
      }
      throw;
    }

    bool use_goldfish = sima_test::has_flag(argc, argv, "--goldfish") || (argc <= 1);
    const bool print_pipeline = sima_test::has_flag(argc, argv, "--print-pipeline");

    std::string image_path;
    std::string model_tar;
    std::string goldfish_url = kGoldfishUrl;
    int expected_id = -1;
    bool have_expected = false;
    int warmup = 10;
    int iters = 80;

    std::string tmp;
    if (sima_test::get_arg(argc, argv, "--image", tmp))
      image_path = tmp;
    if (sima_test::get_arg(argc, argv, "--model", tmp))
      model_tar = tmp;
    if (sima_test::get_arg(argc, argv, "--goldfish-url", tmp))
      goldfish_url = tmp;
    if (sima_test::parse_int_arg(argc, argv, "--expect-id", expected_id)) {
      have_expected = true;
    }
    sima_test::parse_int_arg(argc, argv, "--warmup", warmup);
    sima_test::parse_int_arg(argc, argv, "--iters", iters);

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--image" || arg == "--model" || arg == "--goldfish-url" || arg == "--expect-id" ||
          arg == "--warmup" || arg == "--iters") {
        ++i;
        continue;
      }
      if (arg == "--goldfish" || arg == "--print-pipeline") {
        continue;
      }
      if (!arg.empty() && arg[0] == '-') {
        usage(argv[0]);
        return 2;
      }
      positional.push_back(arg);
    }
    if (image_path.empty() && !use_goldfish && !positional.empty()) {
      image_path = positional[0];
    }
    if (model_tar.empty() && positional.size() >= 2) {
      model_tar = positional[1];
    }

    warmup = std::max(0, warmup);
    iters = std::max(1, iters);

    if (use_goldfish) {
      if (!have_expected) {
        expected_id = kGoldfishId;
        have_expected = true;
      }
      const fs::path out_path = sima_test::default_goldfish_path();
      if (!sima_test::download_file(goldfish_url, out_path)) {
        std::cerr << "Failed to download goldfish image.\n";
        std::cerr << "URL was: " << goldfish_url << "\n";
        return 3;
      }
      image_path = out_path.string();
      std::cout << "Using goldfish image: " << image_path << "\n";
    }

    if (image_path.empty()) {
      usage(argv[0]);
      return 2;
    }
    require(fs::exists(image_path), "Image path missing: " + image_path);

    if (model_tar.empty()) {
      model_tar = sima_test::resolve_resnet50_tar();
      if (model_tar.empty()) {
        return skip_long_test(
            "ResNet50 model pack not found (set SIMA_MODEL_TAR or SIMA_RESNET50_TAR)");
      }
    }
    if (!fs::exists(model_tar)) {
      std::cerr << "Model tar.gz missing: " << model_tar << "\n";
      return 3;
    }

    cv::Mat rgb = load_rgb_resized(image_path, kInferWidth, kInferHeight);

    auto baseline_model = make_resnet_model(model_tar, /*mla_only=*/false);
    auto mla_model = make_resnet_model(model_tar, /*mla_only=*/true);

    simaai::neat::Model::RouteOptions baseline_route_opt;
    baseline_route_opt.include_input = true;
    baseline_route_opt.include_output = true;
    simaai::neat::Graph baseline_graph;
    baseline_graph.add(baseline_model.graph(baseline_route_opt));

    simaai::neat::Graph argmax_graph;
    add_mla_terminal_argmax_route(argmax_graph, mla_model);

    const std::string baseline_pipeline = baseline_graph.describe_backend();
    const std::string argmax_pipeline = argmax_graph.describe_backend();

    if (print_pipeline) {
      std::cout << "[baseline] pipeline:\n" << baseline_pipeline << "\n";
      std::cout << "[argmax] pipeline:\n" << argmax_pipeline << "\n";
    }

    const bool baseline_has_detess = pipeline_has_fused_classifier_post(baseline_pipeline);
    require(baseline_has_detess,
            "baseline pipeline is missing detessdequant stage; cannot validate replacement");

    const bool argmax_has_detess = pipeline_has_fused_classifier_post(argmax_pipeline);
    require(!argmax_has_detess,
            "argmax pipeline still contains detessdequant; expected MLA terminal replacement");

    const bool argmax_has_mla = argmax_pipeline.find("neatprocessmla") != std::string::npos ||
                                argmax_pipeline.find("processmla") != std::string::npos;
    require(argmax_has_mla, "argmax pipeline missing MLA stage");
    require(argmax_pipeline.find("neatargmax") != std::string::npos,
            "argmax pipeline missing neatargmax stage");

    const BenchResult baseline =
        run_bench(baseline_graph, rgb, warmup, iters, "baseline", [](const auto& t) {
          auto scores = scores_from_tensor(t, "baseline");
          return top1_from_scores(scores, "baseline");
        });

    const BenchResult argmax =
        run_bench(argmax_graph, rgb, warmup, iters, "argmax",
                  [](const auto& t) { return top1_from_int32_tensor(t, "argmax"); });

    std::cout << "[baseline] top1=" << baseline.top1 << " avg_ms=" << baseline.avg_ms
              << " p50_ms=" << baseline.p50_ms << " p95_ms=" << baseline.p95_ms
              << " fps=" << baseline.fps << "\n";
    std::cout << "[argmax]   top1=" << argmax.top1 << " avg_ms=" << argmax.avg_ms
              << " p50_ms=" << argmax.p50_ms << " p95_ms=" << argmax.p95_ms << " fps=" << argmax.fps
              << "\n";

    require(argmax.top1 == baseline.top1,
            "top1 mismatch baseline=" + std::to_string(baseline.top1) +
                " argmax=" + std::to_string(argmax.top1));
    if (have_expected) {
      require(baseline.top1 == expected_id,
              "baseline top1 mismatch expected=" + std::to_string(expected_id) +
                  " got=" + std::to_string(baseline.top1));
      require(argmax.top1 == expected_id,
              "argmax top1 mismatch expected=" + std::to_string(expected_id) +
                  " got=" + std::to_string(argmax.top1));
    }

    require(argmax.avg_ms > 0.0, "argmax average latency is invalid");
    const double speedup = baseline.avg_ms / argmax.avg_ms;
    std::cout << "[speedup] baseline_over_argmax=" << speedup << " (informational)\n";

    std::cout << "[OK] resnet50_neatargmax_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
