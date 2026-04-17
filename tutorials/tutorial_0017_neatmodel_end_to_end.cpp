// tutorial_0017_neatmodel_end_to_end.cpp
// Story: simaai::neat::Model is the single entry point for MPK model packs.
// What you learn:
// - Constructing Model with Options (format/dims/naming/preproc overrides).
// - Sync run() vs async build(): differences and when to use each.
// - Building sessions and stages explicitly (preprocess/inference/postprocess).
// - Introspection: input_spec(), output_spec(), metadata().
// - SessionOptions for name isolation and graph integration.

#include "neat/session.h"
#include "neat/models.h"
#include "neat/nodes.h"
#include "gst/GstHelpers.h"


#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <utility>

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

bool wants_print_gst(int argc, char** argv) {
  return has_flag(argc, argv, "--print-gst");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

int skip(const std::string& reason) {
  std::cout << "SKIP: " << reason << "\n";
  return 0;
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

std::filesystem::path find_asset_root() {
  namespace fs = std::filesystem;
  if (const char* env = std::getenv("SIMA_NEAT_TUTORIAL_ASSETS")) {
    fs::path p{env};
    if (fs::exists(p)) return p;
  }
  for (const fs::path& p : {
           fs::path{"/usr/share/sima-neat/tutorials/assets"},
           fs::path{"/neat-resources/core-src/tutorials/assets"},
       }) {
    if (fs::exists(p)) return p;
  }
  return find_repo_root() / "tutorials" / "assets";
}

} // namespace

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>] [--image <path>]\n";
  print_common_flags(std::cout);
  std::cout << "  --mpk <path>         MPK tar.gz (default: search tmp/)\n";
  std::cout << "  --image <path>       Input image (default: shipped tutorial sample)\n";
}

fs::path find_default_mpk(const fs::path& root) {
  const fs::path c1 = root / "tmp" / "yolo_v8s_mpk.tar.gz";
  const fs::path c2 = root / "tmp" / "yolov8s_mpk.tar.gz";
  if (fs::exists(c1))
    return c1;
  if (fs::exists(c2))
    return c2;
  return {};
}

fs::path find_default_image() {
  return find_asset_root() / "ilena_488.jpg";
}

const char* dtype_name(simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    return "UInt8";
  case simaai::neat::TensorDType::Int8:
    return "Int8";
  case simaai::neat::TensorDType::UInt16:
    return "UInt16";
  case simaai::neat::TensorDType::Int16:
    return "Int16";
  case simaai::neat::TensorDType::Int32:
    return "Int32";
  case simaai::neat::TensorDType::BFloat16:
    return "BFloat16";
  case simaai::neat::TensorDType::Float32:
    return "Float32";
  case simaai::neat::TensorDType::Float64:
    return "Float64";
  default:
    return "Unknown";
  }
}

const char* pixel_format_name(simaai::neat::ImageSpec::PixelFormat fmt) {
  switch (fmt) {
  case simaai::neat::ImageSpec::PixelFormat::RGB:
    return "RGB";
  case simaai::neat::ImageSpec::PixelFormat::BGR:
    return "BGR";
  case simaai::neat::ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case simaai::neat::ImageSpec::PixelFormat::NV12:
    return "NV12";
  case simaai::neat::ImageSpec::PixelFormat::I420:
    return "I420";
  default:
    return "Unknown";
  }
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

void print_spec(const char* label, const simaai::neat::TensorSpec& spec) {
  std::cout << label << ": rank=" << spec.rank;
  if (!spec.shape.empty()) {
    std::cout << " shape=" << shape_to_string(spec.shape);
  }
  if (!spec.dtypes.empty()) {
    std::cout << " dtypes=";
    for (size_t i = 0; i < spec.dtypes.size(); ++i) {
      if (i > 0)
        std::cout << ",";
      std::cout << dtype_name(spec.dtypes[i]);
    }
  }
  if (spec.image_format.has_value()) {
    std::cout << " image_format=" << pixel_format_name(*spec.image_format);
  }
  std::cout << "\n";
}

void print_metadata(const simaai::neat::Model& model) {
  const auto meta = model.metadata();
  if (meta.empty()) {
    std::cout << "metadata: (none)\n";
    return;
  }
  std::cout << "metadata:\n";
  for (const auto& kv : meta) {
    std::cout << "  " << kv.first << " = " << kv.second << "\n";
  }
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
    fs::path mpk_path = get_arg(argc, argv, "--mpk", mpk_arg)
                            ? fs::path(mpk_arg)
                            : find_default_mpk(root);
    if (mpk_path.empty() || !fs::exists(mpk_path)) {
      return skip("missing MPK (pass --mpk)");
    }

    std::string img_arg;
    fs::path image_path = get_arg(argc, argv, "--image", img_arg)
                              ? fs::path(img_arg)
                              : find_default_image();
    if (!fs::exists(image_path)) {
      return skip("missing image (pass --image)");
    }

    cv::Mat img_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
      return skip("failed to load image");
    }

    // Check required elements before attempting to run.
    if (!simaai::neat::element_exists("simaaiprocesscvu") ||
        !simaai::neat::element_exists("simaaiprocessmla")) {
      return skip("missing SimaAI plugins (simaaiprocesscvu/mla)");
    }

    // 1) Construct Model with Options.
    // Options let you define input format/dims and override preproc/postproc defaults.
    simaai::neat::Model::Options opt;
    opt.format = "BGR"; // matches img_bgr
    opt.input_max_width = img_bgr.cols;
    opt.input_max_height = img_bgr.rows;
    opt.input_max_depth = 3;
    opt.upstream_name = "decoder"; // upstream element name for preproc
    opt.name_suffix = "_demo";     // avoid name collisions when using multiple models

    simaai::neat::Model model(mpk_path.string(), opt);

    // 2) Introspection: input/output specs and metadata.
    print_spec("input_spec", model.input_spec());
    print_spec("output_spec", model.output_spec());
    print_metadata(model);

    // 3) Build sessions and stages explicitly.
    // session() builds: AppSrc -> Preproc/QuantTess -> Infer -> Postproc -> Output
    // preprocess()/inference()/postprocess() expose stage NodeGroups for composition.
    simaai::neat::Session pipeline_full;
    pipeline_full.add(model.session());

    simaai::neat::Session pipeline_staged;
    pipeline_staged.add(model.preprocess());
    pipeline_staged.add(model.inference());
    pipeline_staged.add(model.postprocess());
    pipeline_staged.add(simaai::neat::nodes::Output());

    if (wants_print_gst(argc, argv)) {
      std::cout << "[pipeline] full:\n" << pipeline_full.describe_backend() << "\n";
      std::cout << "[pipeline] staged:\n" << pipeline_staged.describe_backend() << "\n";

      // Advanced: raw fragments (useful when integrating into a custom graph).
      std::cout << "[fragment] infer gst:\n"
                << model.backend_fragment(simaai::neat::Model::Stage::Inference) << "\n";
      return 0;
    }

    // 4) Sync run() vs async build().
    // Sync run():
    // - convenient for single inputs / request-response
    // - internally uses num-buffers=1 for CVU/MLA stages
    // - caches a sync runner per Model instance
    // Async build():
    // - best for throughput, multiple inputs
    // - returns Runner with push/pull and run helpers

    const bool strict = (std::getenv("SIMA_RUN_TUTORIALS_FULL") != nullptr);
    if (!strict) {
      std::cout << "[OK] tutorial_0017 complete (set SIMA_RUN_TUTORIALS_FULL=1 to run)\n";
      return 0;
    }

    // Sync path: Model::run(cv::Mat)
    std::cout << "[sync] run()\n";
    simaai::neat::Sample sync_out = model.run(img_bgr);
    std::cout << "[sync] output kind=" << static_cast<int>(sync_out.kind) << "\n";

    // Async path: Model::build() -> Runner
    std::cout << "[async] build() + push/pull\n";
    auto runner = model.build(img_bgr);
    simaai::neat::Tensor input = simaai::neat::from_cv_mat(
        img_bgr, simaai::neat::ImageSpec::PixelFormat::BGR, /*read_only=*/true);
    if (!runner.push(input)) {
      throw std::runtime_error("Runner::push failed");
    }
    auto async_out = runner.pull(/*timeout_ms=*/2000);
    if (!async_out.has_value()) {
      throw std::runtime_error("Runner::pull returned no outputs");
    }
    runner.close();

    // 5) SessionOptions for name isolation / integration.
    // Use this when building multiple models or when upstream names differ.
    simaai::neat::Model::SessionOptions popt;
    popt.upstream_name = "decoder_cam0";
    popt.name_suffix = "_cam0";
    popt.buffer_name = "decoder_cam0";
    popt.include_appsrc = false; // use external appsrc / upstream pipeline
    popt.include_appsink = true;

    simaai::neat::Session pipeline_custom;
    pipeline_custom.add(model.session(popt));

    std::cout << "[OK] tutorial_0017 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
