// tutorial_0011_mpk_yolov8_pipeline.cpp
// Story: the canonical YOLOv8 pipeline (Preprocess -> MLA -> BoxDecode).
// What you learn:
// - Model wires model stages into NodeGroups.
// - SimaBoxDecode converts MLA output into detections.
// - The canonical production pipeline shape matches sync_yolov8_test.cpp.

#include "neat/session.h"
#include "neat/models.h"
#include "neat/nodes.h"
#include "gst/GstHelpers.h"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <array>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
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
    if (!cur.has_parent_path())
      break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

std::filesystem::path find_asset_root() {
  namespace fs = std::filesystem;
  if (const char* env = std::getenv("SIMA_NEAT_TUTORIAL_ASSETS")) {
    fs::path p{env};
    if (fs::exists(p))
      return p;
  }
  for (const fs::path& p : {
           fs::path{"/usr/share/sima-neat/tutorials/assets"},
           fs::path{"/neat-resources/core-src/tutorials/assets"},
       }) {
    if (fs::exists(p))
      return p;
  }
  return find_repo_root() / "tutorials" / "assets";
}

} // namespace

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>] [--image <path>] [--print-gst]\n";
  print_common_flags(std::cout);
  std::cout << "  --mpk <path>         Path to YOLOv8 MPK tar.gz (default: search tmp/)\n";
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

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const fs::path root = find_repo_root();

    std::string mpk_arg;
    fs::path mpk_path =
        get_arg(argc, argv, "--mpk", mpk_arg) ? fs::path(mpk_arg) : find_default_mpk(root);
    if (mpk_path.empty() || !fs::exists(mpk_path)) {
      return skip("missing YOLOv8 MPK (pass --mpk)");
    }

    std::string img_arg;
    fs::path image_path =
        get_arg(argc, argv, "--image", img_arg) ? fs::path(img_arg) : find_default_image();
    if (!fs::exists(image_path)) {
      return skip("missing image (pass --image)");
    }

    cv::Mat img_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
      return skip("failed to load image");
    }

    // Check required elements before attempting to run.
    if (!simaai::neat::element_exists("simaaiprocesscvu") ||
        !simaai::neat::element_exists("simaaiprocessmla") ||
        !simaai::neat::element_exists("simaaiboxdecode")) {
      return skip("missing SimaAI plugins (simaaiprocesscvu/mla/boxdecode)");
    }

    // 1) Load the model pack (MPK) with explicit model input bounds.
    simaai::neat::Model::Options model_opt;
    model_opt.input_max_width = img_bgr.cols;
    model_opt.input_max_height = img_bgr.rows;
    model_opt.input_max_depth = img_bgr.channels();
    simaai::neat::Model model(mpk_path.string(), model_opt);

    // 2) Assemble the canonical pipeline.
    simaai::neat::Session p;
    p.add(simaai::neat::nodes::Input());
    p.add(simaai::neat::nodes::groups::Preprocess(model));
    p.add(simaai::neat::nodes::groups::MLA(model));
    p.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", img_bgr.cols, img_bgr.rows,
                                             /*score=*/0.52f, /*nms=*/0.5f, /*topk=*/100));
    p.add(simaai::neat::nodes::Output());

    if (wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    // 3) Run one image and pull detections.
    const bool strict = (std::getenv("SIMA_RUN_TUTORIALS_FULL") != nullptr);
    try {
      auto run = p.build(img_bgr, simaai::neat::RunMode::Sync);
      simaai::neat::Sample out = run.push_and_pull(img_bgr, /*timeout_ms=*/2000);

      std::cout << "Output kind: " << static_cast<int>(out.kind) << "\n";
      if (out.kind == simaai::neat::SampleKind::Bundle) {
        std::cout << "Bundle fields: " << out.fields.size() << "\n";
      }
    } catch (const std::exception& e) {
      if (!strict) {
        return skip(std::string("runtime unavailable: ") + e.what());
      }
      throw;
    }

    std::cout << "[OK] tutorial_0011 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
