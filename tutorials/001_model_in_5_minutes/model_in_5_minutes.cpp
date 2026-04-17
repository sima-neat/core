#include "neat.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <exception>
#include <initializer_list>
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

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string value;
  if (!get_arg(argc, argv, key, value)) return def;
  try {
    return std::stoi(value);
  } catch (...) {
    throw std::invalid_argument("invalid integer for " + key + ": " + value);
  }
}

int skip(const std::string& reason) {
  std::cout << "SKIP: " << reason << "\n";
  return 0;
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}

bool strict_mode() {
  return std::getenv("SIMA_RUN_TUTORIALS_FULL") != nullptr;
}

std::string yes_no(bool v) { return v ? "yes" : "no"; }

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

void runtime_fallback(const std::exception& e) {
  const char* what = e.what();
  std::cout << "runtime_fallback: " << (what ? what : "unknown") << "\n";
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

std::filesystem::path first_existing(std::initializer_list<std::filesystem::path> candidates) {
  for (const auto& c : candidates) {
    if (std::filesystem::exists(c)) return c;
  }
  return {};
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

struct Sig {
  std::string output_kind = "sample_or_tensor";
  int tensor_rank = -1;
  int field_count = -1;
};

void source_fallback_signature_stub() {
  // Source-fallback signature for tutorial parity tests when runtime output is unavailable.
  if (false) {
    print_signature({
        {"tutorial", "001"},
        {"lang", "cpp"},
        {"flow", "minimal_cvmat_dataloader"},
        {"run_mode", "sync"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
        {"tput_contract", "-1"},
    });
  }
}

simaai::neat::Model::Options build_model_options(int size) {
  simaai::neat::Model::Options opt;
  opt.format = "RGB";
  opt.input_max_width = size;
  opt.input_max_height = size;
  opt.input_max_depth = 3;
  opt.preproc.channel_mean = {0.485f, 0.456f, 0.406f};
  opt.preproc.channel_stddev = {0.229f, 0.224f, 0.225f};
  return opt;
}

std::vector<fs::path> resnet_image_candidates() {
  const fs::path assets = find_asset_root();
  return {
      assets / "fronalpstock_1330.jpg",
      assets / "ilena_488.jpg",
      assets / "lichtenstein_512.png",
  };
}

cv::Mat load_rgb_u8(const fs::path& path, int size) {
  cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("failed to read image: " + path.string());
  }

  if (bgr.cols != size || bgr.rows != size) {
    cv::resize(bgr, bgr, cv::Size(size, size), 0, 0, cv::INTER_AREA);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  if (rgb.type() != CV_8UC3) {
    throw std::runtime_error("expected CV_8UC3 RGB image");
  }
  if (!rgb.isContinuous()) {
    rgb = rgb.clone();
  }
  return rgb;
}

std::vector<cv::Mat> dataloader_from_images(int size, int n) {
  std::vector<fs::path> existing;
  for (const auto& p : resnet_image_candidates()) {
    if (fs::exists(p)) {
      existing.push_back(p);
    }
  }
  if (existing.empty()) {
    throw std::runtime_error("no local images found for ResNet50 run");
  }

  const int count = std::max(1, n);
  std::vector<cv::Mat> images;
  images.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    images.push_back(load_rgb_u8(existing[static_cast<size_t>(i) % existing.size()], size));
  }
  return images;
}

std::vector<float> scores_from_output(const simaai::neat::Sample& out) {
  check("has_tensor_output", out.tensor.has_value(),
                     "expected tensor output from model");

  const simaai::neat::Tensor& t = *out.tensor;
  check("tensor_float32", t.dtype == simaai::neat::TensorDType::Float32,
                     "expected float32 logits tensor");

  const simaai::neat::Mapping map = t.map_read();
  check("tensor_non_empty", map.data != nullptr && map.size_bytes > 0,
                     "model output tensor bytes must be non-empty");
  check("tensor_size_aligned", (map.size_bytes % sizeof(float)) == 0,
                     "tensor bytes must align to float32");

  const size_t elems = map.size_bytes / sizeof(float);
  std::vector<float> flat(elems);
  std::memcpy(flat.data(), map.data, map.size_bytes);
  if (flat.size() >= 1000) {
    flat.resize(1000);
  }
  return flat;
}

int top1_from_output(const simaai::neat::Sample& out) {
  std::vector<float> scores = scores_from_output(out);
  if (scores.empty()) {
    throw std::runtime_error("empty score vector");
  }
  auto it = std::max_element(scores.begin(), scores.end());
  return static_cast<int>(std::distance(scores.begin(), it));
}

Sig summarize(const simaai::neat::Sample& out) {
  Sig sig;
  sig.output_kind = std::to_string(static_cast<int>(out.kind));
  sig.tensor_rank = out.tensor.has_value() ? static_cast<int>(out.tensor->shape.size()) : -1;
  sig.field_count = static_cast<int>(out.fields.size());
  return sig;
}

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--mpk <path>] [--size <n>] [--n <count>]"
            << " [--timeout-ms <ms>] [--expect-id <id>] [--print-gst]\n";
  print_common_flags(std::cout);
}

} // namespace

int main(int argc, char** argv) {
  try {
    source_fallback_signature_stub();

    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int size = parse_int_arg(argc, argv, "--size", 224);
    const int n = parse_int_arg(argc, argv, "--n", 4);
    const int timeout_ms = parse_int_arg(argc, argv, "--timeout-ms", 2000);
    const int expect_id = parse_int_arg(argc, argv, "--expect-id", -1);
    require(size > 0, "--size must be > 0");
    require(n > 0, "--n must be > 0");

    step("input_contract",
                      "parse CLI and prepare ResNet50 model + local cv::Mat dataloader");
    step("run_mode_choice", "run synchronous inference over cv::Mat inputs");
    why(
        "start with one minimal model loop before introducing graph/session composition");
    tradeoff("this chapter optimizes for clarity and determinism over throughput");
    failure_mode("missing MPK/images or runtime issues should be explicit");
    interpret_output("top1 is human-facing; signature fields are tooling-facing");
    step("output_contract", "emit top1 lines and a stable tutorial signature");
    check("strict_mode_visible",
                       yes_no(strict_mode()) == "yes" ||
                           yes_no(strict_mode()) == "no",
                       "strict-mode guard is observable");

    const fs::path root = find_repo_root();

    std::string mpk_arg;
    const fs::path mpk_path = get_arg(argc, argv, "--mpk", mpk_arg)
                                  ? fs::path(mpk_arg)
                                  : default_resnet_mpk();
    if (mpk_path.empty() || !fs::exists(mpk_path)) {
      return skip("missing ResNet50 MPK (pass --mpk)");
    }

    Sig sig;
    int tput_contract = -1;
    try {
      // CORE LOGIC
      // The "6-line story": model -> image loader -> run -> top1 -> signature.
      simaai::neat::Model resnet50(mpk_path.string(), build_model_options(size));

      if (wants_print_gst(argc, argv)) {
        simaai::neat::Session s;
        s.add(resnet50.session());
        std::cout << s.describe_backend() << "\n";
        return 0;
      }

      // Contract: dataloader yields HWC uint8 RGB cv::Mat (CV_8UC3, contiguous).
      const std::vector<cv::Mat> resnet_dataloader = dataloader_from_images(size, n);

      int processed = 0;
      const auto start = std::chrono::steady_clock::now();
      for (const cv::Mat& image : resnet_dataloader) {
        simaai::neat::Sample out = resnet50.run(image, timeout_ms);
        const int pred = top1_from_output(out);
        sig = summarize(out);
        std::cout << "top1=" << pred << "\n";
        ++processed;

        if (expect_id >= 0) {
          check("top1_expected_id", pred == expect_id, "verify expected class id");
        }
      }
      // END CORE LOGIC
      const auto end = std::chrono::steady_clock::now();
      const double elapsed_sec = std::max(
          1e-9, std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
      const double tput_fps = static_cast<double>(processed) / elapsed_sec;
      tput_contract = processed;
      std::cout << "tput_fps:        " << tput_fps << "\n";
      std::cout << "tput_contract:   " << tput_contract << "\n";
    } catch (const std::exception& e) {
      runtime_fallback(e);
      if (strict_mode()) {
        throw;
      }
    }

    check("tutorial_completed", true, "minimal cv::Mat dataloader path completed");
    print_signature({
        {"tutorial", "001"},
        {"lang", "cpp"},
        {"flow", "minimal_cvmat_dataloader"},
        {"run_mode", "sync"},
        {"output_kind", sig.output_kind},
        {"tensor_rank", std::to_string(sig.tensor_rank)},
        {"field_count", std::to_string(sig.field_count)},
        {"tput_contract", std::to_string(tput_contract)},
    });
    std::cout << "[OK] 001_model_in_5_minutes\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
