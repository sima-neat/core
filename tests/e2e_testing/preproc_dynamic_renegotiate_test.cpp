#include "pipeline/Session.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"

#include "e2e_pipelines/e2e_utils.h"
#include "gst/GstHelpers.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

using json = nlohmann::json;

std::string filename_from_url(const std::string& url) {
  const std::size_t slash = url.find_last_of('/');
  std::string name = (slash == std::string::npos) ? url : url.substr(slash + 1);
  const std::size_t qmark = name.find('?');
  if (qmark != std::string::npos)
    name = name.substr(0, qmark);
  return name;
}

fs::path find_repo_root() {
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path())
      break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

json read_json_file(const fs::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("preproc: failed to open json: " + path.string());
  }
  json j;
  in >> j;
  return j;
}

std::string json_string(const json& j, const char* key) {
  auto it = j.find(key);
  if (it == j.end() || !it->is_string())
    return "";
  return it->get<std::string>();
}

int json_int(const json& j, const char* key, int def) {
  auto it = j.find(key);
  if (it == j.end() || !it->is_number_integer())
    return def;
  return it->get<int>();
}

bool is_preproc_config(const fs::path& path) {
  const std::string name = path.filename().string();
  return name.rfind("sima_preproc_", 0) == 0 && path.extension() == ".json";
}

void cleanup_preproc_configs(const fs::path& dir) {
  if (!fs::exists(dir))
    return;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file())
      continue;
    if (is_preproc_config(entry.path())) {
      std::error_code ec;
      fs::remove(entry.path(), ec);
    }
  }
}

fs::path find_latest_preproc_config(const fs::path& dir) {
  fs::path latest;
  fs::file_time_type latest_time{};
  if (!fs::exists(dir))
    return latest;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file())
      continue;
    const fs::path p = entry.path();
    if (!is_preproc_config(p))
      continue;
    const fs::file_time_type t = fs::last_write_time(p);
    if (latest.empty() || t > latest_time) {
      latest = p;
      latest_time = t;
    }
  }
  return latest;
}

struct Metrics {
  double mae = 0.0;
  double max_abs = 0.0;
};

Metrics compare_rgb(const cv::Mat& a, const cv::Mat& b) {
  require(!a.empty() && !b.empty(), "compare_rgb: empty image");
  require(a.size() == b.size() && a.type() == b.type(), "compare_rgb: size/type mismatch");

  cv::Mat diff;
  cv::absdiff(a, b, diff);

  cv::Scalar mean_abs = cv::mean(diff);
  const double mae = (mean_abs[0] + mean_abs[1] + mean_abs[2]) / 3.0;

  cv::Mat diff_gray;
  cv::cvtColor(diff, diff_gray, cv::COLOR_RGB2GRAY);
  double minv = 0.0, maxv = 0.0;
  cv::minMaxLoc(diff_gray, &minv, &maxv);

  return {mae, maxv};
}

cv::Mat require_preproc_rgb(const simaai::neat::Sample& out, int expected_w, int expected_h) {
  require(out.tensor.has_value(), "preproc: missing output tensor");
  const simaai::neat::Tensor& tensor = *out.tensor;
  require(tensor.shape.size() >= 2, "preproc: missing output shape");
  require(tensor.shape[0] == expected_h, "preproc: height mismatch");
  require(tensor.shape[1] == expected_w, "preproc: width mismatch");
  require(tensor.semantic.image.has_value(), "preproc: missing image semantic");
  require(tensor.semantic.image->format == simaai::neat::ImageSpec::PixelFormat::RGB,
          "preproc: format mismatch");
  require(tensor.dtype == simaai::neat::TensorDType::UInt8, "preproc: dtype mismatch");

  simaai::neat::Tensor cpu = tensor.cpu().contiguous();
  simaai::neat::Mapping map = cpu.map(simaai::neat::MapMode::Read);
  const size_t expected_bytes =
      static_cast<size_t>(expected_w) * static_cast<size_t>(expected_h) * 3;
  require(map.data != nullptr && map.size_bytes >= expected_bytes, "preproc: output bytes missing");

  size_t step = expected_w * 3;
  if (!cpu.strides_bytes.empty() && cpu.strides_bytes[0] > 0) {
    step = static_cast<size_t>(cpu.strides_bytes[0]);
  }
  cv::Mat rgb(expected_h, expected_w, CV_8UC3, map.data, step);
  return rgb.clone();
}

// bool wait_for_reneg(simaai::neat::Run& run,
//                     std::uint64_t target,
//                     int timeout_ms) {
//   const auto deadline =
//       std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
//   while (std::chrono::steady_clock::now() < deadline) {
//     if (run.input_stats().renegotiations >= target) return true;
//     std::this_thread::sleep_for(std::chrono::milliseconds(10));
//   }
//   return run.input_stats().renegotiations >= target;
// }

struct ImageCase {
  std::string name;
  std::string url;
  std::string filename;
  double mae_base = 0.0;
  double mae_delta = 3.0;
  double max_base = 0.0;
  double max_delta = 12.0;
  cv::Mat rgb;
};

cv::Mat load_rgb_image(const fs::path& path, const std::string& name) {
  cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
  require(!bgr.empty(), "preproc: failed to load image: " + name);
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  if (!rgb.isContinuous())
    rgb = rgb.clone();
  require(rgb.type() == CV_8UC3, "preproc: expected CV_8UC3 image: " + name);
  return rgb;
}

} // namespace

int main() {
  try {
    using namespace simaai::neat;

    require(simaai::neat::element_exists("neatprocesscvu"),
            "Missing SIMA preproc plugin (neatprocesscvu).");

    fs::path root = find_repo_root();
    fs::path assets_dir = root / "tests" / "assets" / "preproc_dynamic";
    fs::path tmp_dir = root / "tmp" / "preproc_dynamic_test";
    fs::path fmt_tmp_dir = tmp_dir / "format";
    fs::create_directories(tmp_dir);
    fs::create_directories(fmt_tmp_dir);
    cleanup_preproc_configs(tmp_dir);
    cleanup_preproc_configs(fmt_tmp_dir);

    std::vector<ImageCase> images = {
        {
            "ilena_488",
            "https://commons.wikimedia.org/wiki/Special:FilePath/ILena.jpg",
            "ilena_488.jpg",
            0.49,
            2.0,
            54.0,
            10.0,
            {},
        },
        {
            "lichtenstein_512",
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            "Lichtenstein_img_processing_test.png",
            "lichtenstein_512.png",
            0.61,
            2.0,
            18.0,
            10.0,
            {},
        },
        {
            "fronalpstock_1330",
            "https://commons.wikimedia.org/wiki/Special:FilePath/Fronalpstock.jpg",
            "fronalpstock_1330.jpg",
            0.71,
            2.0,
            70.0,
            10.0,
            {},
        },
    };

    for (auto& img : images) {
      if (img.filename.empty())
        img.filename = filename_from_url(img.url);
      fs::path local = assets_dir / img.filename;
      if (!fs::exists(local)) {
        if (!sima_e2e::download_file(img.url, local)) {
          throw std::runtime_error("preproc: failed to download " + img.url);
        }
      }
      img.rgb = load_rgb_image(local, img.name);
    }

    std::set<std::pair<int, int>> sizes;
    for (const auto& img : images) {
      sizes.insert({img.rgb.cols, img.rgb.rows});
    }
    require(sizes.size() == images.size(),
            "preproc: expected unique input sizes for renegotiation");

    std::sort(images.begin(), images.end(), [](const ImageCase& a, const ImageCase& b) {
      return static_cast<int64_t>(a.rgb.cols) * a.rgb.rows >
             static_cast<int64_t>(b.rgb.cols) * b.rgb.rows;
    });

    int max_w = 0;
    int max_h = 0;
    size_t max_bytes = 0;
    for (const auto& img : images) {
      max_w = std::max(max_w, img.rgb.cols);
      max_h = std::max(max_h, img.rgb.rows);
      const size_t bytes =
          static_cast<size_t>(img.rgb.cols) * static_cast<size_t>(img.rgb.rows) * 3;
      max_bytes = std::max(max_bytes, bytes);
    }

    const int out_w = 640;
    const int out_h = 640;

    Session p;

    InputOptions src_opt;
    src_opt.media_type = "video/x-raw";
    src_opt.format = "RGB";
    src_opt.use_simaai_pool = true;
    p.add(nodes::Input(src_opt));

    PreprocOptions pre_opt;
    pre_opt.input_width = max_w;
    pre_opt.input_height = max_h;
    pre_opt.output_width = out_w;
    pre_opt.output_height = out_h;
    pre_opt.scaled_width = out_w;
    pre_opt.scaled_height = out_h;
    pre_opt.input_img_type = "RGB";
    pre_opt.output_img_type = "RGB";
    pre_opt.normalize = false;
    pre_opt.aspect_ratio = false;
    pre_opt.dynamic_input_dims = true;
    pre_opt.output_dtype = "EVXX_INT8";
    pre_opt.scaling_type = "BILINEAR";
    pre_opt.padding_type = "CENTER";
    pre_opt.next_cpu = "APU";
    pre_opt.upstream_name = "decoder";
    pre_opt.num_buffers = 4;
    pre_opt.config_dir = tmp_dir.string();
    pre_opt.keep_config = true;
    pre_opt.output_memory_order = {"output_rgb_image", "output_tessellated_image"};

    p.add(nodes::Preproc(pre_opt));

    OutputOptions sink_opt;
    sink_opt.sync = false;
    sink_opt.drop = true;
    sink_opt.max_buffers = 1;
    p.add(nodes::Output(sink_opt));

    RunOptions run_opt;
    run_opt.advanced.max_input_bytes = max_bytes;
    run_opt.output_memory =
        simaai::neat::OutputMemory::Owned; // output is copied to CPU, doesn’t pin pool buffers

    Run run = p.build(images[0].rgb, RunMode::Async, run_opt);

    auto check_image = [&](const ImageCase& img, const Sample& out) {
      cv::Mat out_rgb = require_preproc_rgb(out, out_w, out_h);
      cv::Mat expected;
      cv::resize(img.rgb, expected, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);

      Metrics m = compare_rgb(out_rgb, expected);
      const double mae_thr = img.mae_base + img.mae_delta;
      const double max_thr = img.max_base + img.max_delta;

      std::cerr << "[METRICS] " << img.name << " MAE=" << m.mae << " MaxAbs=" << m.max_abs
                << " (thr: " << mae_thr << ", " << max_thr << ")\n";

      require(m.mae <= mae_thr, "preproc: MAE too high for " + img.name);
      require(m.max_abs <= max_thr, "preproc: MaxAbs too high for " + img.name);
    };

    Sample out1 = run.push_and_pull(images[0].rgb, 5000);
    check_image(images[0], out1);
    require(run.input_stats().renegotiations == 0,
            "preproc: unexpected renegotiation on first frame");

    const fs::path config_path = find_latest_preproc_config(tmp_dir);
    require(!config_path.empty(), "preproc: config json missing");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const fs::file_time_type mtime0 = fs::last_write_time(config_path);
    const json cfg0 = read_json_file(config_path);
    const int input_w0 = json_int(cfg0, "input_width", -1);
    const int input_h0 = json_int(cfg0, "input_height", -1);
    const std::string input_fmt0 = json_string(cfg0, "input_img_type");
    require(input_w0 > 0 && input_h0 > 0, "preproc: invalid input dims in json");
    require(!input_fmt0.empty(), "preproc: missing input_img_type in json");

    std::uint64_t expected_reneg = 0;
    for (size_t i = 1; i < images.size(); ++i) {
      require(run.push(images[i].rgb), "preproc: push new size #1 failed");
      require(run.push(images[i].rgb), "preproc: push new size #2 failed");

      auto out_opt = run.pull(5000);
      require(out_opt.has_value(), "appsink: missing output after renegotiation");
      check_image(images[i], *out_opt);

      expected_reneg++;
      require(wait_for_reneg(run, expected_reneg, 1000),
              "preproc: renegotiation not observed after stability");

      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      const fs::file_time_type mtime = fs::last_write_time(config_path);
      const json cfg = read_json_file(config_path);
      require(json_int(cfg, "input_width", -1) == input_w0,
              "preproc: input_width changed on dims-only renegotiation");
      require(json_int(cfg, "input_height", -1) == input_h0,
              "preproc: input_height changed on dims-only renegotiation");
      require(json_string(cfg, "input_img_type") == input_fmt0,
              "preproc: input_img_type changed on dims-only renegotiation");
      require(mtime == mtime0, "preproc: config json rewritten on dims-only renegotiation");
    }

    require(run.input_stats().renegotiations == expected_reneg,
            "preproc: unexpected renegotiation count");

    {
      PreprocOptions fmt_opt = pre_opt;
      fmt_opt.config_dir = fmt_tmp_dir.string();
      fmt_opt.keep_config = true;

      Session p2;
      InputOptions fmt_src;
      fmt_src.media_type = "video/x-raw";
      fmt_src.format = "";
      fmt_src.use_simaai_pool = true;
      p2.add(nodes::Input(fmt_src));
      p2.add(nodes::Preproc(fmt_opt));
      p2.add(nodes::Output(sink_opt));

      RunOptions fmt_run_opt = run_opt;
      simaai::neat::Tensor tensor_rgb = simaai::neat::Tensor::from_cv_mat(
          images[0].rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true);
      simaai::neat::Tensor tensor_bgr = simaai::neat::Tensor::from_cv_mat(
          images[0].rgb, simaai::neat::ImageSpec::PixelFormat::BGR, true);

      Run run2 = p2.build(tensor_rgb, RunMode::Async, fmt_run_opt);
      (void)run2.push_and_pull(tensor_rgb, 5000);
      const fs::path format_config_path = find_latest_preproc_config(fmt_tmp_dir);
      require(!format_config_path.empty(), "preproc: format config json missing");

      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      const fs::file_time_type fmt_mtime0 = fs::last_write_time(format_config_path);
      const json fmt_cfg0 = read_json_file(format_config_path);
      const std::string fmt0 = json_string(fmt_cfg0, "input_img_type");

      require(run2.push(tensor_bgr), "preproc: push BGR input failed");
      require(wait_for_reneg(run2, 1, 1000), "preproc: format change renegotiation not observed");

      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      const fs::file_time_type fmt_mtime1 = fs::last_write_time(format_config_path);
      const json fmt_cfg1 = read_json_file(format_config_path);
      const std::string fmt1 = json_string(fmt_cfg1, "input_img_type");

      require(!fmt0.empty() && !fmt1.empty(),
              "preproc: input_img_type missing after format change");
      require(fmt0 != fmt1, "preproc: input_img_type did not change on format change");
      require(fmt1 == "BGR", "preproc: input_img_type not updated to BGR");
      require(fmt_mtime1 != fmt_mtime0, "preproc: config json not rewritten on format change");
    }

    std::cout << "[OK] preproc_dynamic_renegotiate_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (is_dispatcher_unavailable(msg)) {
      return fail_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << msg << "\n";
    return 1;
  }
}
