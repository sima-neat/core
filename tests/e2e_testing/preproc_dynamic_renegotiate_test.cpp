#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"

#include "e2e_pipelines/e2e_utils.h"
#include "gst/GstHelpers.h"
#include "test_utils.h"
#include "pipeline/runtime/RunInternal.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
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

int json_input_width(const json& j, int def = -1) {
  const int legacy = json_int(j, "input_width", def);
  if (legacy != def)
    return legacy;
  auto it = j.find("input_shape");
  if (it != j.end() && it->is_array() && it->size() >= 2 && (*it)[1].is_number_integer())
    return (*it)[1].get<int>();
  return def;
}

int json_input_height(const json& j, int def = -1) {
  const int legacy = json_int(j, "input_height", def);
  if (legacy != def)
    return legacy;
  auto it = j.find("input_shape");
  if (it != j.end() && it->is_array() && !it->empty() && (*it)[0].is_number_integer())
    return (*it)[0].get<int>();
  return def;
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
  const simaai::neat::Tensor* tensor_ptr = nullptr;
  if (!out.tensors.empty()) {
    tensor_ptr = &out.tensors.front();
  } else if (out.tensor.has_value()) {
    tensor_ptr = &*out.tensor;
  }
  require(tensor_ptr != nullptr, "preproc: missing output tensor");
  const simaai::neat::Tensor& tensor = *tensor_ptr;
  require(tensor.shape.size() >= 2, "preproc: missing output shape");
  require(tensor.shape[0] == expected_h, "preproc: height mismatch");
  require(tensor.shape[1] == expected_w, "preproc: width mismatch");
  if (tensor.semantic.image.has_value()) {
    require(tensor.semantic.image->format == simaai::neat::ImageSpec::PixelFormat::RGB,
            "preproc: format mismatch");
  }
  require(tensor.dtype == simaai::neat::TensorDType::UInt8 ||
              tensor.dtype == simaai::neat::TensorDType::Int8,
          "preproc: dtype mismatch");

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

simaai::neat::Tensor make_rgb_tensor(const cv::Mat& rgb) {
  return simaai::neat::Tensor::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB,
                                           simaai::neat::TensorMemory::EV74);
}

} // namespace

int main() {
  try {
    using namespace simaai::neat;

    require(simaai::neat::element_exists("neatprocesscvu"),
            "Missing SIMA preproc plugin (neatprocesscvu).");

    fs::path root = find_repo_root();
    fs::path assets_dir = root / "tests" / "assets" / "preproc_dynamic";
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

    Graph p;

    InputOptions src_opt;
    src_opt.payload_type = PayloadType::Image;
    src_opt.format = simaai::neat::FormatTag::RGB;
    src_opt.use_simaai_pool = true;
    p.add(nodes::Input(src_opt));

    PreprocOptions pre_opt;
    pre_opt.set_input_shape({max_h, max_w, 3});
    pre_opt.set_output_shape({out_h, out_w, 3});
    pre_opt.scaled_width = out_w;
    pre_opt.scaled_height = out_h;
    pre_opt.input_img_type = "RGB";
    pre_opt.output_img_type = "RGB";
    pre_opt.normalize = false;
    pre_opt.aspect_ratio = false;
    pre_opt.tessellate = false;
    pre_opt.dynamic_input_dims = true;
    pre_opt.output_dtype = "EVXX_INT8";
    pre_opt.scaling_type = "BILINEAR";
    pre_opt.padding_type = "CENTER";
    pre_opt.next_cpu = "APU";
    pre_opt.upstream_name = "decoder";
    pre_opt.num_buffers = 4;
    pre_opt.q_scale = 0.25;
    pre_opt.q_zp = 0;

    auto preproc = std::make_shared<Preproc>(pre_opt);
    p.add(preproc);

    OutputOptions sink_opt;
    sink_opt.sync = false;
    sink_opt.drop = true;
    sink_opt.max_buffers = 1;
    p.add(nodes::Output(sink_opt));

    RunOptions run_opt;
    run_opt.advanced.max_input_bytes = max_bytes;
    run_opt.output_memory =
        simaai::neat::OutputMemory::Owned; // output is copied to CPU, doesn’t pin pool buffers

    Run run = p.build(TensorList{make_rgb_tensor(images[0].rgb)}, run_opt);

    auto output_matches_image = [&](const ImageCase& img, const Sample& out,
                                    const std::string& label, bool require_match) {
      cv::Mat out_rgb = require_preproc_rgb(out, out_w, out_h);
      cv::Mat expected;
      cv::resize(img.rgb, expected, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);

      Metrics m = compare_rgb(out_rgb, expected);
      const double mae_thr = img.mae_base + img.mae_delta;
      const double max_thr = img.max_base + img.max_delta;
      const bool matches = (m.mae <= mae_thr) && (m.max_abs <= max_thr);

      std::cerr << "[METRICS] " << img.name;
      if (!label.empty())
        std::cerr << " " << label;
      std::cerr << " MAE=" << m.mae << " MaxAbs=" << m.max_abs << " (thr: " << mae_thr << ", "
                << max_thr << ")\n";

      if (require_match) {
        require(m.mae <= mae_thr, "preproc: MAE too high for " + img.name);
        require(m.max_abs <= max_thr, "preproc: MaxAbs too high for " + img.name);
      }
      return matches;
    };

    auto check_image = [&](const ImageCase& img, const Sample& out) {
      (void)output_matches_image(img, out, "", true);
    };

    auto pull_matching_image = [&](const ImageCase& img, int timeout_ms) -> Sample {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
      size_t inspected = 0;
      while (std::chrono::steady_clock::now() < deadline) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   deadline - std::chrono::steady_clock::now())
                                   .count();
        Sample outs = run.pull_samples(static_cast<int>(std::max<int64_t>(1, remaining)));
        for (auto& out : outs) {
          const std::string label = "candidate#" + std::to_string(inspected);
          inspected++;
          if (output_matches_image(img, out, label, false)) {
            return std::move(out);
          }
          std::cerr << "[DRAIN] preproc: ignoring non-matching queued output while waiting for "
                    << img.name << "\n";
        }
      }
      throw std::runtime_error("preproc: no matching output after renegotiation for " + img.name +
                               " after inspecting " + std::to_string(inspected) + " samples");
    };

    require(run.push(TensorList{make_rgb_tensor(images[0].rgb)}), "preproc: first push failed");
    Sample outs1 = run.pull_samples(5000);
    require(outs1.size() == 1, "preproc: expected one output sample");
    Sample out1 = std::move(outs1.front());
    check_image(images[0], out1);
    require(run_internal::input_stats(run).renegotiations == 0,
            "preproc: unexpected renegotiation on first frame");

    require(preproc->config_path().empty(),
            "preproc: standalone config_path should be empty with typed processcvu payloads");
    const json* cfg0_ptr = preproc->config_json();
    require(cfg0_ptr != nullptr, "preproc: config json missing");
    const json cfg0 = *cfg0_ptr;
    const int input_w0 = json_input_width(cfg0);
    const int input_h0 = json_input_height(cfg0);
    const std::string input_fmt0 = json_string(cfg0, "input_img_type");
    require(input_w0 > 0 && input_h0 > 0, "preproc: invalid input dims in json");
    require(!input_fmt0.empty(), "preproc: missing input_img_type in json");

    std::uint64_t expected_reneg = 0;
    for (size_t i = 1; i < images.size(); ++i) {
      require(run.push(TensorList{make_rgb_tensor(images[i].rgb)}),
              "preproc: push new size #1 failed");
      require(run.push(TensorList{make_rgb_tensor(images[i].rgb)}),
              "preproc: push new size #2 failed");

      (void)pull_matching_image(images[i], 5000);

      expected_reneg++;
      require(wait_for_reneg(run, expected_reneg, 1000),
              "preproc: renegotiation not observed after stability");

      const json* cfg_ptr = preproc->config_json();
      require(cfg_ptr != nullptr, "preproc: config json missing after renegotiation");
      const json& cfg = *cfg_ptr;
      require(json_input_width(cfg) == images[i].rgb.cols,
              "preproc: input_width not updated from renegotiated input contract");
      require(json_input_height(cfg) == images[i].rgb.rows,
              "preproc: input_height not updated from renegotiated input contract");
      require(json_string(cfg, "input_img_type") == input_fmt0,
              "preproc: input_img_type changed on dims-only renegotiation");
    }

    require(run_internal::input_stats(run).renegotiations == expected_reneg,
            "preproc: unexpected renegotiation count");

    {
      PreprocOptions fmt_opt = pre_opt;

      Graph p2;
      InputOptions fmt_src;
      fmt_src.payload_type = PayloadType::Image;
      fmt_src.format = "";
      fmt_src.use_simaai_pool = true;
      p2.add(nodes::Input(fmt_src));
      auto fmt_preproc = std::make_shared<Preproc>(fmt_opt);
      p2.add(fmt_preproc);
      p2.add(nodes::Output(sink_opt));

      RunOptions fmt_run_opt = run_opt;
      simaai::neat::Tensor tensor_rgb = simaai::neat::Tensor::from_cv_mat(
          images[0].rgb, simaai::neat::ImageSpec::PixelFormat::RGB,
          simaai::neat::TensorMemory::EV74);
      simaai::neat::Tensor tensor_bgr = simaai::neat::Tensor::from_cv_mat(
          images[0].rgb, simaai::neat::ImageSpec::PixelFormat::BGR,
          simaai::neat::TensorMemory::EV74);

      Run run2 = p2.build(TensorList{tensor_rgb}, fmt_run_opt);
      (void)run2.run(TensorList{tensor_rgb}, 5000);
      require(fmt_preproc->config_path().empty(),
              "preproc: standalone format config_path should stay empty");
      const json* fmt_cfg0_ptr = fmt_preproc->config_json();
      require(fmt_cfg0_ptr != nullptr, "preproc: format config json missing");
      const json fmt_cfg0 = *fmt_cfg0_ptr;
      const std::string fmt0 = json_string(fmt_cfg0, "input_img_type");

      require(run2.push(TensorList{tensor_bgr}), "preproc: push BGR input failed");
      require(wait_for_reneg(run2, 1, 1000), "preproc: format change renegotiation not observed");

      const json* fmt_cfg1_ptr = fmt_preproc->config_json();
      require(fmt_cfg1_ptr != nullptr, "preproc: format config json missing after format change");
      const json fmt_cfg1 = *fmt_cfg1_ptr;
      const std::string fmt1 = json_string(fmt_cfg1, "input_img_type");

      require(!fmt0.empty() && !fmt1.empty(),
              "preproc: input_img_type missing after format change");
      require(fmt0 != fmt1, "preproc: input_img_type did not change on format change");
      require(fmt1 == "BGR", "preproc: input_img_type not updated to BGR");
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
