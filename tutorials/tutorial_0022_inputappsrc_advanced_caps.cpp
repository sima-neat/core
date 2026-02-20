// tutorial_0022_inputappsrc_advanced_caps.cpp
// Story: advanced Input controls for caps and buffer pools.
// What you learn:
// - caps_override disables renegotiation.
// - format switches (RGB <-> BGR) are accepted by default.
// - buffer_name must match Sample.port_name when pushing Samples.
// - max_input_bytes bounds input buffer growth and reports clear errors.

#include "neat/session.h"
#include "neat/nodes.h"

#include "tutorial_common.h"

#include <opencv2/core.hpp>

#include <functional>
#include <iostream>
#include <string>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --width <w>          Width (default 64)\n";
  std::cout << "  --height <h>         Height (default 48)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!sima_tutorial::get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

cv::Mat make_rgb_mat(int w, int h, const cv::Scalar& color) {
  cv::Mat img(h, w, CV_8UC3, color);
  if (!img.isContinuous())
    img = img.clone();
  return img;
}

simaai::neat::Tensor make_rgb_tensor(const cv::Mat& img) {
  return simaai::neat::Tensor::from_cv_mat(img, simaai::neat::ImageSpec::PixelFormat::RGB, true);
}

simaai::neat::Tensor make_bgr_tensor(const cv::Mat& img) {
  return simaai::neat::Tensor::from_cv_mat(img, simaai::neat::ImageSpec::PixelFormat::BGR, true);
}

void run_section(const std::string& label, const std::function<void()>& fn) {
  std::cout << "\n[section] " << label << "\n";
  try {
    fn();
  } catch (const std::exception& e) {
    std::cout << "  caught error: " << e.what() << "\n";
  }
}

struct RunGuard {
  simaai::neat::Run* run = nullptr;
  explicit RunGuard(simaai::neat::Run& r) : run(&r) {}
  ~RunGuard() {
    if (!run)
      return;
    run->close_input();
    run->stop();
    run->close();
  }
};

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int w = parse_int_arg(argc, argv, "--width", 64);
    const int h = parse_int_arg(argc, argv, "--height", 48);

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      simaai::neat::Session p;
      p.add(simaai::neat::nodes::Input());
      p.add(simaai::neat::nodes::Output());
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    cv::Mat small = make_rgb_mat(w, h, cv::Scalar(10, 20, 30));
    cv::Mat large = make_rgb_mat(w * 2, h * 2, cv::Scalar(20, 30, 40));

    run_section("caps_override disables renegotiation", [&]() {
      simaai::neat::Session p;
      simaai::neat::InputOptions in;
      in.block = false;
      in.format = "RGB";
      in.width = w;
      in.height = h;
      in.depth = 3;
      in.caps_override = "video/x-raw,format=RGB,width=" + std::to_string(w) +
                         ",height=" + std::to_string(h) + ",framerate=30/1";
      p.add(simaai::neat::nodes::Input(in));
      p.add(simaai::neat::nodes::Output());

      simaai::neat::RunOptions run_opt;
      run_opt.output_memory = simaai::neat::OutputMemory::Owned;

      auto run = p.build(small, simaai::neat::RunMode::Sync, run_opt);
      RunGuard guard(run);
      run.push(small);
      run.pull(/*timeout_ms=*/1000);

      // This triggers a caps change; should error because caps_override is set.
      run.push(large);
    });

    run_section("format switch RGB <-> BGR is accepted by default", [&]() {
      simaai::neat::Session p;
      simaai::neat::InputOptions in;
      in.block = false;
      in.width = w;
      in.height = h;
      in.depth = 3;
      p.add(simaai::neat::nodes::Input(in));
      p.add(simaai::neat::nodes::Output());

      simaai::neat::RunOptions run_opt;
      run_opt.output_memory = simaai::neat::OutputMemory::Owned;

      auto run = p.build(small, simaai::neat::RunMode::Sync, run_opt);
      RunGuard guard(run);

      simaai::neat::Sample rgb_sample =
          simaai::neat::make_tensor_sample("decoder", make_rgb_tensor(small));
      rgb_sample.payload_tag = "RGB";
      simaai::neat::Sample bgr_sample =
          simaai::neat::make_tensor_sample("decoder", make_bgr_tensor(small));
      bgr_sample.payload_tag = "BGR";

      run.push(rgb_sample);
      run.pull(/*timeout_ms=*/1000);
      run.push(bgr_sample);
      run.pull(/*timeout_ms=*/1000);
    });

    run_section("buffer_name must match Sample.port_name", [&]() {
      simaai::neat::Session p;
      simaai::neat::InputOptions in;
      in.block = false;
      in.format = "RGB";
      in.width = w;
      in.height = h;
      in.depth = 3;
      in.buffer_name = "stream0";
      p.add(simaai::neat::nodes::Input(in));
      p.add(simaai::neat::nodes::Output());

      simaai::neat::RunOptions run_opt;
      run_opt.output_memory = simaai::neat::OutputMemory::Owned;

      auto run = p.build(small, simaai::neat::RunMode::Sync, run_opt);
      RunGuard guard(run);

      simaai::neat::Sample bad =
          simaai::neat::make_tensor_sample("stream1", make_rgb_tensor(small));
      bad.payload_tag = "RGB";
      run.push(bad);
    });

    run_section("max_input_bytes limits growth", [&]() {
      simaai::neat::Session p;
      simaai::neat::InputOptions in;
      in.block = false;
      in.format = "RGB";
      in.width = w;
      in.height = h;
      in.depth = 3;
      p.add(simaai::neat::nodes::Input(in));
      p.add(simaai::neat::nodes::Output());

      simaai::neat::RunOptions run_opt;
      run_opt.output_memory = simaai::neat::OutputMemory::Owned;
      run_opt.advanced.max_input_bytes = static_cast<size_t>(w * 2 * h * 2 * 3); // allow up to 2x

      auto run = p.build(small, simaai::neat::RunMode::Sync, run_opt);
      RunGuard guard(run);
      run.push(small);
      run.pull(/*timeout_ms=*/1000);

      // 3x size should exceed max_input_bytes.
      cv::Mat bigger = make_rgb_mat(w * 3, h * 3, cv::Scalar(5, 10, 15));
      run.push(bigger);
    });

    std::cout << "\n[OK] tutorial_0022 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
