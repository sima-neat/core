#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"

#include <opencv2/core.hpp>

#include <iostream>

namespace {
cv::Mat make_rgb_mat(int w, int h, const cv::Scalar& color) {
  cv::Mat img(h, w, CV_8UC3, color);
  if (!img.isContinuous())
    img = img.clone();
  return img;
}
} // namespace

int main() {
  const int w = 64;
  const int h = 48;
  const int iterations = 5;

  for (int i = 0; i < iterations; ++i) {
    simaai::neat::Graph p;
    simaai::neat::InputOptions in;
    in.format = simaai::neat::FormatTag::RGB;
    in.width = w;
    in.height = h;
    in.depth = 3;
    in.caps_override = "video/x-raw,format=RGB,width=" + std::to_string(w) +
                       ",height=" + std::to_string(h) + ",framerate=30/1";
    p.add(simaai::neat::nodes::Input(in));
    p.add(simaai::neat::nodes::Output());

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run = p.build(std::vector<cv::Mat>{make_rgb_mat(w, h, cv::Scalar(10, 20, 30))}, run_opt);

    run.push(std::vector<cv::Mat>{make_rgb_mat(w, h, cv::Scalar(10, 20, 30))});
    run.pull(/*timeout_ms=*/1000);

    cv::Mat larger = make_rgb_mat(w * 2, h * 2, cv::Scalar(20, 30, 40));
    run.push(std::vector<cv::Mat>{larger});

    run.close_input();
    run.stop();
    run.close();

    if ((i + 1) % 50 == 0) {
      std::cout << "[dbg] iteration " << (i + 1) << "/" << iterations << "\n";
    }
  }

  std::cout << "[OK] unit_inputappsrc_stop_race_test complete\n";
  return 0;
}
