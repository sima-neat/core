#include "pipeline/Session.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"

#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstring>
#include <iostream>
#include <optional>
#include <string>

namespace {

bool compare_rgb_tensor(const cv::Mat& img, const simaai::neat::Tensor& t, std::string& err) {
  if (t.dtype != simaai::neat::TensorDType::UInt8) {
    err = "dtype mismatch";
    return false;
  }
  if (!t.semantic.image.has_value() ||
      t.semantic.image->format != simaai::neat::ImageSpec::PixelFormat::RGB) {
    err = "format mismatch";
    return false;
  }
  if (t.shape.size() < 2) {
    err = "shape missing";
    return false;
  }
  const int h = static_cast<int>(t.shape[0]);
  const int w = static_cast<int>(t.shape[1]);
  if (w != img.cols || h != img.rows) {
    err = "size mismatch";
    return false;
  }

  simaai::neat::Mapping map = t.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes == 0) {
    err = "missing mapping";
    return false;
  }
  const int row_bytes = img.cols * img.channels();
  int64_t stride = row_bytes;
  if (!t.strides_bytes.empty()) {
    stride = t.strides_bytes[0];
  }
  if (stride < row_bytes) {
    err = "stride too small";
    return false;
  }
  for (int r = 0; r < img.rows; ++r) {
    const uint8_t* lhs = static_cast<const uint8_t*>(map.data) + r * stride;
    const uint8_t* rhs = img.ptr<uint8_t>(r);
    if (std::memcmp(lhs, rhs, row_bytes) != 0) {
      err = "byte mismatch";
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  try {
    constexpr int kW = 640;
    constexpr int kH = 640;
    const int iters = 50;

    cv::Mat img(kH, kW, CV_8UC3, cv::Scalar(64, 128, 192));
    if (!img.isContinuous())
      img = img.clone();

    simaai::neat::Session p;
    simaai::neat::InputOptions src_opt;
    src_opt.format = "RGB";
    src_opt.width = kW;
    src_opt.height = kH;
    src_opt.depth = 3;
    src_opt.is_live = true;
    src_opt.do_timestamp = true;
    src_opt.block = true;
    src_opt.use_simaai_pool = false;
    src_opt.max_bytes = static_cast<std::uint64_t>(img.total() * img.elemSize());
    p.add(simaai::neat::nodes::Input(src_opt));

    simaai::neat::OutputOptions sink_opt;
    sink_opt.sync = false;
    sink_opt.drop = false;
    sink_opt.max_buffers = 8;
    p.add(simaai::neat::nodes::Output(sink_opt));

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = iters;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::Block;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    run_opt.advanced.copy_input = false;

    auto run = p.build(img, simaai::neat::RunMode::Async, run_opt);

    for (int i = 0; i < iters; ++i) {
      require(run.push(img), "async push failed");
    }
    run.close_input();

    int outputs = 0;
    bool checked = false;
    std::string err;
    while (true) {
      auto out = run.pull(20000);
      if (!out.has_value())
        break;
      outputs += 1;
      if (!checked) {
        if (out->tensor.has_value()) {
          require(compare_rgb_tensor(img, out->tensor.value(), err), err);
        } else {
          require(false, "unexpected output kind");
        }
        checked = true;
      }
    }

    require(outputs == iters, "async output count mismatch");

    std::cout << "[OK] async_stream_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
