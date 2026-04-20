// Convert a cv::Mat into a NEAT Tensor, map it read-only, and clone it.
//
// Usage:
//   tutorial_v2_008_numpy_torch_tensor_io [--width 128] [--height 96]

#include "neat.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string value;
  if (!get_arg(argc, argv, key, value))
    return def;
  return std::stoi(value);
}

} // namespace

int main(int argc, char** argv) {
  try {
    const int width = parse_int_arg(argc, argv, "--width", 128);
    const int height = parse_int_arg(argc, argv, "--height", 96);

    cv::Mat rgb(height, width, CV_8UC3, cv::Scalar(7, 17, 27));
    if (!rgb.isContinuous())
      rgb = rgb.clone();

    // CORE LOGIC
    // from_cv_mat wraps a cv::Mat as a NEAT Tensor (read-only here).
    simaai::neat::Tensor tensor = simaai::neat::from_cv_mat(
        rgb, simaai::neat::ImageSpec::PixelFormat::RGB, /*read_only=*/true);

    // map_read yields a Mapping with a raw pointer and size in bytes.
    simaai::neat::Mapping mapped = tensor.map_read();
    std::uint64_t checksum = 0;
    const auto* bytes = static_cast<const std::uint8_t*>(mapped.data);
    const std::size_t n = std::min<std::size_t>(mapped.size_bytes, 256);
    for (std::size_t i = 0; i < n; ++i)
      checksum += bytes[i];

    // clone() copies into CPU-owned storage, detached from the cv::Mat buffer.
    simaai::neat::Tensor owned = tensor.clone();
    // END CORE LOGIC

    std::cout << "tensor_rank=" << tensor.shape.size() << "\n";
    std::cout << "tensor_bytes=" << mapped.size_bytes << "\n";
    std::cout << "head_checksum=" << checksum << "\n";
    std::cout << "clone_bytes=" << owned.dense_bytes_tight() << "\n";
    std::cout << "[OK] 008_numpy_torch_tensor_io\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
