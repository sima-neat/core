#pragma once

#include "asset_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace sima_example_test {

namespace fs = std::filesystem;

struct ExampleIoPaths {
  fs::path input_dir;
  fs::path output_dir;
  fs::path input_image;
  std::string stem;
};

inline fs::path resolve_root(int argc, char** argv) {
  const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
  std::error_code ec;
  fs::create_directories(root / "tmp", ec);
  fs::current_path(root, ec);
  return root;
}

inline fs::path resolve_example_binary(const fs::path& root, const char* argv0,
                                       const std::string& example_name) {
  std::error_code ec;
  const fs::path arg0 = argv0 ? fs::path(argv0) : fs::path();
  const fs::path abs_arg0 =
      arg0.empty() ? fs::path() : fs::weakly_canonical(fs::absolute(arg0), ec);
  const fs::path test_dir = abs_arg0.empty() ? fs::current_path() : abs_arg0.parent_path();
  const fs::path build_dir = test_dir.parent_path();

  const std::vector<fs::path> candidates = {
      build_dir / "examples" / example_name,
      build_dir / example_name,
      root / "build" / "examples" / example_name,
      root / "build" / example_name,
  };

  for (const auto& candidate : candidates) {
    if (fs::exists(candidate, ec)) {
      return candidate;
    }
  }

  throw std::runtime_error("Unable to locate example binary: " + example_name);
}

inline ExampleIoPaths prepare_single_input(const fs::path& root, const std::string& test_name) {
  const fs::path sample = sima_test::ensure_coco_sample(root);
  require(!sample.empty() && fs::exists(sample), "Failed to prepare sample image");

  cv::Mat img = cv::imread(sample.string(), cv::IMREAD_COLOR);
  require(!img.empty(), "Failed to read sample image: " + sample.string());

  ExampleIoPaths io;
  io.stem = "sample";
  const fs::path base = root / "tmp" / test_name;
  io.input_dir = base / "input";
  io.output_dir = base / "output";
  io.input_image = io.input_dir / (io.stem + ".jpg");

  std::error_code ec;
  fs::remove_all(base, ec);
  fs::create_directories(io.input_dir, ec);
  fs::create_directories(io.output_dir, ec);

  require(cv::imwrite(io.input_image.string(), img), "Failed to write input image for test");
  return io;
}

inline void run_example_or_throw(const fs::path& example_bin, const std::string& model_tar,
                                 const fs::path& input_dir, const fs::path& output_dir) {
  require(fs::exists(example_bin), "Example binary not found: " + example_bin.string());
  require(!model_tar.empty(), "Model tar path is empty");
  require(fs::exists(model_tar), "Model tar not found: " + model_tar);

  const std::string cmd = sima_test::shell_quote(example_bin.string()) + " " +
                          sima_test::shell_quote(model_tar) + " " +
                          sima_test::shell_quote(input_dir.string()) + " " +
                          sima_test::shell_quote(output_dir.string());
  const int rc = std::system(cmd.c_str());
  require(rc == 0, "Example command failed with code " + std::to_string(rc) + ": " + cmd);
}

inline cv::Mat require_image(const fs::path& image_path, const std::string& label) {
  std::error_code ec;
  require(fs::exists(image_path, ec), label + " missing: " + image_path.string());
  require(fs::is_regular_file(image_path, ec),
          label + " is not a regular file: " + image_path.string());
  require(fs::file_size(image_path, ec) > 0 && !ec, label + " is empty: " + image_path.string());

  cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  require(!img.empty(), label + " is unreadable: " + image_path.string());
  return img;
}

inline std::string resolve_model_tar_or_throw(const std::string& model_name, const fs::path& root) {
  const std::string tar = sima_test::resolve_modelzoo_tar(model_name, root);
  require(!tar.empty(), "Failed to resolve model tar for '" + model_name +
                            "'. Run: sima-cli modelzoo get " + model_name);
  return tar;
}

} // namespace sima_example_test
