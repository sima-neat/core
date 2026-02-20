// tutorial_0005_output_tensor_options.cpp
// Story: OutputTensorOptions lets you standardize output format and size.
// What you learn:
// - add_output_tensor() inserts convert/scale/caps + appsink automatically.
// - OutputTensorOptions controls format, dtype, and target dimensions.
// - RunOptions::output_memory controls output ownership/lifetime behavior.

#include "neat/session.h"
#include "neat/nodes.h"

#include "tutorial_common.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--in-width <w>] [--in-height <h>] [--out-width <w>] [--out-height <h>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --in-width <w>       Input width (default 320)\n";
  std::cout << "  --in-height <h>      Input height (default 240)\n";
  std::cout << "  --out-width <w>      Output width (default 128)\n";
  std::cout << "  --out-height <h>     Output height (default 128)\n";
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

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int in_w = parse_int_arg(argc, argv, "--in-width", 320);
    const int in_h = parse_int_arg(argc, argv, "--in-height", 240);
    const int out_w = parse_int_arg(argc, argv, "--out-width", 128);
    const int out_h = parse_int_arg(argc, argv, "--out-height", 128);

    cv::Mat img(in_h, in_w, CV_8UC3, cv::Scalar(80, 20, 180));
    if (!img.isContinuous())
      img = img.clone();

    simaai::neat::Session p;
    simaai::neat::InputOptions in;
    in.format = "RGB";
    in.width = in_w;
    in.height = in_h;
    in.depth = 3;
    p.add(simaai::neat::nodes::Input(in));

    simaai::neat::OutputTensorOptions out;
    out.format = "RGB";
    out.target_width = out_w;
    out.target_height = out_h;
    // NOTE: add_output_tensor currently supports UInt8 only and uses system memory.
    p.add_output_tensor(out);

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run = p.build(img, simaai::neat::RunMode::Sync, run_opt);
    auto out_tensor = run.push_and_pull(img, /*timeout_ms=*/1000).tensor;
    sima_tutorial::require(out_tensor.has_value(), "missing output tensor");

    const auto& t = *out_tensor;
    std::cout << "Input size:  " << in_w << "x" << in_h << "\n";
    std::cout << "Output size: " << t.shape[1] << "x" << t.shape[0] << "\n";
    std::cout << "Output dtype: " << static_cast<int>(t.dtype) << "\n";

    std::cout << "[OK] tutorial_0005 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
