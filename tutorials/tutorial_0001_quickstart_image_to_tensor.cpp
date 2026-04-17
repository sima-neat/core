// tutorial_0001_quickstart_image_to_tensor.cpp
// Story: the shortest path from an image file to a simaai::neat::Tensor.
// What you learn:
// - ImageInputGroup decodes an image and exposes raw video frames.
// - add_output_tensor() standardizes format/size for tensor outputs.
// - Run::pull_tensor_or_throw() yields a simaai::neat::Tensor directly.
//
// Prereqs:
// - An image file (default: test.jpg or tests/assets/preproc_dynamic/ilena_488.jpg).
//
// Try:
//   ./build/tutorial_0001_quickstart_image_to_tensor --print-gst
//   ./build/tutorial_0001_quickstart_image_to_tensor --image test.jpg

#include "neat.h"

#include "tutorial_common.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--image <path>] [--width <w>] [--height <h>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --image <path>       Image file to decode (jpg/png)\n";
  std::cout << "  --width <w>          Output width (default 256)\n";
  std::cout << "  --height <h>         Output height (default 256)\n";
}

fs::path default_image_path() {
  return sima_tutorial::find_asset_root() / "ilena_488.jpg";
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

void print_tensor_summary(const simaai::neat::Tensor& t) {
  std::cout << "Tensor summary:\n";
  std::cout << "  dtype: " << static_cast<int>(t.dtype) << "\n";
  std::cout << "  shape: [";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (i)
      std::cout << ", ";
    std::cout << t.shape[i];
  }
  std::cout << "]\n";
  if (t.semantic.image.has_value()) {
    std::cout << "  image format: " << static_cast<int>(t.semantic.image->format) << "\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    std::string image_arg;
    fs::path image_path;
    if (sima_tutorial::get_arg(argc, argv, "--image", image_arg)) {
      image_path = fs::path(image_arg);
    } else {
      image_path = default_image_path();
    }

    if (!sima_tutorial::exists_or_skip(image_path, "image")) {
      return 0;
    }

    const int out_w = parse_int_arg(argc, argv, "--width", 256);
    const int out_h = parse_int_arg(argc, argv, "--height", 256);

    // 1) Assemble the pipeline: image input + standardized tensor output.
    simaai::neat::Session p;

    simaai::neat::nodes::groups::ImageInputGroupOptions in;
    in.path = image_path.string();
    in.output_caps.width = out_w;
    in.output_caps.height = out_h;
    in.output_caps.format = "RGB";
    p.add(simaai::neat::nodes::groups::ImageInputGroup(in));

    simaai::neat::OutputTensorOptions out;
    out.format = "RGB";
    out.target_width = out_w;
    out.target_height = out_h;
    // NOTE: add_output_tensor currently supports UInt8 only, forces SystemMemory,
    // and does not transform layout (layout is metadata only).
    p.add_output_tensor(out);

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    // 2) Build and run, then pull a tensor result.
    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    auto run = p.build(run_opt);
    auto tensor = run.pull_tensor_or_throw(/*timeout_ms=*/2000);

    // 3) Print a tiny summary to prove we got data.
    print_tensor_summary(tensor);

    std::cout << "[OK] tutorial_0001 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
