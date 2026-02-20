// tutorial_0004_samples_and_tensortensor.cpp
// Story: when to use Sample vs simaai::neat::Tensor, and how to read metadata.
// What you learn:
// - pull() returns a Sample with metadata (caps, timestamps, stream id).
// - pull_tensor() returns only the tensor payload for simpler cases.
// - Bundle samples contain multiple outputs (shown here with a synthetic example).

#include "neat.h"

#include "tutorial_common.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --width <w>          Input width (default 160)\n";
  std::cout << "  --height <h>         Input height (default 120)\n";
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

void print_sample(const simaai::neat::Sample& s, const std::string& label) {
  std::cout << label << "\n";
  std::cout << "  kind: " << static_cast<int>(s.kind) << "\n";
  if (!s.caps_string.empty()) {
    std::cout << "  caps: " << s.caps_string << "\n";
  }
  if (!s.media_type.empty()) {
    std::cout << "  media_type: " << s.media_type << "\n";
  }
  if (!s.payload_tag.empty()) {
    std::cout << "  payload_tag: " << s.payload_tag << "\n";
  }
  if (s.pts_ns >= 0) {
    std::cout << "  pts_ns: " << s.pts_ns << "\n";
  }
  if (s.kind == simaai::neat::SampleKind::Bundle) {
    std::cout << "  bundle fields: " << s.fields.size() << "\n";
    for (size_t i = 0; i < s.fields.size(); ++i) {
      std::cout << "    field[" << i << "] kind=" << static_cast<int>(s.fields[i].kind) << "\n";
    }
  }
  if (s.tensor.has_value()) {
    std::cout << "  tensor shape: [";
    for (size_t i = 0; i < s.tensor->shape.size(); ++i) {
      if (i)
        std::cout << ", ";
      std::cout << s.tensor->shape[i];
    }
    std::cout << "]\n";
  }
}

simaai::neat::Session make_session(int w, int h) {
  simaai::neat::Session p;
  simaai::neat::InputOptions in;
  in.format = "RGB";
  in.width = w;
  in.height = h;
  in.depth = 3;
  p.add(simaai::neat::nodes::Input(in));
  p.add(simaai::neat::nodes::Output());
  return p;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int w = parse_int_arg(argc, argv, "--width", 160);
    const int h = parse_int_arg(argc, argv, "--height", 120);

    cv::Mat img(h, w, CV_8UC3, cv::Scalar(10, 30, 200));
    if (!img.isContinuous())
      img = img.clone();

    simaai::neat::Session p = make_session(w, h);
    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    // 1) Pull a full Sample to access metadata.
    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    auto run = p.build(img, simaai::neat::RunMode::Sync, run_opt);
    simaai::neat::Sample s = run.push_and_pull(img, /*timeout_ms=*/1000);
    print_sample(s, "Sample from pull():");

    // 2) Pull only the tensor payload (simpler API).
    simaai::neat::Session p2 = make_session(w, h);
    auto run2 = p2.build(img, simaai::neat::RunMode::Sync, run_opt);
    sima_tutorial::require(run2.push(img), "push() failed for pull_tensor demo");
    auto t = run2.pull_tensor_or_throw(/*timeout_ms=*/1000);
    std::cout << "simaai::neat::Tensor only (pull_tensor_or_throw): shape dims=" << t.shape.size()
              << "\n";

    // 3) Bundle samples: typically returned when a pipeline has multiple outputs.
    // Here we construct a tiny synthetic example to show the structure.
    simaai::neat::Sample bundle =
        simaai::neat::make_bundle_sample({simaai::neat::make_tensor_sample("out0", t)});
    print_sample(bundle, "Synthetic bundle example:");

    std::cout << "[OK] tutorial_0004 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
