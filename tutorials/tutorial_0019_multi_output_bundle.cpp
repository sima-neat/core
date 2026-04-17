// tutorial_0019_multi_output_bundle.cpp
// Story: bundle samples carry multiple outputs through a single pipeline.
// What you learn:
// - SampleKind::Bundle preserves multiple fields.
// - Bundles can be pushed/pulled through Input pipelines.

#include "neat/session.h"
#include "neat/nodes.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

bool wants_help(int argc, char** argv) {
  return has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h");
}

bool wants_print_gst(int argc, char** argv) {
  return has_flag(argc, argv, "--print-gst");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

void require(bool ok, const std::string& msg) {
  if (!ok)
    throw std::runtime_error(msg);
}

} // namespace

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  print_common_flags(std::cout);
  std::cout << "  --width <w>          Width (default 64)\n";
  std::cout << "  --height <h>         Height (default 48)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

simaai::neat::Tensor make_fp32_tensor(int w, int h, int c, float value) {
  const std::size_t bytes = static_cast<std::size_t>(w * h * c) * sizeof(float);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes >= bytes) {
    float* buf = static_cast<float*>(map.data);
    const std::size_t count = static_cast<std::size_t>(w * h * c);
    for (std::size_t i = 0; i < count; ++i)
      buf[i] = value;
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, c};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  return t;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int w = parse_int_arg(argc, argv, "--width", 64);
    const int h = parse_int_arg(argc, argv, "--height", 48);
    const int c = 3;

    // Build a tensor-mode pipeline.
    simaai::neat::Session p;
    simaai::neat::InputOptions in;
    in.media_type = "application/vnd.simaai.tensor";
    in.format = "FP32";
    in.width = w;
    in.height = h;
    in.depth = c;
    p.add(simaai::neat::nodes::Input(in));
    p.add(simaai::neat::nodes::Output());

    if (wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    // Seed build caps with a single tensor input.
    simaai::neat::Tensor seed = make_fp32_tensor(w, h, c, 0.0f);
    auto run = p.build(seed, simaai::neat::RunMode::Sync, run_opt);

    // Create two output fields and push them as a bundle.
    simaai::neat::Tensor t0 = make_fp32_tensor(w, h, c, 1.0f);
    simaai::neat::Tensor t1 = make_fp32_tensor(w, h, c, 2.0f);

    simaai::neat::Sample bundle = simaai::neat::make_bundle_sample({
        simaai::neat::make_tensor_sample("out0", t0),
        simaai::neat::make_tensor_sample("out1", t1),
    });

    run.push(bundle);
    auto out_opt = run.pull(/*timeout_ms=*/1000);
    require(out_opt.has_value(), "missing bundle output");
    simaai::neat::Sample out = *out_opt;
    std::cout << "Output kind: " << static_cast<int>(out.kind) << "\n";
    std::cout << "Bundle fields: " << out.fields.size() << "\n";
    for (const auto& field : out.fields) {
      std::cout << "  field name=" << field.port_name
                << " shape_dims=" << (field.tensor ? field.tensor->shape.size() : 0) << "\n";
    }

    std::cout << "[OK] tutorial_0019 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
