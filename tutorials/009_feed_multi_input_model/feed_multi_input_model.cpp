// Build a multi-port bundle Sample and push it through a tensor-in/tensor-out Session.
//
// Usage:
//   tutorial_v2_009_multi_input_samples [--width 64] [--height 48]

#include "neat.h"

#include <cstddef>
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

simaai::neat::Tensor make_fp32_tensor(int w, int h, int c, float fill) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * c * sizeof(float);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  auto* p = static_cast<float*>(map.data);
  const std::size_t n = static_cast<std::size_t>(w) * h * c;
  for (std::size_t i = 0; i < n; ++i)
    p[i] = fill;

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
    const int w = parse_int_arg(argc, argv, "--width", 64);
    const int h = parse_int_arg(argc, argv, "--height", 48);
    const int c = 3;

    // CORE LOGIC
    // Session accepting fp32 tensors as input.
    simaai::neat::Session session;
    simaai::neat::InputOptions in;
    in.media_type = "application/vnd.simaai.tensor";
    in.format = "FP32";
    in.width = w;
    in.height = h;
    in.depth = c;
    session.add(simaai::neat::nodes::Input(in));
    session.add(simaai::neat::nodes::Output());

    simaai::neat::Tensor seed = make_fp32_tensor(w, h, c, 0.0f);
    auto run = session.build(seed, simaai::neat::RunMode::Sync);

    // make_bundle_sample packs multiple named tensors into one Sample.
    simaai::neat::Sample bundle = simaai::neat::make_bundle_sample({
        simaai::neat::make_tensor_sample("left", make_fp32_tensor(w, h, c, 1.0f)),
        simaai::neat::make_tensor_sample("right", make_fp32_tensor(w, h, c, 2.0f)),
    });

    run.push(bundle);
    auto out = run.pull(/*timeout_ms=*/1000);
    // END CORE LOGIC

    if (!out.has_value())
      throw std::runtime_error("bundle output missing");
    if (out->kind != simaai::neat::SampleKind::Bundle)
      throw std::runtime_error("expected bundle output");

    std::cout << "bundle_fields=" << out->fields.size() << "\n";
    for (const auto& field : out->fields) {
      std::cout << "  port=" << field.port_name
                << " has_tensor=" << (field.tensor.has_value() ? "yes" : "no") << "\n";
    }
    std::cout << "[OK] 009_feed_multi_input_model\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
