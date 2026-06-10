// Build a multi-port bundle Sample and push it through a tensor-in/tensor-out Graph.
//
// Usage:
//   tutorial_010_feed_multi_input_model [--width 64] [--height 48]

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

    // STEP configure-tensor-input
    simaai::neat::InputOptions in;
    in.payload_type = simaai::neat::PayloadType::Tensor;
    in.format = "FP32";
    in.width = w;
    in.height = h;
    in.depth = c;
    // END STEP

    simaai::neat::Tensor seed = make_fp32_tensor(w, h, c, 0.0f);

    // CORE LOGIC
    // STEP build-seed-run
    // Graph accepting fp32 tensors as input.
    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::Input(in));
    graph.add(simaai::neat::nodes::Output());
    auto run = graph.build(simaai::neat::TensorList{seed});
    // END STEP

    // STEP make-bundle
    // make_bundle_sample packs multiple named tensors into one Sample.
    simaai::neat::Sample bundle = simaai::neat::make_bundle_sample({
        simaai::neat::make_tensor_sample("left", make_fp32_tensor(w, h, c, 1.0f)),
        simaai::neat::make_tensor_sample("right", make_fp32_tensor(w, h, c, 2.0f)),
    });
    // END STEP

    // STEP push-and-read
    auto outs = run.run(simaai::neat::Sample{bundle}, /*timeout_ms=*/1000);
    // END STEP
    // END CORE LOGIC

    if (outs.empty())
      throw std::runtime_error("bundle output missing");
    // `Run::run(Sample)` returns one Sample. When the logical result has multiple fields,
    // that Sample is itself a Bundle; `front()` would mean "first field inside the bundle",
    // not "first output sample".
    const simaai::neat::Sample& out = outs;
    if (out.kind != simaai::neat::SampleKind::Bundle)
      throw std::runtime_error("expected bundle output");
    if (out.fields.size() != 2U)
      throw std::runtime_error("expected two bundle fields");

    std::cout << "bundle_fields=" << out.fields.size() << "\n";
    for (std::size_t i = 0; i < out.fields.size(); ++i) {
      const auto& field = out.fields[i];
      const bool has_tensor = field.tensor.has_value() || !field.tensors.empty();
      const std::string label =
          !field.port_name.empty()
              ? field.port_name
              : (!field.stream_label.empty() ? field.stream_label : ("field_" + std::to_string(i)));
      std::cout << "  field=" << label << " has_tensor=" << (has_tensor ? "yes" : "no") << "\n";
    }
    std::cout << "[OK] 010_feed_multi_input_model\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
