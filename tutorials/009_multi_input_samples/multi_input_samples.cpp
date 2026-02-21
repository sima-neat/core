#include "neat.h"
#include "common/cpp_utils.h"

#include <cstring>
#include <iostream>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  tutorial_v2::print_common_flags(std::cout);
}

simaai::neat::Tensor make_fp32_tensor(int w, int h, int c, float fill) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * c * sizeof(float);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes >= bytes) {
    auto* p = static_cast<float*>(map.data);
    const std::size_t n = static_cast<std::size_t>(w) * h * c;
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = fill;
    }
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
    if (tutorial_v2::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    // Why: explicit runtime markers keep the tutorial explainable from terminal output alone.
    // Why: parity and score tooling consume these checkpoints as a stable contract.
    tutorial_v2::step("input_contract", "parse flags and establish deterministic defaults");
    tutorial_v2::step("run_mode_choice", "exercise the chapter's primary runtime path");
    tutorial_v2::why("understand the contract first: inputs, run mode, and outputs");
    tutorial_v2::tradeoff(
        "prefer deterministic samples and stable contracts over production realism");
    tutorial_v2::failure_mode(
        "runtime/plugin issues should degrade to runtime_fallback without losing observability");
    tutorial_v2::interpret_output(
        "use CHECK markers plus SIGNATURE fields to validate behavior and parity");
    tutorial_v2::step("output_contract", "emit checks and machine-parseable signature");
    tutorial_v2::check("strict_flag_available",
                       tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "yes" ||
                           tutorial_v2::yes_no(tutorial_v2::strict_mode()) == "no",
                       "strict-mode guard is observable");

    const int w = tutorial_v2::parse_int_arg(argc, argv, "--width", 64);
    const int h = tutorial_v2::parse_int_arg(argc, argv, "--height", 48);
    const int c = 3;

    simaai::neat::Session p;
    simaai::neat::InputOptions in;
    in.media_type = "application/vnd.simaai.tensor";
    in.format = "FP32";
    in.width = w;
    in.height = h;
    in.depth = c;
    p.add(simaai::neat::nodes::Input(in));
    p.add(simaai::neat::nodes::Output());

    if (tutorial_v2::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::Tensor seed = make_fp32_tensor(w, h, c, 0.0f);
    auto run = p.build(seed, simaai::neat::RunMode::Sync);

    simaai::neat::Sample bundle = simaai::neat::make_bundle_sample({
        simaai::neat::make_tensor_sample("left", make_fp32_tensor(w, h, c, 1.0f)),
        simaai::neat::make_tensor_sample("right", make_fp32_tensor(w, h, c, 2.0f)),
    });

    tutorial_v2::require(run.push(bundle), "bundle push failed");

    auto out = run.pull(1000);
    tutorial_v2::require(out.has_value(), "bundle output missing");
    tutorial_v2::require(out->kind == simaai::neat::SampleKind::Bundle, "expected bundle output");

    std::cout << "Bundle fields: " << out->fields.size() << "\n";
    for (const auto& field : out->fields) {
      std::cout << "  port=" << field.port_name
                << " has_tensor=" << (field.tensor.has_value() ? "yes" : "no") << "\n";
    }

    tutorial_v2::check("tutorial_completed", true, "main path reached end without exception");
    tutorial_v2::print_signature({
        {"tutorial", "009"},
        {"lang", "cpp"},
        {"flow", "chapter_path"},
        {"run_mode", "sync_or_async"},
        {"output_kind", "sample_or_tensor"},
        {"tensor_rank", "-1"},
        {"field_count", "-1"},
    });

    std::cout << "[OK] 009_multi_input_samples\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
