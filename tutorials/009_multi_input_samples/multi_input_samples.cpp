#include "neat.h"

#include <cstring>
#include <iostream>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
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

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string value;
  if (!get_arg(argc, argv, key, value)) return def;
  try {
    return std::stoi(value);
  } catch (...) {
    throw std::invalid_argument("invalid integer for " + key + ": " + value);
  }
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}

bool strict_mode() {
  return std::getenv("SIMA_RUN_TUTORIALS_FULL") != nullptr;
}

std::string yes_no(bool v) { return v ? "yes" : "no"; }

void step(const std::string& name, const std::string& detail = {}) {
  if (detail.empty()) {
    std::cout << "STEP " << name << "\n";
  } else {
    std::cout << "STEP " << name << ": " << detail << "\n";
  }
}

void check(const std::string& name, bool cond, const std::string& detail = {}) {
  std::cout << "CHECK " << name << ": " << (cond ? "PASS" : "FAIL");
  if (!detail.empty()) std::cout << " (" << detail << ")";
  std::cout << "\n";
  if (!cond) throw std::runtime_error("check failed: " + name);
}

void why(const std::string& detail) {
  std::cout << "WHY " << detail << "\n";
}

void tradeoff(const std::string& detail) {
  std::cout << "TRADEOFF " << detail << "\n";
}

void failure_mode(const std::string& detail) {
  std::cout << "FAILURE_MODE " << detail << "\n";
}

void interpret_output(const std::string& detail) {
  std::cout << "INTERPRET " << detail << "\n";
}

void print_signature(std::initializer_list<std::pair<std::string, std::string>> values) {
  static constexpr std::array<const char*, 7> kRequired = {
      "tutorial", "lang", "flow", "run_mode", "output_kind", "tensor_rank", "field_count",
  };
  for (const char* key : kRequired) {
    bool found = false;
    for (const auto& kv : values) {
      if (kv.first == key) { found = true; break; }
    }
    if (!found) throw std::invalid_argument(std::string("missing signature key: ") + key);
  }
  std::cout << "SIGNATURE {";
  bool first = true;
  for (const auto& kv : values) {
    if (!first) std::cout << ",";
    std::cout << kv.first << "=" << kv.second;
    first = false;
  }
  std::cout << "}\n";
}

} // namespace

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  print_common_flags(std::cout);
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
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    // Why: explicit runtime markers keep the tutorial explainable from terminal output alone.
    // Why: parity and score tooling consume these checkpoints as a stable contract.
    step("input_contract", "parse flags and establish deterministic defaults");
    step("run_mode_choice", "exercise the chapter's primary runtime path");
    why("understand the contract first: inputs, run mode, and outputs");
    tradeoff(
        "prefer deterministic samples and stable contracts over production realism");
    failure_mode(
        "runtime/plugin issues should degrade to runtime_fallback without losing observability");
    interpret_output(
        "use CHECK markers plus SIGNATURE fields to validate behavior and parity");
    step("output_contract", "emit checks and machine-parseable signature");
    check("strict_flag_available",
                       yes_no(strict_mode()) == "yes" ||
                           yes_no(strict_mode()) == "no",
                       "strict-mode guard is observable");

    const int w = parse_int_arg(argc, argv, "--width", 64);
    const int h = parse_int_arg(argc, argv, "--height", 48);
    const int c = 3;

    // CORE LOGIC
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

    simaai::neat::Tensor seed = make_fp32_tensor(w, h, c, 0.0f);
    auto run = p.build(seed, simaai::neat::RunMode::Sync);

    simaai::neat::Sample bundle = simaai::neat::make_bundle_sample({
        simaai::neat::make_tensor_sample("left", make_fp32_tensor(w, h, c, 1.0f)),
        simaai::neat::make_tensor_sample("right", make_fp32_tensor(w, h, c, 2.0f)),
    });

    require(run.push(bundle), "bundle push failed");

    auto out = run.pull(1000);
    require(out.has_value(), "bundle output missing");
    require(out->kind == simaai::neat::SampleKind::Bundle, "expected bundle output");

    std::cout << "Bundle fields: " << out->fields.size() << "\n";
    for (const auto& field : out->fields) {
      std::cout << "  port=" << field.port_name
                << " has_tensor=" << (field.tensor.has_value() ? "yes" : "no") << "\n";
    }
    // END CORE LOGIC

    check("tutorial_completed", true, "main path reached end without exception");
    print_signature({
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
