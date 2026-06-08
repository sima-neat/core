// Multistream public Graph: named inputs -> Combine(ByFrame) -> named output bundle.
//
// Usage:
//   tutorial_014_run_multiple_streams [--streams 8] [--frames 4]

#include "neat.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

std::vector<int64_t> contiguous_strides_bytes(const std::vector<int64_t>& shape,
                                              int64_t elem_bytes) {
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = elem_bytes;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

simaai::neat::Sample make_rgb_sample(const std::string& stream_id, int frame_id) {
  const int w = 8;
  const int h = 6;
  const int c = 3;
  const std::size_t bytes = static_cast<std::size_t>(w) * h * c;

  simaai::neat::Tensor t;
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, c};
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::RGB, ""};
  t.storage = simaai::neat::make_cpu_owned_storage(bytes);
  t.strides_bytes = contiguous_strides_bytes(t.shape, 1);
  t.read_only = false;
  {
    auto map = t.map(simaai::neat::MapMode::Write);
    auto* p = static_cast<std::uint8_t*>(map.data);
    for (std::size_t i = 0; i < bytes; ++i)
      p[i] = static_cast<std::uint8_t>(i % 255);
  }
  t.read_only = true;

  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = std::move(t);
  sample.frame_id = frame_id;
  sample.stream_id = stream_id;
  return sample;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const int streams = parse_int_arg(argc, argv, "--streams", 8);
    const int frames = parse_int_arg(argc, argv, "--frames", 4);

    // CORE LOGIC
    // `graphs::Combine` is a normal public Graph fragment. It declares two
    // named inputs ("left", "right") and one named output ("combined"). ByFrame
    // means the runtime emits one bundle only after both inputs have delivered
    // samples with the same Sample::frame_id.
    // STEP build-combine-graph
    simaai::neat::Graph graph = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                              simaai::neat::CombinePolicy::ByFrame);

    std::cout << graph.describe() << "\n";

    simaai::neat::Run run = graph.build();
    // END STEP

    // STEP push-streams
    for (int frame = 0; frame < frames; ++frame) {
      for (int sid = 0; sid < streams; ++sid) {
        const int logical_frame = frame * streams + sid;
        if (!run.push("left", make_rgb_sample(std::to_string(sid), logical_frame))) {
          throw std::runtime_error("left push failed: " + run.last_error());
        }
        if (!run.push("right", make_rgb_sample(std::to_string(sid), logical_frame))) {
          throw std::runtime_error("right push failed: " + run.last_error());
        }
      }
    }
    // END STEP

    // STEP pull-bundles
    const int expected = streams * frames;
    int received = 0;
    int first_fields = -1;
    for (int i = 0; i < expected; ++i) {
      auto maybe_bundle = run.pull("combined", /*timeout_ms=*/2000);
      if (!maybe_bundle.has_value()) {
        throw std::runtime_error("timed out waiting for combined output");
      }
      const auto& bundle = *maybe_bundle;
      if (first_fields < 0)
        first_fields = static_cast<int>(bundle.fields.size());
      ++received;
      if (i < 4) {
        std::cout << "bundle stream=" << bundle.stream_id << " fields=" << bundle.fields.size()
                  << "\n";
      }
    }

    run.close();
    // END STEP
    // END CORE LOGIC

    if (received != expected)
      throw std::runtime_error("expected=" + std::to_string(expected) +
                               " received=" + std::to_string(received));
    if (first_fields != 2)
      throw std::runtime_error("join should emit an image+bbox bundle");

    std::cout << "received=" << received << " fields=" << first_fields << "\n";
    std::cout << "[OK] 014_run_multiple_streams\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
