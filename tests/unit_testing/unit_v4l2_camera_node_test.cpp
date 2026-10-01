#include "nodes/io/CameraInput.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace n = simaai::neat;
namespace {
unsigned checks = 0;
void check(bool value) {
  ++checks;
  if (!value)
    throw std::runtime_error("raw camera node check failed");
}
void rejects(const std::function<void()>& action) {
  bool threw = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw);
}
} // namespace
int main() {
  try {
    n::CameraInputOptions options;
    options.width = 1920;
    options.height = 360;
    options.format = "RAW8";
    options.buffer_name = "raw_src";
    n::CameraV4L2Options backend;
    backend.device = "/dev/video3";
    backend.fourcc = "BA81";
    auto camera = n::nodes::CameraInputWithV4L2(options, backend);
    check(camera->kind() == "CameraInput");
    check(camera->input_role() == n::InputRole::Source);
    check(camera->buffer_name_hint(4) == "raw_src");
    const auto fragment = camera->backend_fragment(4);
    check(fragment.find("neatv4l2copysrc name=n4_camera_src") == 0);
    check(fragment.find("device=\"/dev/video3\"") != std::string::npos);
    check(fragment.find("fourcc=\"BA81\"") != std::string::npos);
    check(fragment.find("libcamera") == std::string::npos);
    check(fragment.find("width=1920 height=360") != std::string::npos);
    check(camera->element_names(4).size() == 2);
    const auto* provider = dynamic_cast<const n::OutputSpecProvider*>(camera.get());
    check(provider != nullptr);
    const auto spec = provider->output_spec({});
    check(spec.payload_type == n::PayloadType::Tensor && spec.format == "V4L2_BYTES");
    check(spec.width == 1920 && spec.height == 360 && spec.dtype == "UInt8" &&
          spec.memory == "SystemMemory");
    options.insert_queue = false;
    camera = n::nodes::CameraInputWithV4L2(options, backend);
    check(camera->element_names(0).size() == 1);
    check(camera->backend_fragment(0).find(" ! ") == std::string::npos);
    auto bad = options;
    bad.format = "NV12";
    rejects([&] { n::nodes::CameraInputWithV4L2(bad, backend); });
    bad = options;
    bad.width = 0;
    rejects([&] { n::nodes::CameraInputWithV4L2(bad, backend); });
    bad = options;
    bad.allow_cpu_fallback = true;
    rejects([&] { n::nodes::CameraInputWithV4L2(bad, backend); });
    bad = options;
    bad.camera_name = "wrong-source";
    rejects([&] { n::nodes::CameraInputWithV4L2(bad, backend); });
    bad = options;
    bad.buffer_name.clear();
    rejects([&] { n::nodes::CameraInputWithV4L2(bad, backend); });
    auto bad_backend = backend;
    bad_backend.device.clear();
    rejects([&] { n::nodes::CameraInputWithV4L2(options, bad_backend); });
    bad_backend = backend;
    bad_backend.fourcc = "RGB";
    rejects([&] { n::nodes::CameraInputWithV4L2(options, bad_backend); });
    bad_backend = backend;
    bad_backend.capture_buffer_count = 129;
    rejects([&] { n::nodes::CameraInputWithV4L2(options, bad_backend); });
    bad_backend = backend;
    bad_backend.capture_buffer_count = 3;
    rejects([&] { n::nodes::CameraInputWithV4L2(options, bad_backend); });
    bad_backend = backend;
    bad_backend.zero_copy = true;
    rejects([&] { n::nodes::CameraInputWithV4L2(options, bad_backend); });
    bad_backend = backend;
    bad_backend.output_buffer_count = 0;
    rejects([&] { n::nodes::CameraInputWithV4L2(options, bad_backend); });
    bad_backend = backend;
    bad_backend.frame_timeout_ms = 0;
    rejects([&] { n::nodes::CameraInputWithV4L2(options, bad_backend); });
    bad_backend = backend;
    bad_backend.fourcc = "NV12";
    rejects([&] { n::nodes::CameraInputWithV4L2(options, bad_backend); });
    bad = options;
    bad.insert_queue = true;
    bad.queue_depth = 0;
    rejects([&] { n::nodes::CameraInputWithV4L2(bad, backend); });
    // One-argument source keeps its existing layout/API and libcamera behavior.
    // Keep untyped address-taking source-compatible with the original factory.
    auto factory = &n::nodes::CameraInput;
    auto legacy = factory({});
    check(legacy->memory_contract() == n::MemoryContract::PreferDeviceZeroCopy);
    check(camera->memory_contract() == n::MemoryContract::RequireSystemMemoryMappable);
    std::cout << "PASS raw camera node " << checks << " checks\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
