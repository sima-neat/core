#include "nodes/io/CameraInput.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace n = simaai::neat;
namespace {
template <typename T>
concept PublicFourCC = requires(T options) { options.fourcc; };
static_assert(!PublicFourCC<n::CameraInputOptions>);
unsigned checks = 0;
void check(bool value) {
  ++checks;
  if (!value)
    throw std::runtime_error("camera node check failed");
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
    n::CameraInputOptions defaults;
    check(defaults.profile == n::CameraProfile::Default);
    n::CameraInput default_camera(defaults);
    check(!default_camera.options().zero_copy.has_value());
    check(!default_camera.options().allow_cpu_fallback);
    check(default_camera.memory_contract() == n::MemoryContract::PreferDeviceZeroCopy);
    check(default_camera.output_spec({}).payload_type == n::PayloadType::Image);
    auto copy_defaults = defaults;
    copy_defaults.zero_copy = false;
    check(n::CameraInput(copy_defaults).backend_fragment(0).find("libcamerasrc") == 0);
    check(n::CameraInput(copy_defaults).options().allow_cpu_fallback);
    auto strict_defaults = defaults;
    strict_defaults.zero_copy = true;
    check(!n::CameraInput(strict_defaults).options().allow_cpu_fallback);
    strict_defaults.allow_cpu_fallback = true;
    rejects([&] { n::nodes::CameraInput(strict_defaults); });
    defaults.device = "/dev/not-a-camera-must-not-be-opened";
    rejects([&] { n::CameraInput invalid(defaults); });
    rejects([&] { n::nodes::CameraInput(defaults); });
    defaults.device.clear();
    defaults.profile = static_cast<n::CameraProfile>(2);
    rejects([&] { n::nodes::CameraInput(defaults); });
    defaults.profile = static_cast<n::CameraProfile>(999);
    rejects([&] { n::nodes::CameraInput(defaults); });
    // Format alone is not a backend selector. Routing only, not Bayer negotiation.
    auto raw_format = copy_defaults;
    raw_format.format = "RAW8";
    check(n::CameraInput(raw_format).backend_fragment(0).find("libcamerasrc") == 0);
    check(n::CameraInput(raw_format).output_spec({}).payload_type == n::PayloadType::Image);
    auto factory = &n::nodes::CameraInput;
    check(factory({})->kind() == "CameraInput");

    n::CameraInputOptions options;
    options.profile = n::CameraProfile::MetoakSimor;
    options.device = "/dev/video3";
    options.buffer_name = "raw_src";
#if !defined(__linux__)
    bool unavailable = false;
    try {
      factory(options);
    } catch (const std::runtime_error& error) {
      unavailable = std::string(error.what()).find("requires Linux") != std::string::npos;
    }
    check(unavailable);
#else
    // The profile alone owns BA81 and the default copy policy; no backend options.
    auto camera = factory(options);
    n::CameraInput direct(options);
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
    check(direct.backend_fragment(4) == fragment);
    check(direct.options().width == 1920 && direct.options().height == 360);
    check(direct.options().format == "RAW8");
    check(direct.options().zero_copy == false);
    check(direct.options().capture_buffer_count == 8);
    check(direct.memory_contract() == n::MemoryContract::RequireSystemMemoryMappable);
    const auto spec = direct.output_spec({});
    check(spec.payload_type == n::PayloadType::Tensor && spec.format == "V4L2_BYTES");
    check(spec.width == 1920 && spec.height == 360 && spec.dtype == "UInt8" &&
          spec.memory == "SystemMemory");
    auto explicit_copy = options;
    explicit_copy.zero_copy = false;
    check(factory(explicit_copy)->backend_fragment(4) == fragment);
    explicit_copy.height = 360;
    explicit_copy.format = "RAW8";
    check(factory(explicit_copy)->backend_fragment(4) == fragment);
    options.insert_queue = false;
    check(factory(options)->element_names(0).size() == 1);
    check(factory(options)->backend_fragment(0).find(" ! ") == std::string::npos);
    options.capture_buffer_count = 12;
    check(factory(options)->backend_fragment(0).find("capture-buffer-count=12") !=
          std::string::npos);
    check(n::nodes::CameraInputWithCaptureBuffers(options, 10)
              ->backend_fragment(0)
              .find("capture-buffer-count=10") != std::string::npos);
    // All invalid options fail before discovery, even when no device is given.
    options.device.clear();
    const auto invalid = [&](const std::function<void(n::CameraInputOptions&)>& mutate) {
      auto bad = options;
      mutate(bad);
      rejects([&] { factory(bad); });
    };
    invalid([](auto& opt) { opt.zero_copy = true; });
    invalid([](auto& opt) { opt.format = "RGB"; });
    invalid([](auto& opt) { opt.width = 0; });
    invalid([](auto& opt) { opt.width = 640; });
    invalid([](auto& opt) { opt.height = 480; });
    invalid([](auto& opt) { opt.allow_cpu_fallback = true; });
    invalid([](auto& opt) { opt.camera_name = "wrong-source"; });
    invalid([](auto& opt) { opt.buffer_name.clear(); });
    invalid([](auto& opt) { opt.capture_buffer_count = 3; });
    invalid([](auto& opt) { opt.capture_buffer_count = 129; });
    invalid([](auto& opt) { opt.output_buffer_count = 1; });
    invalid([](auto& opt) { opt.output_buffer_count = 129; });
    invalid([](auto& opt) { opt.frame_timeout_ms = 0; });
    invalid([](auto& opt) { opt.frame_timeout_ms = 60001; });
    invalid([](auto& opt) {
      opt.insert_queue = true;
      opt.queue_depth = 0;
    });
    invalid([](auto& opt) {
      opt.insert_queue = true;
      opt.queue_depth = opt.output_buffer_count;
    });
#endif
    std::cout << "PASS camera node " << checks << " checks\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
