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
    n::CameraInputOptions defaults;
    check(defaults.profile == n::CameraProfile::Default);
    n::CameraInput default_camera(defaults);
    // Backend selection is independent of the installed libcamera plugin's
    // optional external-buffer support. Keep its strict zero-copy guard intact.
    auto copy_defaults = defaults;
    copy_defaults.zero_copy = false;
    check(n::CameraInput(copy_defaults).backend_fragment(0).find("libcamerasrc") == 0);
    check(default_camera.output_spec({}).payload_type == n::PayloadType::Image);
    defaults.device = "/dev/not-a-camera-must-not-be-opened";
    rejects([&] { n::CameraInput invalid(defaults); });
    rejects([&] { n::nodes::CameraInput(defaults); });
    defaults.device.clear();
    defaults.profile = static_cast<n::CameraProfile>(999);
    rejects([&] { n::nodes::CameraInput(defaults); });
    n::CameraInputOptions options;
    options.width = 1920;
    options.height = 360;
    options.format = "RAW8";
    options.buffer_name = "raw_src";
    n::CameraV4L2Options backend;
    backend.device = "/dev/video3";
    backend.fourcc = "BA81";
#if !defined(__linux__)
    bool unavailable = false;
    try {
      n::nodes::CameraInputWithV4L2(options, backend);
    } catch (const std::runtime_error& error) {
      unavailable = std::string(error.what()).find("requires Linux") != std::string::npos;
    }
    check(unavailable);
    auto factory = &n::nodes::CameraInput;
    check(factory({})->kind() == "CameraInput");
    std::cout << "PASS V4L2 unavailable without changing legacy camera API\n";
#else
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
    // The original factory and direct public constructor must use one resolver.
    auto unified = options;
    unified.device = backend.device;
    unified.profile = n::CameraProfile::Raw;
    unified.zero_copy = false;
    unified.fourcc = backend.fourcc;
    auto standard = n::nodes::CameraInput(unified);
    n::CameraInput direct(unified);
    check(standard->backend_fragment(0) == camera->backend_fragment(0));
    check(direct.backend_fragment(0) == camera->backend_fragment(0));
    check(direct.memory_contract() == n::MemoryContract::RequireSystemMemoryMappable);
    check(direct.output_spec({}).payload_type == n::PayloadType::Tensor);
    n::CameraInputOptions profile;
    profile.profile = n::CameraProfile::MetoakSimor;
    profile.device = "/dev/video3";
    profile.zero_copy = false;
    n::CameraInput selected(profile);
    check(selected.options().width == 1920 && selected.options().height == 360);
    check(selected.options().format == "RAW8" && selected.options().fourcc == "BA81");
    check(selected.options().profile == n::CameraProfile::MetoakSimor);
    check(selected.backend_fragment(0).find("libcamera") == std::string::npos);
    profile.zero_copy = true;
    rejects([&] { n::nodes::CameraInput(profile); });
    profile.zero_copy.reset();
    rejects([&] { n::nodes::CameraInput(profile); });
    profile.zero_copy = false;
    profile.profile = n::CameraProfile::Default;
    rejects([&] { n::nodes::CameraInput(profile); });
    profile.profile = n::CameraProfile::MetoakSimor;
    profile.width = 640;
    rejects([&] { n::nodes::CameraInput(profile); });
    auto no_device = unified;
    no_device.device.clear();
    rejects([&] { n::nodes::CameraInput(no_device); });
    unified.capture_buffer_count = 12;
    check(n::nodes::CameraInput(unified)->backend_fragment(0).find("capture-buffer-count=12") !=
          std::string::npos);
    check(n::nodes::CameraInputWithCaptureBuffers(unified, 10)
              ->backend_fragment(0)
              .find("capture-buffer-count=10") != std::string::npos);
    n::CameraInputOptions legacy_copy;
    legacy_copy.zero_copy = false;
    check(n::CameraInput(legacy_copy).options().allow_cpu_fallback);
    legacy_copy.zero_copy = true;
    legacy_copy.allow_cpu_fallback = true;
    rejects([&] { n::CameraInput invalid(legacy_copy); });
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
    bad_backend.capture_buffer_count = 0;
    rejects([&] { n::nodes::CameraInputWithV4L2(options, bad_backend); });
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
    // One-argument source keeps its existing entry point and libcamera behavior.
    // Keep untyped address-taking source-compatible with the original factory.
    auto factory = &n::nodes::CameraInput;
    auto legacy = factory({});
    check(legacy->memory_contract() == n::MemoryContract::PreferDeviceZeroCopy);
    check(camera->memory_contract() == n::MemoryContract::RequireSystemMemoryMappable);
    std::cout << "PASS raw camera node " << checks << " checks\n";
#endif
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
