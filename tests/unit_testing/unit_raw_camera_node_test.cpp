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
    auto camera = n::nodes::CameraInput(options, backend);
    check(camera->kind() == "CameraInput");
    check(camera->input_role() == n::InputRole::Source);
    check(camera->buffer_name_hint(4) == "raw_src");
    const auto fragment = camera->backend_fragment(4);
    check(fragment.find("neatrawcamerasrc name=n4_camera_src") == 0);
    check(fragment.find("device=\"/dev/video3\"") != std::string::npos);
    check(fragment.find("fourcc=\"BA81\"") != std::string::npos);
    check(fragment.find("libcamera") == std::string::npos);
    check(fragment.find("width=1920 height=360") != std::string::npos);
    check(camera->element_names(4).size() == 2);
    const auto* provider = dynamic_cast<const n::OutputSpecProvider*>(camera.get());
    check(provider != nullptr);
    const auto spec = provider->output_spec({});
    check(spec.payload_type == n::PayloadType::Tensor && spec.format == "RAW_CAMERA_U8");
    check(spec.width == 1920 && spec.height == 360 && spec.dtype == "UInt8" && spec.layout == "HW");
    options.insert_queue = false;
    camera = n::nodes::CameraInput(options, backend);
    check(camera->element_names(0).size() == 1);
    check(camera->backend_fragment(0).find(" ! ") == std::string::npos);
    auto bad = options;
    bad.format = "NV12";
    rejects([&] { n::nodes::CameraInput(bad, backend); });
    bad = options;
    bad.width = 0;
    rejects([&] { n::nodes::CameraInput(bad, backend); });
    bad = options;
    bad.allow_cpu_fallback = true;
    rejects([&] { n::nodes::CameraInput(bad, backend); });
    bad = options;
    bad.camera_name = "wrong-source";
    rejects([&] { n::nodes::CameraInput(bad, backend); });
    bad = options;
    bad.buffer_name.clear();
    rejects([&] { n::nodes::CameraInput(bad, backend); });
    auto bad_backend = backend;
    bad_backend.device.clear();
    rejects([&] { n::nodes::CameraInput(options, bad_backend); });
    bad_backend = backend;
    bad_backend.fourcc = "RGB";
    rejects([&] { n::nodes::CameraInput(options, bad_backend); });
    bad_backend = backend;
    bad_backend.capture_buffer_count = 129;
    rejects([&] { n::nodes::CameraInput(options, bad_backend); });
    bad_backend = backend;
    bad_backend.capture_buffer_count = 3;
    rejects([&] { n::nodes::CameraInput(options, bad_backend); });
    std::cout << "PASS raw camera node " << checks << " checks\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
