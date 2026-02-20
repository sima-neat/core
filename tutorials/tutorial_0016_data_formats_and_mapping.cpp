// tutorial_0016_data_formats_and_mapping.cpp
// Story: how format tokens map to simaai::neat::Tensor layout/planes, and how to read data.
// What you learn:
// - RGB/BGR/GRAY8 are dense tensors (HWC/HW).
// - NV12/I420 are composite tensors with explicit planes.
// - map_nv12_read() / map_i420_read() provide safe access.

#include "neat/session.h"
#include "neat/nodes.h"

#include "tutorial_common.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --width <w>          Width (default 64)\n";
  std::cout << "  --height <h>         Height (default 48)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!sima_tutorial::get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

simaai::neat::Tensor make_rgb_tensor(int w, int h, uint8_t value) {
  const std::size_t bytes = static_cast<std::size_t>(w * h * 3);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes >= bytes) {
    std::memset(map.data, value, bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, 3};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::RGB, ""};
  return t;
}

simaai::neat::Tensor make_nv12_tensor(int w, int h, uint8_t value) {
  const std::size_t y_bytes = static_cast<std::size_t>(w * h);
  const std::size_t uv_bytes = static_cast<std::size_t>(w * h / 2);
  auto storage = simaai::neat::make_cpu_owned_storage(y_bytes + uv_bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes >= y_bytes + uv_bytes) {
    std::memset(map.data, value, y_bytes + uv_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HW;
  t.shape = {h, w};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::NV12, ""};

  simaai::neat::Plane y;
  y.role = simaai::neat::PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;

  simaai::neat::Plane uv;
  uv.role = simaai::neat::PlaneRole::UV;
  uv.shape = {h / 2, w};
  uv.strides_bytes = {w, 1};
  uv.byte_offset = static_cast<int64_t>(y_bytes);

  t.planes = {y, uv};
  return t;
}

simaai::neat::Tensor make_i420_tensor(int w, int h, uint8_t value) {
  const std::size_t y_bytes = static_cast<std::size_t>(w * h);
  const std::size_t u_bytes = static_cast<std::size_t>(w * h / 4);
  const std::size_t v_bytes = static_cast<std::size_t>(w * h / 4);
  auto storage = simaai::neat::make_cpu_owned_storage(y_bytes + u_bytes + v_bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes >= y_bytes + u_bytes + v_bytes) {
    std::memset(map.data, value, y_bytes + u_bytes + v_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HW;
  t.shape = {h, w};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::I420, ""};

  simaai::neat::Plane y;
  y.role = simaai::neat::PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;

  simaai::neat::Plane u;
  u.role = simaai::neat::PlaneRole::U;
  u.shape = {h / 2, w / 2};
  u.strides_bytes = {w / 2, 1};
  u.byte_offset = static_cast<int64_t>(y_bytes);

  simaai::neat::Plane v;
  v.role = simaai::neat::PlaneRole::V;
  v.shape = {h / 2, w / 2};
  v.strides_bytes = {w / 2, 1};
  v.byte_offset = static_cast<int64_t>(y_bytes + u_bytes);

  t.planes = {y, u, v};
  return t;
}

simaai::neat::Tensor run_passthrough(const simaai::neat::Tensor& input) {
  simaai::neat::Session p;
  simaai::neat::InputOptions in;
  in.media_type = "video/x-raw";
  p.add(simaai::neat::nodes::Input(in));
  p.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;

  auto run = p.build(input, simaai::neat::RunMode::Sync, run_opt);
  simaai::neat::Sample out = run.push_and_pull(input, /*timeout_ms=*/1000);
  sima_tutorial::require(out.tensor.has_value(), "missing output tensor");
  return *out.tensor;
}

void print_tensor_summary(const simaai::neat::Tensor& t, const std::string& label) {
  std::cout << label << "\n";
  std::cout << "  dtype: " << static_cast<int>(t.dtype) << "\n";
  std::cout << "  layout: " << static_cast<int>(t.layout) << "\n";
  std::cout << "  shape: [";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (i)
      std::cout << ", ";
    std::cout << t.shape[i];
  }
  std::cout << "]\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int w = parse_int_arg(argc, argv, "--width", 64);
    const int h = parse_int_arg(argc, argv, "--height", 48);
    if ((w % 2) != 0 || (h % 2) != 0) {
      throw std::runtime_error("NV12/I420 require even width/height");
    }

    const simaai::neat::Tensor rgb = make_rgb_tensor(w, h, 0x2A);
    const simaai::neat::Tensor nv12 = make_nv12_tensor(w, h, 0x10);
    const simaai::neat::Tensor i420 = make_i420_tensor(w, h, 0x22);

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      simaai::neat::Session p;
      p.add(simaai::neat::nodes::Input());
      p.add(simaai::neat::nodes::Output());
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    // RGB: dense HWC
    simaai::neat::Tensor out_rgb = run_passthrough(rgb);
    print_tensor_summary(out_rgb, "RGB output:");
    {
      auto map = out_rgb.map_read();
      const uint8_t* bytes = static_cast<const uint8_t*>(map.data);
      std::cout << "  first bytes: " << static_cast<int>(bytes[0]) << ", "
                << static_cast<int>(bytes[1]) << ", " << static_cast<int>(bytes[2]) << "\n";
    }

    // NV12: composite planes
    simaai::neat::Tensor out_nv12 = run_passthrough(nv12);
    print_tensor_summary(out_nv12, "NV12 output:");
    {
      auto mapped = out_nv12.map_nv12_read();
      sima_tutorial::require(mapped.has_value(), "map_nv12_read failed");
      std::cout << "  Y[0]: " << static_cast<int>(mapped->view.y[0]) << "\n";
      std::cout << "  UV[0]: " << static_cast<int>(mapped->view.uv[0]) << "\n";
    }

    // I420: composite planes
    simaai::neat::Tensor out_i420 = run_passthrough(i420);
    print_tensor_summary(out_i420, "I420 output:");
    {
      auto mapped = out_i420.map_i420_read();
      sima_tutorial::require(mapped.has_value(), "map_i420_read failed");
      std::cout << "  Y[0]: " << static_cast<int>(mapped->view.y[0]) << "\n";
      std::cout << "  U[0]: " << static_cast<int>(mapped->view.u[0]) << "\n";
      std::cout << "  V[0]: " << static_cast<int>(mapped->view.v[0]) << "\n";
    }

    std::cout << "[OK] tutorial_0016 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
