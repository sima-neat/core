#include "simaai/neat/pcie/SimaPCIeHost.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

struct Args {
  std::string model;
  std::string raw_image;
  std::string card_host;
  std::string format = "RGB";
  int width = 0;
  int height = 0;
  int queue = 0;
};

std::string value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

Args parse(int argc, char** argv) {
  Args out;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model") {
      out.model = value(argc, argv, i, "--model");
    } else if (arg == "--raw-image") {
      out.raw_image = value(argc, argv, i, "--raw-image");
    } else if (arg == "--card-host") {
      out.card_host = value(argc, argv, i, "--card-host");
    } else if (arg == "--format") {
      out.format = value(argc, argv, i, "--format");
    } else if (arg == "--width") {
      out.width = std::stoi(value(argc, argv, i, "--width"));
    } else if (arg == "--height") {
      out.height = std::stoi(value(argc, argv, i, "--height"));
    } else if (arg == "--queue") {
      out.queue = std::stoi(value(argc, argv, i, "--queue"));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (out.model.empty() || out.raw_image.empty() || out.width <= 0 || out.height <= 0) {
    throw std::runtime_error("--model, --raw-image, --width, and --height are required");
  }
  return out;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open image payload: " + path);
  }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string upper_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return value;
}

pcie::PixelFormat pixel_format(const std::string& format) {
  const std::string v = upper_copy(format);
  if (v == "RGB")
    return pcie::PixelFormat::RGB;
  if (v == "BGR")
    return pcie::PixelFormat::BGR;
  if (v == "GRAY" || v == "GRAY8")
    return pcie::PixelFormat::GRAY8;
  if (v == "NV12")
    return pcie::PixelFormat::NV12;
  if (v == "I420" || v == "IYUV")
    return pcie::PixelFormat::I420;
  throw std::runtime_error("unsupported image format: " + format);
}

std::size_t expected_image_bytes(const std::string& format, const int width, const int height) {
  const std::string v = upper_copy(format);
  const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (v == "RGB" || v == "BGR")
    return pixels * 3U;
  if (v == "GRAY" || v == "GRAY8")
    return pixels;
  if (v == "NV12" || v == "I420" || v == "IYUV")
    return (pixels * 3U) / 2U;
  throw std::runtime_error("unsupported image format: " + format);
}

pcie::Tensor make_image_tensor(const std::vector<std::uint8_t>& bytes, const std::string& format,
                               const int width, const int height) {
  const std::size_t expected = expected_image_bytes(format, width, height);
  if (bytes.size() != expected) {
    throw std::runtime_error("image payload size mismatch: got=" + std::to_string(bytes.size()) +
                             " expected=" + std::to_string(expected));
  }

  auto owner = std::make_shared<std::vector<std::uint8_t>>(bytes);
  pcie::Tensor tensor;
  tensor.dtype = pcie::TensorDType::UInt8;
  tensor.owner = owner;
  tensor.data = owner->data();
  tensor.size_bytes = owner->size();
  tensor.read_only = false;
  tensor.image_format = pixel_format(format);
  tensor.route.name = "input_image";

  const std::string v = upper_copy(format);
  if (v == "RGB" || v == "BGR") {
    tensor.layout = pcie::TensorLayout::HWC;
    tensor.shape = {height, width, 3};
    tensor.strides_bytes = {static_cast<std::int64_t>(width * 3), 3, 1};
  } else if (v == "GRAY" || v == "GRAY8") {
    tensor.layout = pcie::TensorLayout::HW;
    tensor.shape = {height, width};
    tensor.strides_bytes = {width, 1};
  } else if (v == "NV12") {
    tensor.layout = pcie::TensorLayout::Unknown;
    tensor.shape = {height, width};
    tensor.planes = {
        pcie::Plane{.role = pcie::PlaneRole::Y,
                    .shape = {height, width},
                    .strides_bytes = {width, 1},
                    .byte_offset = 0},
        pcie::Plane{.role = pcie::PlaneRole::UV,
                    .shape = {height / 2, width},
                    .strides_bytes = {width, 1},
                    .byte_offset = static_cast<std::int64_t>(width * height)},
    };
  } else {
    tensor.layout = pcie::TensorLayout::Unknown;
    tensor.shape = {height, width};
    const auto y_size = static_cast<std::int64_t>(width * height);
    const auto uv_size = static_cast<std::int64_t>((width / 2) * (height / 2));
    tensor.planes = {
        pcie::Plane{.role = pcie::PlaneRole::Y,
                    .shape = {height, width},
                    .strides_bytes = {width, 1},
                    .byte_offset = 0},
        pcie::Plane{.role = pcie::PlaneRole::U,
                    .shape = {height / 2, width / 2},
                    .strides_bytes = {width / 2, 1},
                    .byte_offset = y_size},
        pcie::Plane{.role = pcie::PlaneRole::V,
                    .shape = {height / 2, width / 2},
                    .strides_bytes = {width / 2, 1},
                    .byte_offset = y_size + uv_size},
    };
  }
  return tensor;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    auto bytes = read_file(args.raw_image);

    pcie::ConnectionOptions conn;
    conn.card_host = args.card_host;
    conn.queue = args.queue;
    pcie::ModelOptions model_options;
    model_options.preprocess.kind = pcie::InputKind::Image;

    pcie::SimaPCIeHost host(conn);
    host.init_pipeline(args.model, model_options);

    pcie::Tensor image = make_image_tensor(bytes, args.format, args.width, args.height);
    const auto result = host.run(image, 30000);
    std::cout << "received tensors=" << result.size() << "\n";
    host.stop();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
