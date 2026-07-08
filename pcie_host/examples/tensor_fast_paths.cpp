#include <simaai/neat/pcie/Model.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

struct Args {
  std::string model;
  std::string card_host;
  std::string user = "sima";
  int card_id = 0;
  int queue = 0;
  int readiness_timeout_ms = 180000;
  int pull_timeout_ms = 30000;
  bool shared = false;
};

std::string require_value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " [--model model.tar.gz] [--card-host host] [--user user]"
               " [--card-id n] [--queue n] [--shared]"
               " [--readiness-timeout-ms ms] [--pull-timeout-ms ms]\n\n"
               "Without --model and --card-host, this prints the constructed tensor layout only.\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg == "--model") {
      args.model = require_value(argc, argv, i, "--model");
    } else if (arg == "--card-host") {
      args.card_host = require_value(argc, argv, i, "--card-host");
    } else if (arg == "--user") {
      args.user = require_value(argc, argv, i, "--user");
    } else if (arg == "--card-id") {
      args.card_id = std::stoi(require_value(argc, argv, i, "--card-id"));
    } else if (arg == "--queue") {
      args.queue = std::stoi(require_value(argc, argv, i, "--queue"));
    } else if (arg == "--readiness-timeout-ms") {
      args.readiness_timeout_ms = std::stoi(require_value(argc, argv, i, "--readiness-timeout-ms"));
    } else if (arg == "--pull-timeout-ms") {
      args.pull_timeout_ms = std::stoi(require_value(argc, argv, i, "--pull-timeout-ms"));
    } else if (arg == "--shared") {
      args.shared = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (!args.model.empty() && !std::filesystem::is_regular_file(args.model)) {
    throw std::runtime_error("model path does not exist: " + args.model);
  }
  return args;
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

std::size_t logical_float_payload_bytes(const pcie::Tensor& tensor) {
  std::size_t count = 1;
  for (const auto dim : tensor.shape) {
    count *= static_cast<std::size_t>(dim);
  }
  return count * sizeof(float);
}

pcie::Tensor make_single_contiguous_tensor() {
  constexpr std::int64_t height = 640;
  constexpr std::int64_t width = 640;
  constexpr std::int64_t channels = 3;
  std::vector<float> data(height * width * channels, 0.25F);
  pcie::Tensor tensor =
      pcie::Tensor::from_vector(std::move(data), {height, width, channels}, "images");
  tensor.route.logical_index = 0;
  tensor.route.physical_index = 0;
  tensor.route.route_slot = 0;
  return tensor;
}

pcie::TensorList make_shared_packed_tensors() {
  constexpr std::size_t first_count = 8 * 8;
  constexpr std::size_t second_count = 4 * 8;
  auto storage = std::make_shared<std::vector<float>>(first_count + second_count, 1.0F);

  pcie::Tensor first =
      pcie::Tensor::from_external(storage->data(), storage->size(), storage, {8, 8}, "input_0");
  first.route.logical_index = 0;
  first.route.physical_index = 0;
  first.route.route_slot = 0;

  pcie::Tensor second =
      pcie::Tensor::from_external(storage->data(), storage->size(), storage, {4, 8}, "input_1",
                                  static_cast<std::int64_t>(first_count * sizeof(float)));
  second.route.logical_index = 1;
  second.route.physical_index = 1;
  second.route.route_slot = 1;
  return {first, second};
}

void print_tensors(const pcie::TensorList& tensors, const std::string& expected_path) {
  std::cout << "expected pciehost path: " << expected_path << "\n";
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    std::cout << "  [" << i << "] name=" << tensor.route.name
              << " shape=" << shape_string(tensor.shape) << " byte_offset=" << tensor.byte_offset
              << " logical_payload_bytes=" << logical_float_payload_bytes(tensor)
              << " backing_size_bytes=" << tensor.size_bytes
              << " owner=" << (tensor.owner ? "yes" : "no") << "\n";
  }
}

void maybe_run(const Args& args, const pcie::TensorList& tensors) {
  if (args.model.empty() || args.card_host.empty()) {
    std::cout << "not running: pass --model and --card-host to push through PCIe\n";
    return;
  }

  pcie::ConnectionOptions connection;
  connection.card_host = args.card_host;
  connection.user = args.user;
  connection.card_id = args.card_id;
  connection.queue = args.queue;

  pcie::Model model(args.model, {}, connection);
  model.build(args.readiness_timeout_ms);
  const pcie::TensorList outputs = model.run(tensors, args.pull_timeout_ms);
  model.close();
  std::cout << "received outputs: " << outputs.size() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    pcie::TensorList tensors = args.shared ? make_shared_packed_tensors()
                                           : pcie::TensorList{make_single_contiguous_tensor()};
    print_tensors(tensors, args.shared ? "shared packed tensor-set fast path"
                                       : "single contiguous tensor fast path");
    maybe_run(args, tensors);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
