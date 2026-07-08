#include <simaai/neat/pcie/Model.h>

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
               " [--card-id n] [--queue n]"
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

pcie::Tensor make_padded_row_tensor() {
  constexpr std::int64_t height = 640;
  constexpr std::int64_t width = 640;
  constexpr std::int64_t channels = 3;
  constexpr std::int64_t padded_width = width + 16;
  auto storage = std::make_shared<std::vector<float>>(height * padded_width * channels, 0.5F);

  pcie::Tensor tensor = pcie::Tensor::from_external(
      storage->data(), storage->size(), storage, {height, width, channels}, "images", 0,
      {padded_width * channels * static_cast<std::int64_t>(sizeof(float)),
       channels * static_cast<std::int64_t>(sizeof(float)),
       static_cast<std::int64_t>(sizeof(float))});
  tensor.route.logical_index = 0;
  tensor.route.physical_index = 0;
  tensor.route.route_slot = 0;
  return tensor;
}

void maybe_run(const Args& args, const pcie::Tensor& tensor) {
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
  const pcie::TensorList outputs = model.run(tensor, args.pull_timeout_ms);
  model.close();
  std::cout << "received outputs: " << outputs.size() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const pcie::Tensor tensor = make_padded_row_tensor();
    std::cout << "expected pciehost path: fallback staging copy\n"
              << "  name=" << tensor.route.name << " shape=[640, 640, 3]"
              << " row_stride_bytes=" << tensor.strides_bytes.front()
              << " contiguous_row_bytes=" << (640 * 3 * static_cast<int>(sizeof(float)))
              << " size_bytes=" << tensor.size_bytes << " owner=" << (tensor.owner ? "yes" : "no")
              << "\n";
    maybe_run(args, tensor);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
