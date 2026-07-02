#include <simaai/neat/pcie/SimaPCIeHost.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace pcie = simaai::neat::pcie;

namespace {

std::string env_or_default(const char* name, const char* fallback) {
  if (const char* value = std::getenv(name)) {
    if (*value != '\0') {
      return value;
    }
  }
  return fallback ? fallback : "";
}

struct Args {
  std::string model = env_or_default("SIMAPCIE_YOLOV8_MODEL", DEFAULT_MODEL_PATH);
};

std::string require_value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " [--model model.tar.gz]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg == "--model") {
      args.model = require_value(argc, argv, i, "--model");
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (args.model.empty()) {
    throw std::runtime_error("model path is empty");
  }
  if (!std::filesystem::is_regular_file(args.model)) {
    throw std::runtime_error("model path does not exist or is not a regular file: " + args.model);
  }
  return args;
}

void print_shape(const std::vector<std::int64_t>& shape) {
  std::cout << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << shape[i];
  }
  std::cout << "]";
}

void print_tensor_info(const pcie::TensorInfo& tensor, std::size_t index) {
  std::cout << "  [" << index << "]"
            << " name=" << (tensor.name.empty() ? "<unnamed>" : tensor.name)
            << " dtype=" << (tensor.dtype.empty() ? "<unknown>" : tensor.dtype) << " shape=";
  print_shape(tensor.shape);
  std::cout << " size_bytes=" << tensor.size_bytes << "\n";
}

void print_model_info(const std::string& title, const pcie::ModelInfo& info) {
  std::cout << title << "\n";
  std::cout << "  has_preprocess=" << (info.has_preprocess ? "true" : "false")
            << " has_boxdecode=" << (info.has_boxdecode ? "true" : "false") << "\n";

  std::cout << "  inputs (" << info.inputs.size() << "):\n";
  for (std::size_t i = 0; i < info.inputs.size(); ++i) {
    print_tensor_info(info.inputs[i], i);
  }

  std::cout << "  outputs (" << info.outputs.size() << "):\n";
  for (std::size_t i = 0; i < info.outputs.size(); ++i) {
    print_tensor_info(info.outputs[i], i);
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    std::cout << "model=" << args.model << "\n\n";

    pcie::SimaPCIeHost host;
    print_model_info("tensor route", host.load_metadata(args.model));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
