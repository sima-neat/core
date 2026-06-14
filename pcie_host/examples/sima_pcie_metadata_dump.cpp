#include "simaai/neat/pcie/SimaPCIeHost.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace pcie = simaai::neat::pcie;

namespace {

struct Args {
  std::string model;
  bool accelerator_only = false;
};

std::string value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

void usage() {
  std::cerr << "usage: sima_pcie_metadata_dump --model model.tar.gz [--accelerator-only]\n";
}

Args parse(int argc, char** argv) {
  Args out;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg == "--model") {
      out.model = value(argc, argv, i, "--model");
    } else if (arg == "--accelerator-only") {
      out.accelerator_only = true;
    } else if (arg == "-h" || arg == "--help") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (out.model.empty() || !std::filesystem::exists(out.model)) {
    throw std::runtime_error("--model must point to a model archive");
  }
  return out;
}

void print_tensor(const pcie::TensorInfo& tensor, const std::size_t index) {
  std::cout << "  [" << index << "] name=" << (tensor.name.empty() ? "<unnamed>" : tensor.name)
            << " dtype=" << (tensor.dtype.empty() ? "<unknown>" : tensor.dtype) << " shape=[";
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << tensor.shape[i];
  }
  std::cout << "] size_bytes=" << tensor.size_bytes << "\n";
}

void print_model_info(const char* title, const pcie::ModelInfo& info) {
  std::cout << title << "\n";
  std::cout << "has_preprocess=" << (info.has_preprocess ? "true" : "false")
            << " has_boxdecode=" << (info.has_boxdecode ? "true" : "false") << "\n";
  std::cout << "inputs:\n";
  for (std::size_t i = 0; i < info.inputs.size(); ++i) {
    print_tensor(info.inputs[i], i);
  }
  std::cout << "outputs:\n";
  for (std::size_t i = 0; i < info.outputs.size(); ++i) {
    print_tensor(info.outputs[i], i);
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    pcie::SimaPCIeHost host;
    if (!args.accelerator_only) {
      print_model_info("tensor route", host.load_metadata(args.model));
      std::cout << "\n";
    }
    constexpr bool accelerator = true;
    print_model_info("accelerator route", host.load_metadata(args.model, {}, accelerator));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    usage();
    return 1;
  }
}
