#include "simaai/neat/pcie/Model.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace pcie = simaai::neat::pcie;

int main() {
  try {
    const char* model_env = std::getenv("SIMAPCIE_YOLOV8_MODEL");
    if (!model_env || !std::filesystem::is_regular_file(model_env)) {
      std::cout << "[SKIP] SIMAPCIE_YOLOV8_MODEL is not set to a readable model\n";
      return 0;
    }

    pcie::Model model(model_env);
    if (model.running()) {
      throw std::runtime_error("pcie::Model should not be running after construction");
    }
    if (model.info().inputs.empty()) {
      throw std::runtime_error("pcie::Model should load metadata during construction");
    }

    const std::string expected =
        "PCIe model is not built; call model.build() before run/push/pull";
    bool threw = false;
    try {
      (void)model.push(pcie::Tensor{});
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()) == expected;
    }
    if (!threw) {
      throw std::runtime_error("push before build must throw the expected lifecycle error");
    }
    threw = false;
    try {
      (void)model.pull(0);
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()) == expected;
    }
    if (!threw) {
      throw std::runtime_error("pull before build must throw the expected lifecycle error");
    }
    model.close();
    model.close();
    std::cout << "[PASS] lifecycle guards\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
