#include "simaai/neat/pcie/SimaPCIeHost.h"

#include <iostream>
#include <stdexcept>

namespace pcie = simaai::neat::pcie;

int main() {
  try {
    pcie::SimaPCIeHost host;
    if (host.status().state != pcie::PipelineState::Uninitialized) {
      throw std::runtime_error("SimaPCIeHost should be uninitialized after construction");
    }
    bool threw = false;
    try {
      (void)host.push(pcie::Tensor{});
    } catch (const std::runtime_error&) {
      threw = true;
    }
    if (!threw) {
      throw std::runtime_error("push before init_pipeline must throw");
    }
    threw = false;
    try {
      (void)host.pull(0);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    if (!threw) {
      throw std::runtime_error("pull before init_pipeline must throw");
    }
    host.stop();
    host.stop();
    std::cout << "[PASS] lifecycle guards\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
