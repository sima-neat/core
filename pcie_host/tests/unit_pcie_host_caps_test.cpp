#include "HostPcieChannel.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace pcie_internal = simaai::neat::pcie::internal;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  try {
    const std::string tensor_caps = pcie_internal::HostPcieChannel::tensor_set_caps();
    require(tensor_caps == "application/vnd.simaai.tensor, representation=(string)tensor-set, storage=(string)tensorbuffer",
            "unexpected tensor-set caps: " + tensor_caps);

    std::cout << "[PASS] host channel caps\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
