#include "genai/DlSvcClient.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  // A host without the PCIe host package must get a clear message, not a crash.
  try {
    simaai::neat::pcie::genai::internal::DlSvcClient client(0, "libsimaaipcie-does-not-exist.so");
    std::cerr << "[FAIL] a missing svc library must throw\n";
    return 1;
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find("PCIe support unavailable") == std::string::npos) {
      std::cerr << "[FAIL] unexpected message: " << e.what() << "\n";
      return 1;
    }
  }
  std::cout << "[PASS] DlSvcClient reports a missing svc library\n";
  return 0;
}
