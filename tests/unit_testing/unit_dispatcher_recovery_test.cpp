#include "pipeline/internal/DispatcherRecovery.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main() {
  try {
    const std::string err = "GST ERROR: Unable to connect to the server from dispatcher";
    simaai::neat::SessionReport rep;
    if (simaai::neat::pipeline_internal::match_dispatcher_unavailable(err)) {
      rep.error_code = simaai::neat::pipeline_internal::kDispatcherUnavailableError;
    }

    bool called = false;
    auto cb = [&](const simaai::neat::SessionReport&) { called = true; };
    if (simaai::neat::pipeline_internal::is_dispatcher_unavailable(rep)) {
      cb(rep);
    }

    if (!called) {
      throw std::runtime_error("dispatcher callback was not invoked");
    }

    std::cout << "[OK] unit_dispatcher_recovery_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
