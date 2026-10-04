#include "simaai/neat/pcie/Runtime.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace pcie = simaai::neat::pcie;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
  try {
    callable();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

} // namespace

int main() {
  try {
    pcie::ConnectionOptions connection;
    connection.queue = 2;
    pcie::Runtime runtime(connection);

    require(!runtime.retrieve(0).has_value(), "empty runtime must have no completion");
    require_throws<std::invalid_argument>([&] { (void)runtime.load_models({}); },
                                          "empty model batch must be rejected");
    require_throws<std::invalid_argument>(
        [&] { (void)runtime.try_enqueue(99, 42, pcie::Tensor{}); },
        "unknown model ID must be rejected");
    require_throws<std::invalid_argument>([&] { runtime.unload(99, 0); },
                                          "unknown model unload must be rejected");

    runtime.close();
    runtime.close();
    require(!runtime.retrieve(0).has_value(), "closed runtime must have no completion");
    require_throws<std::runtime_error>([&] { (void)runtime.load("missing-model.tar.gz"); },
                                       "closed runtime must reject model loading");

    for (const int queue : {4, 5}) {
      pcie::ConnectionOptions valid;
      valid.queue = queue;
      pcie::Runtime valid_runtime(valid);
      require(!valid_runtime.retrieve(0).has_value(), "new queues must accept an empty runtime");
    }
    for (const int queue : {-1, 6}) {
      pcie::ConnectionOptions invalid;
      invalid.queue = queue;
      require_throws<std::invalid_argument>([&] { pcie::Runtime invalid_runtime(invalid); },
                                            "runtime must reject an invalid first queue");
    }

    std::cout << "[PASS] PCIe runtime API guards\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
