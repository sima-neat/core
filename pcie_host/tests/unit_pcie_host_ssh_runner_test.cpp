#include "SshRunner.h"

#include <iostream>
#include <stdexcept>

namespace pcie_internal = simaai::neat::pcie::internal;

int main() {
  try {
    const std::string escaped = pcie_internal::SshRunner::shell_escape("ab'cd");
    if (escaped != "'ab'\\''cd'") {
      throw std::runtime_error("shell escape mismatch");
    }
    const auto ok = pcie_internal::SshRunner::run({"/bin/sh", "-c", "echo hello"}, 2);
    if (ok.exit_code != 0 || ok.output.find("hello") == std::string::npos) {
      throw std::runtime_error("expected echo command to succeed");
    }
    const auto bad = pcie_internal::SshRunner::run({"/bin/sh", "-c", "exit 7"}, 2);
    if (bad.exit_code != 7) {
      throw std::runtime_error("expected exit code 7");
    }
    std::cout << "[PASS] ssh runner\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
