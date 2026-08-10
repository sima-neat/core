#include "SshRunner.h"
#include "RemoteRuntime.h"

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
    const std::string first =
        pcie_internal::RemoteRuntime::unique_remote_upload_path("/first/model.tar.gz");
    const std::string second =
        pcie_internal::RemoteRuntime::unique_remote_upload_path("/second/model.tar.gz");
    if (first == second) {
      throw std::runtime_error("remote upload paths must be unique");
    }
    if (first.find("/tmp/sima-neat-pcie-") != 0 ||
        first.rfind("-model.tar.gz") != first.size() - std::string("-model.tar.gz").size()) {
      throw std::runtime_error("unexpected remote upload path: " + first);
    }
    std::cout << "[PASS] ssh runner\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
