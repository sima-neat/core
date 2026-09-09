#include "pipeline/internal/DispatcherRecovery.h"
#include "pipeline/internal/RuntimeRecoveryPolicy.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
} // namespace

// The standalone host test links with --wrap=system. No test path can execute
// recovery commands, even if the fail-closed guard regresses.
#ifdef NEAT_TEST_WRAP_SYSTEM
static unsigned command_count = 0;
extern "C" int __wrap_system(const char*) {
  ++command_count;
  return 0;
}
#endif

int main() {
  using namespace simaai::neat::pipeline_internal;
  char pattern[] = "/tmp/neat-recovery-policy-XXXXXX";
  const char* directory = mkdtemp(pattern);
  if (!directory)
    return 1;
  const std::filesystem::path root(directory);
  try {
    std::filesystem::create_directories(root / "etc");
    const auto write_metadata = [&](const std::string& value) {
      std::ofstream(root / "etc/buildinfo") << value;
    };
    require(!legacy_runtime_recovery_allowed_at(root, false), "missing identity accepted");
    for (const char* version : {"2.1.2", "2.1.3~pre4744", "2.1.3+local"}) {
      write_metadata(std::string("MACHINE = modalix\nDISTRO_VERSION = ") + version + "\n");
      require(legacy_runtime_recovery_allowed_at(root, false), "legacy identity rejected");
      require(!legacy_runtime_recovery_allowed_at(root, true), "direct build allowed recovery");
    }
    for (const char* metadata : {"MACHINE=modalix\nDISTRO_VERSION=3.0.0\n",
                                 "MACHINE=modalix\nDISTRO_VERSION=2.1.3garbage\n",
                                 "MACHINE=davinci\nDISTRO_VERSION=2.1.3\n",
                                 "MACHINE=modalix\nDISTRO_VERSION=2.1.3\nDISTRO_VERSION=3.0.0\n",
                                 "MACHINE=modalix\nMACHINE=modalix\nDISTRO_VERSION=2.1.3\n"}) {
      write_metadata(metadata);
      require(!legacy_runtime_recovery_allowed_at(root, false), "ambiguous identity accepted");
    }
    write_metadata("MACHINE=modalix\nDISTRO_VERSION=2.1.3\n");
    for (const char* name : {"sima-neat", "sima-neat-internals"}) {
      const auto receipt = root / "usr/share" / name / "runtime-profile.json";
      std::filesystem::create_directories(receipt.parent_path());
      std::ofstream(receipt) << "malformed";
      require(!legacy_runtime_recovery_allowed_at(root, false), "profile marker ignored");
      std::filesystem::remove(receipt);
      std::filesystem::create_symlink("missing", receipt);
      require(!legacy_runtime_recovery_allowed_at(root, false), "dangling marker ignored");
      std::filesystem::remove(receipt);
    }
#ifdef NEAT_TEST_WRAP_SYSTEM
    setenv("SIMA_NEAT_RECOVERY_HARD_RESET", "1", 1);
    setenv("SIMA_NEAT_RECOVERY_ALLOW_UNSAFE_RESET", "1", 1);
    simaai::neat::GraphReport report;
    require(!attempt_dispatcher_recovery(&report, true), "direct recovery reported success");
    require(command_count == 0, "direct recovery executed a command");
    require(report.repro_note.find("legacy recovery refused") != std::string::npos,
            "refusal lacked actionable diagnostics");
#endif
    std::filesystem::remove_all(root);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    std::filesystem::remove_all(root);
    return 1;
  }
}
