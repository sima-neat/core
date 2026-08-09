#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/DmabufEligibility.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

int usage(const char* argv0, const char* detail = nullptr) {
  if (detail) {
    std::cerr << "error: " << detail << "\n";
  }
  std::cerr << "usage: " << (argv0 ? argv0 : "neat-dmabuf-plan-audit")
            << " --mpk <mpk.json> --elf <model.elf> [--pretty]\n";
  return 64;
}

} // namespace

int main(int argc, char** argv) {
  std::filesystem::path mpk;
  std::filesystem::path elf;
  bool pretty = false;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index] ? argv[index] : "";
    if (arg == "--pretty") {
      pretty = true;
      continue;
    }
    if ((arg == "--mpk" || arg == "--elf") && index + 1 < argc) {
      auto& destination = arg == "--mpk" ? mpk : elf;
      if (!destination.empty()) {
        return usage(argv[0], arg == "--mpk" ? "duplicate --mpk" : "duplicate --elf");
      }
      destination = argv[++index];
      continue;
    }
    return usage(argv[0], "unknown or incomplete argument");
  }
  if (mpk.empty() || elf.empty()) {
    return usage(argv[0], "both --mpk and --elf are required");
  }

  const auto result = simaai::neat::pipeline_internal::try_compile_dmabuf_plan(mpk, elf);
  std::cout << simaai::neat::pipeline_internal::dmabuf_plan_audit_json(result, mpk, elf, pretty)
            << "\n";
  return result.eligible() ? 0 : 2;
}
