#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/DmabufEligibility.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int usage(const char* argv0, const char* detail = nullptr) {
  if (detail) {
    std::cerr << "error: " << detail << "\n";
  }
  std::cerr << "usage: " << (argv0 ? argv0 : "neat-dmabuf-plan-audit")
            << " --mpk <mpk.json> (--elf <model.elf> |"
               " --mla-artifact <logical-stage-id> <manifest-executable> <resolved-file>"
               " [--mla-artifact ...]) [--pretty]\n";
  return 64;
}

} // namespace

int main(int argc, char** argv) {
  std::filesystem::path mpk;
  std::filesystem::path elf;
  std::vector<simaai::neat::pipeline_internal::MlaExecutableArtifact> artifacts;
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
    if (arg == "--mla-artifact" && index + 3 < argc) {
      artifacts.push_back({argv[index + 1] ? argv[index + 1] : "",
                           argv[index + 2] ? argv[index + 2] : "",
                           argv[index + 3] ? argv[index + 3] : ""});
      index += 3;
      continue;
    }
    return usage(argv[0], "unknown or incomplete argument");
  }
  if (mpk.empty() || (elf.empty() == artifacts.empty())) {
    return usage(argv[0],
                 "--mpk and exactly one of --elf or one-or-more --mla-artifact entries are "
                 "required");
  }

  const auto result = elf.empty()
                          ? simaai::neat::pipeline_internal::try_compile_dmabuf_plan(mpk, artifacts)
                          : simaai::neat::pipeline_internal::try_compile_dmabuf_plan(mpk, elf);
  std::cout << (elf.empty() ? simaai::neat::pipeline_internal::dmabuf_plan_audit_json(
                                  result, mpk, artifacts, pretty)
                            : simaai::neat::pipeline_internal::dmabuf_plan_audit_json(result, mpk,
                                                                                      elf, pretty))
            << "\n";
  return result.eligible() ? 0 : 2;
}
