#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

namespace simaai::neat::pipeline_internal {

// Pure classification seam for tests; production paths are fixed below.
inline bool legacy_runtime_recovery_allowed_at(const std::filesystem::path& root,
                                               bool direct_build) {
  if (direct_build)
    return false;
  std::ifstream input(root / "etc/buildinfo");
  if (!input)
    return false;
  std::string machine, version, line;
  bool have_machine = false, have_version = false;
  const auto trim = [](const std::string& text) {
    const auto start = text.find_first_not_of(" \t\r");
    if (start == std::string::npos)
      return std::string{};
    return text.substr(start, text.find_last_not_of(" \t\r") - start + 1);
  };
  while (std::getline(input, line)) {
    if (line.size() > 4096)
      return false;
    const auto equals = line.find('=');
    if (equals == std::string::npos)
      continue;
    const auto key = trim(line.substr(0, equals));
    if (key == "MACHINE") {
      if (have_machine)
        return false;
      have_machine = true;
      machine = trim(line.substr(equals + 1));
    } else if (key == "DISTRO_VERSION") {
      if (have_version)
        return false;
      have_version = true;
      version = trim(line.substr(equals + 1));
    }
  }
  return !input.bad() && machine == "modalix" &&
         std::regex_match(version, std::regex("2[.]1[.][0-9]+([.~+_-][A-Za-z0-9_.+~-]+)?"));
}

} // namespace simaai::neat::pipeline_internal
