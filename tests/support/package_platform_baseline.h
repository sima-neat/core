#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace simaai::neat::test_support {

// This is the platform role catalogue, not a requirement that every historical
// role exists on every image. Match verify_core_dist.py's capture contract.
inline const std::set<std::string>& native_platform_roles() {
  static const std::set<std::string> roles = {"appcomplex",
                                              "simaai-common",
                                              "simaai-palette-modalix",
                                              "libcamera",
                                              "libcamera-tools",
                                              "simaai-a65-plat-tests",
                                              "simaai-base-files-modalix",
                                              "simaai-gst-plugins",
                                              "simaai-hpi-modalix",
                                              "simaai-log",
                                              "simaai-logd",
                                              "simaai-mlart-modalix",
                                              "simaai-parser",
                                              "simaai-pcie-ep",
                                              "simaai-traced",
                                              "simaai-socpipeline",
                                              "simaai-trace"};
  return roles;
}

// Legitimate absence is distinct from a damaged Palette installation whose
// package still declares the executable. Inspect real filesystem properties
// when resolution succeeds; a pathname alone is not executable availability.
inline std::string resolve_platform_ota(const std::string& palette_files,
                                        const std::string& command_path) {
  std::istringstream files(palette_files);
  std::string filename;
  std::set<std::filesystem::path> declared;
  while (std::getline(files, filename)) {
    if (std::filesystem::path(filename).filename() == "simaai-ota")
      declared.insert(filename);
  }
  if (command_path.empty()) {
    if (!declared.empty())
      throw std::runtime_error("Palette declares simaai-ota but its executable is unavailable");
    return {};
  }
  if (command_path.find('\n') != std::string::npos)
    throw std::runtime_error("ambiguous simaai-ota command resolution");
  const auto path = std::filesystem::canonical(command_path);
  if (!std::filesystem::is_regular_file(path) || access(path.c_str(), X_OK) != 0)
    throw std::runtime_error("resolved simaai-ota is not a regular executable");
  bool matches_declared = false;
  for (const auto& filename : declared) {
    std::error_code error;
    const auto declared_path = std::filesystem::canonical(filename, error);
    if (!error && declared_path == path)
      matches_declared = true;
  }
  if (!matches_declared)
    throw std::runtime_error("resolved simaai-ota differs from Palette-declared executable");
  return path.string();
}

inline void validate_platform_baseline(const nlohmann::json& baseline) {
  const auto require = [](bool ok, const std::string& message) {
    if (!ok)
      throw std::runtime_error(message);
  };
  require(baseline.value("schema_version", 0) == 2 &&
              baseline.value("scope", "") == "native-platform-before-install",
          "invalid native platform baseline schema");
  const auto& packages = baseline.at("packages");
  const auto& absent = baseline.at("absent_packages");
  require(packages.is_object() && absent.is_array(), "invalid platform role sets");
  std::set<std::string> observed;
  for (const auto& entry : packages.items()) {
    const auto& value = entry.value();
    require(value.at("Package") == entry.key() && value.at("Version").is_string() &&
                !value.at("Version").get<std::string>().empty() &&
                (value.at("Architecture") == "arm64" || value.at("Architecture") == "all") &&
                value.at("Status") == "install ok installed",
            "invalid installed platform identity: " + entry.key());
    observed.insert(entry.key());
  }
  for (const auto& value : absent) {
    require(value.is_string(), "invalid absent platform role");
    require(observed.insert(value.get<std::string>()).second,
            "duplicate or overlapping platform role");
  }
  require(observed == native_platform_roles(), "incomplete platform baseline role coverage");
  require(packages.contains("simaai-palette-modalix"), "OTA platform package must be installed");
  require(baseline.at("ota_present").is_boolean(), "missing OTA presence authority");
  if (!baseline.at("ota_present").get<bool>()) {
    require(baseline.at("ota_path").is_null() && baseline.at("ota_owner").is_null() &&
                baseline.at("ota_sha256").is_null(),
            "absent OTA command has unexpected identity");
    return;
  }
  require(baseline.at("ota_owner") == "simaai-palette-modalix", "OTA ownership differs");
  const std::filesystem::path ota_path = baseline.at("ota_path").get<std::string>();
  require(ota_path.is_absolute() && ota_path.lexically_normal() == ota_path,
          "invalid canonical OTA executable path");
  const std::string hash = baseline.at("ota_sha256");
  require(hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == std::string::npos,
          "invalid OTA content identity");
}

inline void require_platform_preserved(const nlohmann::json& before,
                                       const nlohmann::json& selected_packages,
                                       const nlohmann::json& current) {
  validate_platform_baseline(before);
  validate_platform_baseline(current);
  if (!selected_packages.is_object())
    throw std::runtime_error("invalid selected package identities");
  for (const auto& name : native_platform_roles()) {
    // The caller separately verifies exact selected controls and loaded payload.
    if (selected_packages.contains(name))
      continue;
    const bool was_installed = before.at("packages").contains(name);
    const bool is_installed = current.at("packages").contains(name);
    if (was_installed != is_installed ||
        (was_installed && before.at("packages").at(name) != current.at("packages").at(name)))
      throw std::runtime_error("unselected platform package changed: " + name);
  }
  if (before.at("ota_present") != current.at("ota_present") ||
      before.at("ota_path") != current.at("ota_path") ||
      before.at("ota_owner") != current.at("ota_owner") ||
      before.at("ota_sha256") != current.at("ota_sha256"))
    throw std::runtime_error("OTA command changed during NEAT installation");
}

} // namespace simaai::neat::test_support
