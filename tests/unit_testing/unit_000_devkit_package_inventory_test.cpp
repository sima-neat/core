#include "test_utils.h"
#include "support/package_loaded_identity.h"
#include "support/package_platform_baseline.h"
#include <neat/version.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (const char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

std::string run_capture(const std::string& command, bool allow_failure = false) {
  std::array<char, 256> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("failed to run command: " + command);
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  const int status = pclose(pipe);
  if (status != 0 && !allow_failure) {
    throw std::runtime_error("command failed: " + command + " output=" + output);
  }
  return output;
}

bool command_succeeds(const std::string& command) {
  const int status = std::system(command.c_str());
  return status == 0;
}

bool is_modalix_devkit() {
  std::ifstream input("/etc/buildinfo");
  if (!input) {
    return false;
  }

  std::string line;
  while (std::getline(input, line)) {
    if (line.find("MACHINE") != std::string::npos && line.find("modalix") != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::string installed_status(const std::string& package) {
  return run_capture("dpkg-query -W -f='${db:Status-Abbrev}\\t${Version}' " + shell_quote(package) +
                         " 2>/dev/null",
                     true);
}

bool is_installed(const std::string& package) {
  const std::string status = installed_status(package);
  return status.rfind("ii ", 0) == 0;
}

nlohmann::json platform_baseline() {
  const char* filename = std::getenv("SIMA_TEST_PLATFORM_BASELINE");
  require(filename && *filename,
          "installed-package qualification requires SIMA_TEST_PLATFORM_BASELINE captured before "
          "installation by verify_core_dist.py --capture-platform-baseline");
  std::ifstream input(filename);
  require(input.good(), "cannot open pre-install platform baseline");
  auto value = nlohmann::json::parse(input);
  simaai::neat::test_support::validate_platform_baseline(value);
  return value;
}

nlohmann::json current_platform_inventory() {
  nlohmann::json current = {{"schema_version", 2},
                            {"scope", "native-platform-before-install"},
                            {"packages", nlohmann::json::object()},
                            {"absent_packages", nlohmann::json::array()}};
  // Query the full database so a genuinely absent role is distinct from a
  // failed query or a present but incompletely configured package.
  std::istringstream records(
      run_capture("dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\t${Status}\n'"));
  std::string record;
  for (; std::getline(records, record);) {
    std::istringstream fields(record);
    std::string name, version, architecture, status;
    require(static_cast<bool>(std::getline(fields, name, '\t')) &&
                static_cast<bool>(std::getline(fields, version, '\t')) &&
                static_cast<bool>(std::getline(fields, architecture, '\t')) &&
                static_cast<bool>(std::getline(fields, status)),
            "malformed installed package record");
    if (!simaai::neat::test_support::native_platform_roles().contains(name))
      continue;
    require(!current["packages"].contains(name), "ambiguous installed platform role: " + name);
    current["packages"][name] = {{"Package", name},
                                 {"Version", version},
                                 {"Architecture", architecture},
                                 {"Status", status}};
  }
  for (const auto& name : simaai::neat::test_support::native_platform_roles()) {
    if (!current["packages"].contains(name))
      current["absent_packages"].push_back(name);
  }
  current["ota_present"] = false;
  current["ota_path"] = nullptr;
  current["ota_owner"] = nullptr;
  current["ota_sha256"] = nullptr;
  std::string ota_command = run_capture("command -v simaai-ota 2>/dev/null", true);
  if (!ota_command.empty() && ota_command.back() == '\n')
    ota_command.pop_back();
  const std::string filename = simaai::neat::test_support::resolve_platform_ota(
      run_capture("dpkg-query -L simaai-palette-modalix"), ota_command);
  if (filename.empty()) {
    simaai::neat::test_support::validate_platform_baseline(current);
    return current;
  }
  const auto owner = run_capture("dpkg-query -S -- " + shell_quote(filename) + " 2>/dev/null");
  require(owner == "simaai-palette-modalix: " + filename + "\n" ||
              owner == "simaai-palette-modalix:arm64: " + filename + "\n",
          "simaai-ota has unexpected package ownership: " + owner);
  current["ota_present"] = true;
  current["ota_path"] = filename;
  current["ota_owner"] = "simaai-palette-modalix";
  const auto identity = run_capture("sha256sum -- " + shell_quote(filename));
  require(identity.size() > 64 && identity.at(64) == ' ', "invalid OTA checksum result");
  current["ota_sha256"] = identity.substr(0, 64);
  simaai::neat::test_support::validate_platform_baseline(current);
  return current;
}

// The expected inventory comes from verified selected DEBs, not from this
// machine's installed state. Prefix-only tests must not claim a global install.
nlohmann::json selected_inventory() {
  const char* filename = std::getenv("SIMA_TEST_PACKAGE_INVENTORY");
  require(filename && *filename,
          "installed-package qualification requires SIMA_TEST_PACKAGE_INVENTORY from "
          "verify_core_dist.py --write-package-inventory; private-prefix integration is a separate "
          "scope");
  std::ifstream input(filename);
  require(input.good(), "cannot open selected package inventory");
  auto value = nlohmann::json::parse(input);
  require(value.value("schema_version", 0) == 1 &&
              value.value("scope", "") == "expected-installed-packages",
          "unsupported selected package inventory");
  const std::string profile = value.at("profile");
  require(profile == "core" || profile == "core-codecs" || profile == "core-codecs-llima",
          "unknown installed-package profile: " + profile);
  std::vector<std::string> required = {"sima-neat",     "sima-neat-dev",    "simaai-kernels-dev",
                                       "simaai-runner", "simaai-scheduler", "simaai-runtime"};
  if (profile != "core") {
    required.insert(required.end(), {"neat-codecs", "neat-codecs-dev"});
  }
  if (profile == "core-codecs-llima") {
    required.insert(required.end(), {"sima-lmm-core", "sima-lmm-cli"});
  }
  for (const auto& name : required) {
    require(value.at("packages").contains(name), "missing required profile role: " + name);
  }
  return value;
}

void require_selected_packages(const nlohmann::json& inventory) {
  for (const auto& entry : inventory.at("packages").items()) {
    const auto& expected = entry.value();
    const std::string package = expected.at("Package");
    const std::string version = expected.at("Version");
    const std::string architecture = expected.at("Architecture");
    require(package == entry.key() && (architecture == "arm64" || architecture == "all"),
            "invalid selected package identity: " + entry.key());
    const std::string actual =
        run_capture("dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\t${Status}\n' " +
                    shell_quote(package) + " 2>/dev/null");
    const std::string wanted =
        package + "\t" + version + "\t" + architecture + "\tinstall ok installed\n";
    require(actual == wanted,
            "installed package differs: " + package + " expected=" + wanted + " actual=" + actual);
  }
}

void require_selected_loaded_libraries(const nlohmann::json& inventory) {
  // Force a real Core link even under --as-needed, without initializing a
  // backend. The existing docs consumer separately proves installed headers.
  require(sima_neat_version() != nullptr, "Core version symbol unavailable");
  std::map<std::string, std::set<std::string>> expected_by_soname;
  for (const auto& item : inventory.at("payload").items()) {
    if (item.value().contains("soname")) {
      expected_by_soname[item.value().at("soname").get<std::string>()].insert(item.key());
    }
  }
  require(!expected_by_soname.empty(), "selected inventory lacks ELF SONAME identities");
  bool found_core = false;
  for (const auto& library :
       simaai::neat::test_support::selected_loaded_libraries(expected_by_soname)) {
    const std::string& filename = library.path;
    const std::string package = inventory.at("payload").at(filename).at("package");
    const std::string owner =
        run_capture("dpkg-query -S " + shell_quote(filename) + " 2>/dev/null");
    require(owner == package + ": " + filename + "\n" ||
                owner == package + ":arm64: " + filename + "\n",
            "loaded library has unexpected package ownership: " + filename + " owner=" + owner);
    if (package == "sima-neat" && library.soname.rfind("libsima_neat.so", 0) == 0)
      found_core = true;
  }
  require(found_core, "actual Core DSO is not from the selected installed package");
}

} // namespace

int main() {
  try {
    if (!is_modalix_devkit() || !command_succeeds("command -v dpkg-query >/dev/null 2>&1")) {
      std::cout
          << "[SKIP] unit_000_devkit_package_inventory_test requires a Modalix DevKit with dpkg\n";
      return 77;
    }

    const auto inventory = selected_inventory();

    const auto before = platform_baseline();

    require_selected_packages(inventory);
    require_selected_loaded_libraries(inventory);
    simaai::neat::test_support::require_platform_preserved(before, inventory.at("packages"),
                                                           current_platform_inventory());

    // Runtime replaces the old implementation packages, not the native platform.
    for (const char* retired : {"neat-runtime", "neat-gst-plugins", "neat-internals-dev"}) {
      require(!is_installed(retired), std::string("retired package remains installed: ") + retired);
    }

    std::cout << "[OK] unit_000_devkit_package_inventory_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
