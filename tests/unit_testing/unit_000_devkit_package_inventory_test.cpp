#include "test_utils.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct PackageExpectation {
  const char* name;
  const char* group;
};

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

std::string run_capture(const std::string& command) {
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
  (void)status;
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
                     " 2>/dev/null");
}

bool is_installed(const std::string& package) {
  const std::string status = installed_status(package);
  return status.rfind("ii ", 0) == 0;
}

std::string package_version(const std::string& package) {
  const std::string status = installed_status(package);
  const auto tab = status.find('\t');
  if (tab == std::string::npos) {
    return "<not-installed>";
  }
  return status.substr(tab + 1);
}

void require_installed_packages(const std::vector<PackageExpectation>& packages) {
  std::vector<PackageExpectation> missing;
  for (const auto& package : packages) {
    if (!is_installed(package.name)) {
      missing.push_back(package);
    }
  }

  if (missing.empty()) {
    return;
  }

  std::ostringstream message;
  message << "expected package(s) to remain installed after NEAT install:";
  for (const auto& package : missing) {
    message << "\n  [" << package.group << "] " << package.name
            << " status=" << installed_status(package.name);
  }
  throw std::runtime_error(message.str());
}

void require_package_version(const std::string& package, const std::string& expected) {
  const std::string version = package_version(package);
  require(version == expected,
          package + " should have package version " + expected + ", got " + version);
}

void require_versioned_provide(const std::string& package, const std::string& provided_name,
                               const std::string& provided_version) {
  const std::string provides =
      run_capture("dpkg-query -W -f='${Provides}' " + shell_quote(package) + " 2>/dev/null");
  const std::string expected = provided_name + " (= " + provided_version + ")";
  require(provides == expected, package + " should provide " + expected + ", got: " + provides);
}

} // namespace

int main() {
  try {
    if (!is_modalix_devkit() || !command_succeeds("command -v dpkg-query >/dev/null 2>&1")) {
      std::cout
          << "[SKIP] unit_000_devkit_package_inventory_test requires a Modalix DevKit with dpkg\n";
      return 77;
    }

    const std::vector<PackageExpectation> neat_packages = {
        {"sima-neat", "neat"},          {"sima-neat-dev", "neat"},
        {"neat-common", "neat"},        {"neat-appcomplex", "neat"},
        {"neat-runtime", "neat"},       {"neat-gst-plugins", "neat"},
        {"neat-ev74-firmware", "neat"}, {"neat-internals-dev", "neat"},
        {"sima-lmm-core", "neat"},      {"sima-lmm-dev", "neat"},
        {"sima-lmm-cli", "neat"},       {"libcamera", "neat"},
        {"libcamera-dev", "neat"},      {"libcamera-tools", "neat"},
        {"simaai-memory-lib", "neat"},  {"simaai-memory-lib-dev", "neat"},
    };

    const std::vector<PackageExpectation> native_sima_packages = {
        {"simaai-palette-modalix", "native-sima"},
        {"simaai-palette-upgrade", "native-sima"},
        {"simaai-a65-plat-tests", "native-sima"},
        {"simaai-base-files-modalix", "native-sima"},
        {"simaai-gst-plugins", "native-sima"},
        {"simaai-hpi-modalix", "native-sima"},
        {"simaai-log", "native-sima"},
        {"simaai-logd", "native-sima"},
        {"simaai-mlart-modalix", "native-sima"},
        {"simaai-parser", "native-sima"},
        {"simaai-pcie-ep", "native-sima"},
        {"simaai-rctd", "native-sima"},
        {"simaai-socpipeline", "native-sima"},
        {"simaai-trace", "native-sima"},
        {"simaai-utils", "native-sima"},
    };

    require_installed_packages(neat_packages);
    require_installed_packages(native_sima_packages);

    require_package_version("libcamera", "2.1.1+neat1");
    require_package_version("libcamera-tools", "2.1.1+neat1");
    require_package_version("libcamera-dev", "2.1.1+neat1");
    require_package_version("simaai-memory-lib", "2.1.1-0neat1");
    require_package_version("simaai-memory-lib-dev", "2.1.1-0neat1");
    require_versioned_provide("libcamera", "libcamera", "2.1.3~pre4040");
    require_versioned_provide("libcamera-tools", "libcamera-tools", "2.1.3~pre4040");
    require_versioned_provide("libcamera-dev", "libcamera-dev", "2.1.3~pre4040");
    require_versioned_provide("simaai-memory-lib", "simaai-memory-lib", "2.1.1~pre4040");
    require_versioned_provide("simaai-memory-lib-dev", "simaai-memory-lib-dev", "2.1.1~pre4040");

    require(command_succeeds("command -v simaai-ota >/dev/null 2>&1"),
            "simaai-ota command should remain available through simaai-palette-modalix");
    const std::string ota_owner = run_capture("dpkg-query -S /usr/bin/simaai-ota 2>/dev/null");
    require(ota_owner.find("simaai-palette-modalix:") != std::string::npos,
            "simaai-ota should be owned by simaai-palette-modalix, got: " + ota_owner);

    std::cout << "[OK] unit_000_devkit_package_inventory_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
