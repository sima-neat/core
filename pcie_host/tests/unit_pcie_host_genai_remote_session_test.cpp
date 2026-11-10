#include "SshRunner.h"
#include "genai/RemoteSession.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>

namespace g = simaai::neat::pcie::genai;
using Runner = simaai::neat::pcie::internal::SshRunner;

namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}

struct Directory {
  std::filesystem::path path;
  Directory() {
    char name[] = "/tmp/neat-genai-remote-session-XXXXXX";
    const auto created = mkdtemp(name);
    if (!created)
      throw std::runtime_error("mkdtemp failed");
    path = created;
  }
  ~Directory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

struct SearchPath {
  std::optional<std::string> previous;
  explicit SearchPath(const std::filesystem::path& directory) {
    if (const auto* value = std::getenv("PATH"))
      previous = value;
    const auto path = directory.string() + ":" + previous.value_or("/usr/bin:/bin");
    if (setenv("PATH", path.c_str(), 1) != 0)
      throw std::runtime_error("Cannot set test executable search path");
  }
  ~SearchPath() {
    if (previous)
      setenv("PATH", previous->c_str(), 1);
    else
      unsetenv("PATH");
  }
};

struct Environment {
  std::string name;
  std::optional<std::string> previous;
  Environment(std::string variable, const std::string& value) : name(std::move(variable)) {
    if (const auto* current = std::getenv(name.c_str()))
      previous = current;
    if (setenv(name.c_str(), value.c_str(), 1) != 0)
      throw std::runtime_error("Cannot set test environment variable");
  }
  ~Environment() {
    if (previous)
      setenv(name.c_str(), previous->c_str(), 1);
    else
      unsetenv(name.c_str());
  }
};

template <class Function> void require_rejection(Function function) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("Expected invalid GenAI card environment rejection");
}
} // namespace

int main() {
  try {
    Directory directory;
    const auto arguments = directory.path / "ssh-arguments";
    const auto fake_ssh = directory.path / "ssh";
    std::ofstream(fake_ssh) << "#!/bin/sh\nprintf '%s\\n' \"$@\" > "
                            << Runner::shell_escape(arguments.string()) << '\n';
    std::filesystem::permissions(fake_ssh, std::filesystem::perms::owner_all);
    SearchPath search(directory.path);
    Environment card_environment(
        "SIMA_NEAT_PCIE_GENAI_CARD_ENV",
        "SIMA_NEAT_PCIE_GENAI_HELPER_PATH=/tmp/private/neat-pcie-genai-worker "
        "LD_LIBRARY_PATH=/tmp/private/lib");

    g::ConnectionOptions connection;
    connection.card_host = "192.168.1.42";
    g::internal::RemoteSession remote(connection);
    remote.start("target model", "draft model");

    std::ifstream input(arguments);
    const std::string contents((std::istreambuf_iterator<char>(input)), {});
    require(contents.find("/tmp/private/neat-pcie-genai-worker") != std::string::npos,
            "GenAI worker launch did not use the selected helper");
    require(contents.find("LD_LIBRARY_PATH=/tmp/private/lib") != std::string::npos,
            "GenAI worker launch did not carry the selected environment");
    require(contents.find("nohup /usr/bin/neat-pcie-genai-worker") == std::string::npos,
            "GenAI worker launch fell back to the ambient helper");
    require(contents.find("target model") != std::string::npos &&
                contents.find("draft model") != std::string::npos,
            "GenAI worker launch lost the selected model directories");

    g::ConnectionOptions invalid;
    {
      Environment invalid_environment("SIMA_NEAT_PCIE_GENAI_CARD_ENV",
                                      "SIMA_NEAT_PCIE_GENAI_HELPER_PATH=relative/path");
      require_rejection([&] { g::internal::RemoteSession rejected(invalid); });
    }
    {
      Environment invalid_environment("SIMA_NEAT_PCIE_GENAI_CARD_ENV", "INVALID_ENV_ENTRY");
      require_rejection([&] { g::internal::RemoteSession rejected(invalid); });
    }
    std::cout << "GenAI private worker selection: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
