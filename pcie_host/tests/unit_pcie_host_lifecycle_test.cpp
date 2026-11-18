#include "simaai/neat/pcie/Model.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace pcie = simaai::neat::pcie;

namespace {

// Stand-in ssh/scp on PATH: the card builder reports ready on the first status read.
void build_configures_channel_once_card_is_ready(const std::string& model_path) {
  namespace fs = std::filesystem;
  const fs::path bin =
      fs::temp_directory_path() / ("sima-neat-pcie-fake-card-" + std::to_string(::getpid()));
  fs::create_directories(bin);
  const auto write_tool = [&](const char* name, const std::string& body) {
    std::ofstream(bin / name) << "#!/bin/sh\n" << body << "exit 0\n";
    fs::permissions(bin / name, fs::perms::owner_all);
  };
  write_tool("scp", "");
  write_tool("ssh", "for arg; do cmd=\"$arg\"; done\n"
                    "case \"$cmd\" in\n"
                    "  *launched_pid=*) echo launched_pid=4242 ;;\n"
                    "  cat\\ *) echo '{\"state\":\"ready\",\"pid\":4242,"
                    "\"output_buffer_bytes\":4096}' ;;\n"
                    "esac\n");
  const std::string path = std::getenv("PATH") ? std::getenv("PATH") : "";
  setenv("PATH", (bin.string() + ":" + path).c_str(), 1);

  pcie::Model model(model_path);
  const auto start = std::chrono::steady_clock::now();
  model.build();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const bool running = model.running();
  model.close();
  setenv("PATH", path.c_str(), 1);
  fs::remove_all(bin);
  if (!running) {
    throw std::runtime_error("build() must leave the model running once the card is ready");
  }
  if (elapsed >= std::chrono::seconds(2)) {
    throw std::runtime_error(
        "build() must configure the channel as soon as the card reports ready; took " +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()) +
        " ms");
  }
}

} // namespace

int main() {
  try {
    for (const std::string invalid_user : {"", "-invalid", "user@host", "user name"}) {
      pcie::ConnectionOptions connection;
      connection.user = invalid_user;
      bool threw = false;
      try {
        (void)pcie::Model("unused-model.tar.gz", {}, connection);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (!threw) {
        throw std::runtime_error("invalid PCIe SSH user must be rejected");
      }
    }

    for (const std::string invalid_host : {"-invalid", "user@host", "host name"}) {
      pcie::ConnectionOptions connection;
      connection.card_host = invalid_host;
      bool threw = false;
      try {
        (void)pcie::Model("unused-model.tar.gz", {}, connection);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (!threw) {
        throw std::runtime_error("invalid PCIe card host must be rejected");
      }
    }

    {
      pcie::ConnectionOptions connection;
      connection.card_id = -1;
      bool threw = false;
      try {
        (void)pcie::Model("unused-model.tar.gz", {}, connection);
      } catch (const std::invalid_argument& error) {
        threw = std::string(error.what()) == "card_id must be non-negative";
      }
      if (!threw) {
        throw std::runtime_error("negative PCIe card ID must be rejected");
      }
    }

    for (const int queue : {-1, 6}) {
      pcie::ConnectionOptions connection;
      connection.queue = queue;
      bool threw = false;
      try {
        (void)pcie::Model("unused-model.tar.gz", {}, connection);
      } catch (const std::invalid_argument& error) {
        threw = std::string(error.what()) == "queue must be in range 0..5";
      }
      if (!threw) {
        throw std::runtime_error("invalid PCIe queue must be rejected");
      }
    }

    const char* model_env = std::getenv("SIMAPCIE_YOLOV8_MODEL");
    if (!model_env || !std::filesystem::is_regular_file(model_env)) {
      std::cout << "[SKIP] SIMAPCIE_YOLOV8_MODEL is not set to a readable model\n";
      return 0;
    }

    for (const int queue : {4, 5}) {
      pcie::ConnectionOptions connection;
      connection.queue = queue;
      pcie::Model model(model_env, {}, connection);
      if (model.info().inputs.empty()) {
        throw std::runtime_error("new PCIe queues must accept model construction");
      }
    }

    pcie::Model model(model_env);
    if (model.running()) {
      throw std::runtime_error("pcie::Model should not be running after construction");
    }
    if (model.info().inputs.empty()) {
      throw std::runtime_error("pcie::Model should load metadata during construction");
    }

    const std::string expected = "PCIe model is not built; call model.build() before run/push/pull";
    bool threw = false;
    try {
      (void)model.push(pcie::Tensor{});
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()) == expected;
    }
    if (!threw) {
      throw std::runtime_error("push before build must throw the expected lifecycle error");
    }
    threw = false;
    try {
      (void)model.pull(0);
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()) == expected;
    }
    if (!threw) {
      throw std::runtime_error("pull before build must throw the expected lifecycle error");
    }
    model.close();
    model.close();
    build_configures_channel_once_card_is_ready(model_env);
    std::cout << "[PASS] lifecycle guards\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
