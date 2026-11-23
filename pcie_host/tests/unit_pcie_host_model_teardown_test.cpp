// Links the real Model.cpp against fake channel, card-runner and archive definitions so
// teardown ordering can be checked without a card.
#include "HostPcieChannel.h"
#include "ModelArchiveSnapshot.h"
#include "ModelOptionsJsonWriter.h"
#include "PcieModelFactsReader.h"
#include "RemoteRuntime.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace simaai::scheduler::pcie {
class Channel {};
} // namespace simaai::scheduler::pcie

namespace {

struct FakeCard {
  bool channel_stop_fails = false;
  bool remote_stop_fails = false;
  int channel_stops = 0;
  int remote_stops = 0;
  int upload_removals = 0;
};

FakeCard card;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

namespace simaai::neat::pcie::internal {

ScopedTempDirectory::ScopedTempDirectory(const std::string& prefix) : path_(prefix) {}
ScopedTempDirectory::~ScopedTempDirectory() noexcept = default;
std::string host_path_string(const std::filesystem::path& path) {
  return path.string();
}

ModelArchiveSnapshot::ModelArchiveSnapshot(const std::string& source_path)
    : directory_("snapshot"), archive_path_(source_path) {}
ModelArchiveSnapshot::~ModelArchiveSnapshot() noexcept = default;

ModelOptionsJson write_model_options_json(const ModelOptions&) {
  return {};
}
PcieModelFacts read_model_facts(const std::string&, const ModelOptions&) {
  return {};
}
ModelInfo to_public_model_info(const PcieModelFacts&) {
  return {};
}

HostPcieChannel::HostPcieChannel() = default;
HostPcieChannel::~HostPcieChannel() = default;
void HostPcieChannel::configure(const PcieModelFacts&, int, int, int, bool, std::size_t) {}
void HostPcieChannel::request_stop() {}
void HostPcieChannel::stop() {
  ++card.channel_stops;
  if (card.channel_stop_fails) {
    throw std::runtime_error("channel stop deadline expired");
  }
}
bool HostPcieChannel::is_running() const {
  return false;
}
bool HostPcieChannel::push(const TensorList&) {
  return false;
}
bool HostPcieChannel::try_push(std::int32_t, const TensorList&) {
  return false;
}
std::optional<TensorList> HostPcieChannel::pull(int) {
  return std::nullopt;
}
std::optional<RuntimeInferenceResult> HostPcieChannel::pull_result(int) {
  return std::nullopt;
}

RemoteRuntime::RemoteRuntime(ConnectionOptions connection) : connection_(std::move(connection)) {}
std::string RemoteRuntime::upload_file(const std::string&) const {
  return "/tmp/upload";
}
int RemoteRuntime::start(int, const std::string&, const std::optional<std::string>&) const {
  return 42;
}
RemoteStatus RemoteRuntime::wait_ready(int, int, int) const {
  throw std::runtime_error("card builder did not become ready");
}
void RemoteRuntime::stop(int, int) const {
  ++card.remote_stops;
  if (card.remote_stop_fails) {
    throw std::runtime_error("card worker stop failed");
  }
}
void RemoteRuntime::stop_process(int, const std::string&) const {
  ++card.remote_stops;
}
void RemoteRuntime::remove_upload(const std::string&) const {
  ++card.upload_removals;
}

} // namespace simaai::neat::pcie::internal

namespace pcie = simaai::neat::pcie;

namespace {

// Leaves the model failed with its card worker still running: the build fails after the
// worker starts, and the build's own attempt to stop the worker fails too.
void build_with_running_card_worker(pcie::Model& model) {
  card.remote_stop_fails = true;
  try {
    model.build(1);
  } catch (const std::runtime_error&) {
  }
  card.remote_stop_fails = false;
  card.channel_stops = card.remote_stops = card.upload_removals = 0;
}

} // namespace

int main() {
  try {
    {
      pcie::Model model("model.tar.gz");
      card.channel_stop_fails = true;
      std::string error;
      try {
        model.build(1);
      } catch (const std::runtime_error& e) {
        error = e.what();
      }
      require(error == "card builder did not become ready",
              "build must report its own failure, not the channel stop failure: " + error);
      require(card.remote_stops == 1, "failed build must stop the card worker");
      require(card.upload_removals == 1,
              "failed build must remove uploads once the worker stopped");
      card.channel_stop_fails = false;
    }

    {
      pcie::Model model("model.tar.gz");
      build_with_running_card_worker(model);
      card.channel_stop_fails = true;
      std::string error;
      try {
        model.close();
      } catch (const std::runtime_error& e) {
        error = e.what();
      }
      require(error == "channel stop deadline expired", "close must rethrow the channel error");
      require(card.remote_stops == 1, "close must stop the card worker when channel stop fails");
      require(card.upload_removals == 1, "close must remove uploads once the worker stopped");

      card.channel_stop_fails = false;
      model.close();
      require(card.channel_stops == 2, "a later close must retry the failed channel stop");
      require(card.remote_stops == 1, "a stopped card worker must not be stopped again");
    }

    {
      pcie::Model model("model.tar.gz");
      build_with_running_card_worker(model);
      card.channel_stop_fails = true;
      card.remote_stop_fails = true;
      std::string error;
      try {
        model.close();
      } catch (const std::runtime_error& e) {
        error = e.what();
      }
      require(error == "channel stop deadline expired; card worker stop also failed: card worker "
                       "stop failed",
              "close must report both stop failures: " + error);
      require(card.upload_removals == 0, "close must keep uploads while the worker may run");

      card.channel_stop_fails = card.remote_stop_fails = false;
      model.close();
      require(card.remote_stops == 2, "a later close must retry the failed card worker stop");
      require(card.upload_removals == 1, "close must remove uploads once the worker stopped");
    }

    {
      auto model = std::make_unique<pcie::Model>("model.tar.gz");
      build_with_running_card_worker(*model);
      card.channel_stop_fails = true;
      model.reset();
      require(card.remote_stops == 1,
              "destroying a model must stop the card worker when channel stop fails");
      card.channel_stop_fails = false;
    }

    std::cout << "[PASS] PCIe model teardown\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
