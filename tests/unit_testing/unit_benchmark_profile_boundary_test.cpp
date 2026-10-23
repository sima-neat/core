#include "model/internal/MlaLayerStats.h"
#include "test_main.h"

#include <simaai/scheduler/client/mla_layer_profile.h>

#include <functional>
#include <stdexcept>

namespace {
namespace client = simaai::scheduler::client;
using namespace simaai::neat;

int calls = 0;
int failure_mode = 0;
client::OfflineMlaProfileAvailability availability =
    client::OfflineMlaProfileAvailability::Available;

internal::LayerProfileStage stage(std::size_t index, const char* executable) {
  return {index, executable, "source.yaml", {{1, {"real", 5, 20, 0}}}};
}

void require_failure(const std::function<void()>& action, const std::string& expected) {
  try {
    action();
  } catch (const std::exception& error) {
    require(std::string(error.what()).find(expected) != std::string::npos, error.what());
    return;
  }
  require(false, "expected failure: " + expected);
}
} // namespace

// Executable-local replacement of the declared Scheduler client boundary. The
// real Core collector is compiled into this test; no product hook or hardware.
namespace simaai::scheduler::client {
OfflineMlaProfileCapability mla_layer_statistics_capability() noexcept {
  return {availability, "profile capability fixture"};
}

OfflineMlaLayerProfileResult
profile_mla_layers_offline(const OfflineMlaLayerProfileRequest& request) {
  ++calls;
  require(request.executable.filename() == std::to_string(calls) + ".elf",
          "Core forwards the selected executable in execution order");
  if (calls == 2) {
    if (failure_mode == 1)
      throw OfflineMlaProfileError(OfflineMlaProfileCompletion::NotLaunched,
                                   OfflineMlaProfileErrorCode::Unavailable, "provider missing");
    if (failure_mode == 2)
      throw OfflineMlaProfileError(OfflineMlaProfileCompletion::Unknown,
                                   OfflineMlaProfileErrorCode::RecoveryRequired,
                                   "recovery required");
    if (failure_mode == 3)
      throw OfflineMlaProfileError(OfflineMlaProfileCompletion::NotLaunched,
                                   OfflineMlaProfileErrorCode::Busy,
                                   "another diagnostic is active");
    if (failure_mode == 4)
      return {"not a mapping", "SECOND-DIAGNOSTIC"};
    if (failure_mode == 6)
      throw OfflineMlaProfileError(static_cast<OfflineMlaProfileCompletion>(999),
                                   OfflineMlaProfileErrorCode::DataInvalid,
                                   "unknown completion enum");
    if (failure_mode == 7)
      throw OfflineMlaProfileError(OfflineMlaProfileCompletion::NotLaunched,
                                   static_cast<OfflineMlaProfileErrorCode>(999),
                                   "unknown error code");
    if (failure_mode == 5)
      throw OfflineMlaProfileError(OfflineMlaProfileCompletion::Completed,
                                   OfflineMlaProfileErrorCode::DataInvalid,
                                   "future authoritative completed-data failure");
  }
  return {"1:\n  name: real\n  start_cycle: 5\n  end_cycle: 20\n  run_time: 12.5us\n", ""};
}
} // namespace simaai::scheduler::client

RUN_TEST(
    "unit_benchmark_profile_boundary_test", ([] {
      const std::vector<internal::LayerProfileStage> stages{stage(2, "1.elf"), stage(7, "2.elf")};
      BenchmarkReport report;
      report.fps = 123;
      report.latency_ms = 4;
      internal::collect_mla_layer_stats_for_report(report, stages);
      require(report.layer_stats_status == "ok" && report.layer_stats.size() == 2 &&
                  report.layer_stats[0].stage_index == 2 && report.layer_stats[1].stage_index == 7,
              "real collector preserves stage identity and complete success");
      failure_mode = 1;
      calls = 0;
      internal::collect_mla_layer_stats_for_report(report, stages);
      require(
          report.layer_stats.empty() && report.fps == 123 && report.latency_ms == 4 &&
              report.layer_stats_status.find("provider missing") != std::string::npos &&
              report.layer_stats_status.find("MLA stage 7") != std::string::npos,
          "safe second-stage failure preserves headlines, empties partial rows and names stage");
      // Completed is reserved for a future authoritative provider completion proof.
      failure_mode = 5;
      calls = 0;
      internal::collect_mla_layer_stats_for_report(report, stages);
      require(report.layer_stats.empty() && report.fps == 123 &&
                  report.layer_stats_status.find("completed-data") != std::string::npos,
              "authoritatively completed data errors preserve headline measurements");
      for (const auto& [mode, reason] :
           std::vector<std::pair<int, std::string>>{{2, "recovery required"},
                                                    {3, "another diagnostic is active"},
                                                    {4, "SECOND-DIAGNOSTIC"},
                                                    {6, "unknown completion enum"},
                                                    {7, "unknown error code"}}) {
        failure_mode = mode;
        calls = 0;
        require_failure([&] { internal::collect_mla_layer_stats_for_report(report, stages); },
                        reason);
      }
      for (const auto state : {client::OfflineMlaProfileAvailability::Busy,
                               client::OfflineMlaProfileAvailability::RecoveryRequired}) {
        availability = state;
        require_failure([] { internal::require_owned_mla_layer_profiling(); },
                        "profile capability fixture");
      }
      availability = static_cast<client::OfflineMlaProfileAvailability>(999);
      require_failure([] { internal::require_owned_mla_layer_profiling(); },
                      "unrecognized MLA profiling capability");
      availability = client::OfflineMlaProfileAvailability::Unavailable;
      pipeline_internal::sima::MpkContract contract;
      std::vector<internal::LayerProfileStage> prepared = stages;
      const auto reason = internal::prepare_mla_layer_stats(contract, {}, prepared);
      require(prepared.empty() && reason == "profile capability fixture",
              "missing provider is a safe pre-launch status, not a benchmark failure");
      availability = client::OfflineMlaProfileAvailability::Available;
      calls = 0;
      const auto metadata_reason = internal::prepare_mla_layer_stats(contract, {}, prepared);
      require(!metadata_reason.empty() && prepared.empty() && calls == 0,
              "invalid local diagnostic metadata is reported before any provider call");
    }));
