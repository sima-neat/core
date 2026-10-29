#pragma once

// TEST-only observation of the actual Core owners. This is not a retirement
// certificate: InputStream's private worker/NULL completion must additionally
// be proven by the source-pinned stop trace, synchronous qualification profile
// and combined device/family audit. A plain transport's default Deferred policy
// alone does NOT establish synchronous NULL completion.
// In particular, neither a false Run::running() nor an empty InputStream nor
// absence of a joinable thread is substituted for device completion here.
#include "pipeline/runtime/RunInternal.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/InputStreamUtil.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace core_render_test {

// Check before public initialization and before any native ownership. These
// existing Core controls select synchronous teardown for ordinary Gst transport
// without changing its policy or the stronger driver-backed MustReachNull path.
// Incomplete synchronous attempts always warn before falling back to the reaper;
// the external source-bound audit rejects every such warning (and any detach).
inline void require_synchronous_teardown_environment() {
  for (const auto name : {"SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "SIMA_GST_TEARDOWN_ASYNC"}) {
    const char* value = std::getenv(name);
    if (!value || std::string_view(value) != "0")
      throw std::runtime_error(std::string("Core renderer qualification requires ") + name + "=0");
  }
}

inline bool is_driver_stream(std::string_view compiled_pipeline,
                             simaai::neat::pipeline_internal::InputStreamTeardownPolicy policy) {
  using Policy = simaai::neat::pipeline_internal::InputStreamTeardownPolicy;
  if (compiled_pipeline.empty())
    throw std::runtime_error("Core stream has no compiled pipeline for retirement classification");
  // Use the same semantic authority as GraphBuildInput, not a second catalogue
  // or a guess based on push/pull capability, element names or graph position.
  const bool driver = simaai::neat::inputstream_pipeline_teardown_policy(compiled_pipeline) ==
                      Policy::MustReachNull;
  if (driver && policy != Policy::MustReachNull)
    throw std::runtime_error("Core driver-backed stream does not require MustReachNull completion");
  return driver;
}

inline bool is_driver_stream(std::string_view compiled_pipeline,
                             const simaai::neat::InputStream& stream) {
  const auto policy = stream.effective_teardown_policy();
  if (!policy)
    throw std::runtime_error("Core stream has no finalized teardown policy");
  return is_driver_stream(compiled_pipeline, *policy);
}

class RunRetirementObservation {
public:
  // Call from the sole fixture controller, after the last frame is consumed
  // and before close_input/EOS. No further pushes or competing close are
  // permitted. Keep this object outside the fixture's containment try block.
  void capture(const simaai::neat::Run& run) {
    demand(owners_.empty(), "Core retirement observation already captured");
    auto root = simaai::neat::run_internal::core(run);
    demand(root != nullptr, "Core retirement observation has no Run owner");
    capture_core(std::move(root));
    demand(driver_streams_ != 0, "Core fixture captured no driver-backed stream");
  }

  // Uses only existing source-owned lifecycle operations. A failed assertion
  // throws into the caller's lifetime-preserving containment handler: it must
  // park, not unwind these retained owners or exit a native-owner process.
  void stop_and_close(simaai::neat::Run& run) {
    demand(!owners_.empty() && !closed_, "Core close observation not prepared");
    demand(simaai::neat::run_internal::core(run) == owners_.front(),
           "Core Run identity changed after capture");
    run.stop();
    for (const auto& owner : owners_)
      check_outer_stop(*owner);

    // Do not hold stop_mu while entering close: close calls stop itself.
    run.close();
    for (const auto& owner : owners_) {
      check_outer_stop(*owner);
      std::lock_guard<std::mutex> lock(owner->stop_mu);
      demand(owner->stream_close_state.load(std::memory_order_acquire) ==
                 simaai::neat::runtime::InputStreamCloseState::Closed,
             "Core stream close duty is not complete");
    }
    // Closed is evidence of RunCore's close duty only. It does not expose the
    // private InputStream worker, and does not replace the external audit.
    closed_ = true;
  }

  void release() {
    demand(closed_, "cannot release Core owners before checked outer close");
    owners_.clear();
  }

  std::size_t owner_count() const noexcept {
    return owners_.size();
  }
  std::size_t stream_count() const noexcept {
    return streams_;
  }
  std::size_t driver_stream_count() const noexcept {
    return driver_streams_;
  }

private:
  using Core = simaai::neat::runtime::RunCore;
  using Owner = std::shared_ptr<const Core>;
  using Duty = simaai::neat::runtime::InputStreamCloseState;

  static void demand(bool condition, const char* diagnostic) {
    if (!condition)
      throw std::runtime_error(diagnostic);
  }

  void capture_core(Owner core) {
    if (std::find(owners_.begin(), owners_.end(), core) != owners_.end())
      return;
    owners_.push_back(core);
    std::vector<Owner> children;
    {
      std::lock_guard<std::mutex> stop_lock(core->stop_mu);
      if (core->graph_execution_) {
        demand(!core->graph_execution_->has_detached_workers.load(std::memory_order_acquire),
               "Core graph already has detached workers");
        for (const auto& pipe : core->graph_execution_->pipelines) {
          if (!pipe)
            continue;
          // Lazy pipeline construction publishes run_core under transport.mu.
          // The fixture does not call close concurrently with this capture.
          std::lock_guard<std::mutex> build_lock(pipe->transport.mu);
          demand(!pipe->transport.building, "Core child still being constructed at capture");
          if (pipe->run_core)
            children.push_back(pipe->run_core);
          else
            demand(pipe->seg.boundary.direct_graph_source,
                   "Core fixture child pipeline has no captured owner");
        }
      } else if (core->pipeline.supports_push || core->pipeline.supports_pull) {
        const auto diag = core->pipeline.stream.diag_ctx();
        demand(diag != nullptr, "Core stream has no retained compiled diagnostics");
        // GraphBuildInput finalizes a local options copy before creating this
        // stream. RunCore's caller-options snapshot is not that authority.
        const bool driver = is_driver_stream(diag->pipeline_string, core->pipeline.stream);
        // Retain/check every leaf, including plain ingress/egress transport.
        // The external stop-end count must equal streams_, not driver_streams_.
        ++streams_;
        if (driver)
          ++driver_streams_;
      }
    }
    for (auto& child : children)
      capture_core(std::move(child));
  }

  static void check_outer_stop(const Core& core) {
    // stop_mu protects the corresponding lifecycle's std::thread joins and
    // detaches. Read those thread objects only after source stop has returned,
    // under its actual owner lock; never inspect InputStream private threads.
    std::lock_guard<std::mutex> lock(core.stop_mu);
    demand(core.stop_started, "Core source-owned stop was not entered");
    const auto duty = core.stream_close_state.load(std::memory_order_acquire);
    demand(duty != Duty::StreamStopThreadOwns && duty != Duty::InputThreadOwns,
           "Core teardown remains owned by a detached worker");
    if (!core.graph_execution_) {
      demand(core.pipeline.input_thread_done.load(std::memory_order_acquire) &&
                 !core.pipeline.input_thread.joinable(),
             "Core input controller has not completed and joined");
      return;
    }
    const auto& graph = *core.graph_execution_;
    demand(!graph.has_detached_workers.load(std::memory_order_acquire),
           "Core graph teardown detached a worker");
    for (const auto& stage : graph.stages) {
      if (stage)
        demand(stage->worker_done.load(std::memory_order_acquire) && !stage->worker.joinable(),
               "Core stage worker has not completed and joined");
    }
    for (const auto& pipe : graph.pipelines) {
      if (!pipe)
        continue;
      demand(!pipe->transport.push_thread.joinable() && !pipe->transport.pull_thread.joinable(),
             "Core pipeline transport thread was not joined");
      if (pipe->transport.has_input)
        demand(pipe->transport.push_done.load(std::memory_order_acquire),
               "Core pipeline push worker has not completed");
      if (pipe->transport.has_output)
        demand(pipe->transport.pull_done.load(std::memory_order_acquire),
               "Core pipeline pull worker has not completed");
    }
    // stop_graph unconditionally joins RealtimeLatestLink workers before its
    // return. Their private threads are not read or inferred from stats here.
    // The source-bound outer stop-end trace supplies that ordered evidence.
  }

  std::vector<Owner> owners_;
  std::size_t streams_ = 0;
  std::size_t driver_streams_ = 0;
  bool closed_ = false;
};

} // namespace core_render_test
