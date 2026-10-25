#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <simaai/scheduler/client/stage.h>

#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

struct StageStaticSpec;

// Output ownership is independent of execution slots. Reserve one carrier for
// appsink's cached current sample, plus the graph's permitted retained samples.
// Count physical frame carriers, not the number of logical tensor views.
[[nodiscard]] ::simaai::scheduler::client::Result<std::size_t>
kernel_runner_output_pool_capacity(std::size_t execution_capacity,
                                   std::size_t retained_sample_window);

struct KernelRunnerBudget {
  std::size_t execution_capacity;
  std::size_t output_pool_capacity;
};

// Preserve graph retention/spare capacity when a node authors a deeper window.
// A shallower authored window never reduces an independent output-pool floor.
[[nodiscard]] ::simaai::scheduler::client::Result<KernelRunnerBudget>
kernel_runner_stage_budget(std::optional<std::size_t> authored_capacity, std::size_t graph_capacity,
                           std::size_t graph_output_pool_capacity);

// Graph-owned EV service capacity: a single serial stage keeps one endpoint;
// overlapping frames or stages use two. Runtime still validates each recipe.
[[nodiscard]] std::size_t kernel_runner_ev_device_lanes(
    std::span<const ::simaai::scheduler::client::StageAttachment> attachments);

// Prepare every native stage before attaching any of them. Placement, storage
// and numerical contracts are resolved once while the graph is stopped.
struct KernelRunnerStageRequest {
  std::string_view element_name;
  std::string_view stage_key;
  std::size_t capacity = 1;
  std::size_t output_pool_capacity = 0;
  std::string_view compact_sink;
  const StageStaticSpec* native_stage = nullptr; // Borrowed for cold preparation only.
  std::string_view terminal_host_sink;
};
[[nodiscard]] ::simaai::scheduler::client::Status
prepare_and_attach_kernel_runners(GstElement* pipeline,
                                  std::span<const KernelRunnerStageRequest> requests,
                                  std::size_t a65_workers = 2);

} // namespace simaai::neat::pipeline_internal::sima
