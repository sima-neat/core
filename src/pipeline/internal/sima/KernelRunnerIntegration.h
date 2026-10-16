#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <simaai/runtime/gstreamer/prepared_tensor_stage.h>
#include <simaai/runtime/gstreamer/stage.h>
#include <simaai/runtime/gstreamer/metrics.h>

#include <memory>
#include <span>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

struct StageStaticSpec;

struct KernelRunnerStageBinding {
  GstElement* element = nullptr; // Borrowed; must belong to the candidate bin.
  std::shared_ptr<const ::simaai::runtime::gstreamer::Stage> stage;
  std::shared_ptr<const ::simaai::runtime::gstreamer::StageMetrics> metrics;
};

// Default graph-construction attachment for prepared numerical stages.
// The caller prepares the entire matched cohort, transport and resource budgets
// before calling this function. This helper never opens hardware or renders a
// replacement launch string. It validates every attachment before changing any
// element; callers must serialize pipeline construction/state changes.
// Empty bindings are valid only when the bin contains no kernel runners.
// Legacy execution factories are deliberately rejected, not silently mixed
// with native-epoch EV firmware. All candidate elements must be in NULL state.
[[nodiscard]] ::simaai::runtime::Status
attach_kernel_runner_stages(GstElement* candidate_pipeline,
                            std::span<const KernelRunnerStageBinding> bindings);

// Output ownership is independent of execution slots. Reserve one carrier for
// appsink's cached current sample, plus the graph's permitted retained samples.
// Count physical frame carriers, not the number of logical tensor views.
[[nodiscard]] ::simaai::runtime::Result<std::size_t>
kernel_runner_output_pool_capacity(std::size_t execution_capacity,
                                   std::size_t retained_sample_window);

// Graph-owned EV service capacity: a single serial stage keeps one endpoint;
// overlapping frames or stages use two. Runtime still validates each recipe.
[[nodiscard]] std::size_t kernel_runner_ev_device_lanes(
    std::span<const ::simaai::runtime::gstreamer::TensorStageRequest> requests);

// Prepare every native stage before attaching any of them. Placement, storage
// and numerical contracts are resolved once while the graph is stopped.
struct KernelRunnerStageRequest {
  std::string_view element_name;
  std::string_view stage_key;
  std::size_t capacity = 1;
  std::size_t output_pool_capacity = 0;
  std::string_view compact_sink;
  const StageStaticSpec* native_stage = nullptr; // Borrowed for cold preparation only.
};
[[nodiscard]] ::simaai::runtime::Status
prepare_and_attach_kernel_runners(GstElement* pipeline,
                                  std::span<const KernelRunnerStageRequest> requests,
                                  std::size_t a65_workers = 2);

} // namespace simaai::neat::pipeline_internal::sima
