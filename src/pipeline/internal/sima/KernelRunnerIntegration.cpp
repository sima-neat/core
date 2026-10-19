#include "pipeline/internal/sima/KernelRunnerIntegration.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/PreparedRuntimeBuild.h"
#include <gst/SimaPluginStaticManifestAbi.h>
#include <algorithm>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {
namespace rt = ::simaai::scheduler::client;
namespace gs = ::simaai::scheduler::client;
namespace {
rt::Error invalid(const char* field, std::string message) {
  return {rt::ErrorCode::InvalidRequest, field, std::move(message)};
}
template<class T> auto retain_gst(T* value) {
  return std::shared_ptr<T>(value, [](T* p) { if (p) gst_object_unref(p); });
}
}
rt::Result<std::size_t> kernel_runner_output_pool_capacity(std::size_t execution_capacity,
                                                           std::size_t retained_sample_window) {
  // Runtime permits 4096 execution slots and its existing extra carrier. Core
  // limits active execution to 64 slots per stage.
  constexpr std::size_t max_output_carriers = 4097;
  if (execution_capacity == 0 || execution_capacity > 64 ||
      retained_sample_window > max_output_carriers - execution_capacity - 1) {
    return invalid("kernel-runner.output-pool",
                   "execution and retained-output budgets exceed the bounded pool");
  }
  // GstAppSink retains its latest pulled sample even when enable-last-sample is
  // false. This spare lets the next publication replace that cached carrier.
  return execution_capacity + retained_sample_window + 1;
}


rt::Status wire_kernel_stage_factories(const KernelStageRecipe& recipe,
                                     gs::TensorStageRequest& request) {
  if (request.frame_adapter_factory || request.members.size() != recipe.members.size())
    return invalid("kernel-runner.instance", "fresh request with matching checked members required");
  for (std::size_t i = 0; i < recipe.members.size(); ++i)
    if (recipe.members[i].resolve_factory || request.members[i].resolve_factory ||
        request.members[i].executable != recipe.members[i].executable)
      return invalid("kernel-runner.instance", "recipe must contain immutable checked descriptors only");
  std::vector<bool> seen(recipe.members.size(), false);
  for (const auto& factory : recipe.resolver_factories) {
    if (factory.member >= seen.size() || seen[factory.member] || !factory.instantiate)
      return invalid("kernel-runner.instance", "invalid or duplicate resolver factory member");
    seen[factory.member] = true;
  }
  if (recipe.frame_adapter_factory &&
      (recipe.members.size() != 1 || !recipe.resolver_factories.empty()))
    return invalid("kernel-runner.instance", "image policy requires one member without resolvers");
  // Wiring is cold data transfer, not admission. No factory is invoked here.
  for (const auto& factory : recipe.resolver_factories)
    request.members[factory.member].resolve_factory = factory.instantiate;
  request.frame_adapter_factory = recipe.frame_adapter_factory;
  return {};
}
std::size_t kernel_runner_ev_device_lanes(std::span<const gs::TensorStageRequest> requests) {
  bool ev_stage = false;
  for (const auto& request : requests) {
    if (request.target != rt::Hardware::EV74)
      continue;
    // Invalid recipes are rejected by Scheduler before hardware connection.
    if (ev_stage || request.options.capacity > 1)
      return 2;
    ev_stage = true;
  }
  return 1;
}


rt::Status prepare_and_attach_kernel_runners(GstElement* pipeline,
    std::span<const KernelRunnerStageRequest> requests, std::size_t a65_workers) {
  if (!pipeline || !GST_IS_BIN(pipeline))
    return invalid("kernel-runner.graph", "a candidate graph bin is required");
  std::vector<std::shared_ptr<GstElement>> elements, sinks;
  std::vector<gs::StageAttachment> attachments;
  std::vector<gs::TensorStageRequest> preparations;
  elements.reserve(requests.size());
  sinks.reserve(requests.size());
  attachments.reserve(requests.size());
  preparations.reserve(requests.size());
  auto manifest = std::shared_ptr<GstContext>(
      gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE),
      [](GstContext* p) { if (p) gst_context_unref(p); });
  for (const auto& request : requests) {
    if (request.capacity == 0 || request.capacity > 64)
      return invalid("kernel-runner.graph", "Core execution capacity must be between 1 and 64");
    const std::string name(request.element_name), key(request.stage_key);
    auto element = retain_gst(gst_bin_get_by_name(GST_BIN(pipeline), name.c_str()));
    if (!element)
      return invalid("kernel-runner.graph", "selected semantic stage was not rendered");
    SimaPluginManifestLookupStatus status{};
    const auto* placement = sima_plugin_manifest_context_stage_lookup_typed_checked(
        manifest.get(), key.c_str(), name.c_str(), SIMA_PLUGIN_STAGE_PAYLOAD_NONE, &status);
    if (!placement)
      return invalid("kernel-runner.placement", sima_plugin_manifest_lookup_status_name(status));
    const auto* native = request.native_stage;
    if (!native || !native->native_kernel)
      return invalid("kernel-runner.recipe", "stage has no Core-authored canonical recipe");
    const auto& recipe = *native->native_kernel;
    gs::TensorStageRequest prepared;
    prepared.target = recipe.target;
    prepared.members = recipe.members;
    prepared.constant_data = recipe.constant_data;
    prepared.placement = std::shared_ptr<const SimaPluginStageSpec>(manifest, placement);
    prepared.options.capacity = request.capacity;
    prepared.options.output_pool_capacity = request.output_pool_capacity;
    std::string error;
    if (!build_kernel_stage_publication(*native, &prepared.publication, &error))
      return invalid("kernel-runner.publication", std::move(error));
    if (auto failure = wire_kernel_stage_factories(recipe, prepared))
      return failure;
    auto sink = retain_gst(request.compact_sink.empty() ? nullptr :
        gst_bin_get_by_name(GST_BIN(pipeline), std::string(request.compact_sink).c_str()));
    preparations.push_back(prepared);
    attachments.push_back({element.get(), std::move(prepared), recipe.image_publication,
                           recipe.metrics, sink.get()});
    elements.push_back(std::move(element));
    sinks.push_back(std::move(sink));
  }
  gs::TensorStageCohortOptions options;
  options.a65_workers = a65_workers;
  options.ev74_device_lanes = kernel_runner_ev_device_lanes(preparations);
  // Scheduler owns stopped-topology checks, storage/session selection, fresh
  // factory admission and all-or-nothing context/coherency publication.
  return gs::prepare_and_attach_tensor_stages(pipeline, attachments, options,
      gs::CompactFrameReaderContract::SourceValuesV3);
}
} // namespace simaai::neat::pipeline_internal::sima
