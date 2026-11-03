#include "pipeline/internal/sima/SchedulerStageIntegration.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
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
template <class T> auto retain_gst(T* value) {
  return std::shared_ptr<T>(value, [](T* p) {
    if (p)
      gst_object_unref(p);
  });
}
} // namespace
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

rt::Result<KernelRunnerBudget>
kernel_runner_stage_budget(std::optional<std::size_t> authored_capacity, std::size_t graph_capacity,
                           std::size_t graph_output_pool_capacity) {
  auto minimum = kernel_runner_output_pool_capacity(graph_capacity, 0);
  if (const auto* failure = std::get_if<rt::Error>(&minimum))
    return *failure;
  const auto graph_pool =
      graph_output_pool_capacity ? graph_output_pool_capacity : std::get<std::size_t>(minimum);
  if (graph_pool < std::get<std::size_t>(minimum) || graph_pool > 4097)
    return invalid("kernel-runner.output-pool", "invalid graph output carrier budget");
  const auto execution = authored_capacity.value_or(graph_capacity);
  // Validation above makes this subtraction safe. The surplus may include a
  // downstream pool floor, so preserve it conservatively rather than shrink it.
  auto expanded = kernel_runner_output_pool_capacity(execution, graph_pool - graph_capacity - 1);
  if (const auto* failure = std::get_if<rt::Error>(&expanded))
    return *failure;
  return KernelRunnerBudget{execution, std::max(graph_pool, std::get<std::size_t>(expanded))};
}

rt::Status wire_kernel_stage_factories(const KernelStageRecipe& recipe,
                                       gs::KernelProgram& request) {
  if (request.frame_adapter_factory || request.members.size() != recipe.members.size())
    return invalid("kernel-runner.instance",
                   "fresh request with matching checked members required");
  for (std::size_t i = 0; i < recipe.members.size(); ++i)
    if (recipe.members[i].resolve_factory || request.members[i].resolve_factory ||
        request.members[i].executable != recipe.members[i].executable)
      return invalid("kernel-runner.instance",
                     "recipe must contain immutable checked descriptors only");
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
std::size_t kernel_runner_ev_device_lanes(std::span<const gs::StageAttachment> attachments) {
  bool ev_stage = false;
  for (const auto& attachment : attachments) {
    const auto& request = attachment.request;
    const auto* program = std::get_if<gs::KernelProgram>(&request.program);
    if (!program || program->target != rt::Hardware::EV74)
      continue;
    // Invalid recipes are rejected by Scheduler before hardware connection.
    if (ev_stage || request.options.capacity > 1)
      return 2;
    ev_stage = true;
  }
  return 1;
}

rt::Status prepare_and_attach_stages(GstElement* pipeline,
                                     std::span<const SchedulerStageRequest> requests,
                                     std::size_t a65_workers) {
  if (!pipeline || !GST_IS_BIN(pipeline))
    return invalid("kernel-runner.graph", "a candidate graph bin is required");
  std::vector<std::shared_ptr<GstElement>> elements, sinks;
  std::vector<gs::StageAttachment> attachments;
  gs::HostCpuTerminalEndpoint terminal_host;
  std::shared_ptr<GstElement> terminal_host_sink;
  elements.reserve(requests.size());
  sinks.reserve(requests.size());
  attachments.reserve(requests.size());
  for (const auto& request : requests) {
    if (request.capacity == 0 || request.capacity > 64)
      return invalid("kernel-runner.graph", "Core execution capacity must be between 1 and 64");
    const std::string name(request.element_name);
    auto element = retain_gst(gst_bin_get_by_name(GST_BIN(pipeline), name.c_str()));
    if (!element)
      return invalid("kernel-runner.graph", "selected semantic stage was not rendered");
    const auto* stage = request.stage;
    if (!stage || !stage->frame_layout || !stage->tensor_publication)
      return invalid("stage.layout", "finalized Core layout and publication are required");
    gs::StageRequest prepared;
    prepared.layout = stage->frame_layout;
    prepared.publication = *stage->tensor_publication;
    prepared.options.capacity = request.capacity;
    prepared.options.output_pool_capacity = request.output_pool_capacity;
    std::shared_ptr<const gs::TensorStageImagePublication> image;
    std::shared_ptr<const gs::StageMetrics> metrics;
    if (stage->native_kernel) {
      const auto& recipe = *stage->native_kernel;
      gs::KernelProgram program;
      program.target = recipe.target;
      program.members = recipe.members;
      program.constant_data = recipe.constant_data;
      if (auto failure = wire_kernel_stage_factories(recipe, program))
        return failure;
      prepared.program = std::move(program);
      image = stage->image_publication;
      metrics = recipe.metrics;
    } else if (stage->tvm_program) {
      prepared.program = *stage->tvm_program;
    } else if (stage->mla_program) {
      prepared.program = *stage->mla_program;
    } else
      return invalid("stage.program", "final stage has no typed program");
    auto sink = retain_gst(
        request.compact_sink.empty()
            ? nullptr
            : gst_bin_get_by_name(GST_BIN(pipeline), std::string(request.compact_sink).c_str()));
    if (!request.terminal_host_sink.empty()) {
      if (terminal_host.stage)
        return invalid("kernel-runner.host-terminal", "one proven public terminal is required");
      terminal_host_sink = retain_gst(
          gst_bin_get_by_name(GST_BIN(pipeline), std::string(request.terminal_host_sink).c_str()));
      terminal_host = {element.get(), terminal_host_sink.get()};
    }
    attachments.push_back(
        {element.get(), std::move(prepared), std::move(image), std::move(metrics), sink.get()});
    elements.push_back(std::move(element));
    sinks.push_back(std::move(sink));
  }
  gs::StageCohortOptions options;
  options.a65_workers = a65_workers;
  options.ev74_device_lanes = kernel_runner_ev_device_lanes(attachments);
  // Scheduler owns stopped-topology checks, storage/session selection, fresh
  // factory admission and all-or-nothing context/coherency publication.
  return gs::prepare_and_attach_stages(pipeline, attachments, options, terminal_host);
}
} // namespace simaai::neat::pipeline_internal::sima
