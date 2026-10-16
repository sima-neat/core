#include "pipeline/internal/sima/KernelRunnerIntegration.h"
#include "pipeline/internal/MemoryCoherencyPolicy.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/PreparedRuntimeBuild.h"

#include <simaai/runtime/gstreamer/prepared_tensor_stage.h>
#include <simaai/runtime/gstreamer/storage.h>
#include <gst/SimaPluginStaticManifestAbi.h>
#include <gstsimaaitensorbuffer.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {
namespace {
namespace rt = simaai::runtime;
namespace gs = rt::gstreamer;
rt::Error invalid(const char* field, std::string message) {
  return {rt::ErrorCode::InvalidRequest, field, std::move(message)};
}
bool stopped(GstElement* element) {
  GstState state = GST_STATE_VOID_PENDING;
  GstState pending = GST_STATE_VOID_PENDING;
  gst_element_get_state(element, &state, &pending, 0);
  return state == GST_STATE_NULL && pending == GST_STATE_VOID_PENDING;
}
const char* factory_name(GstElement* element) {
  auto* factory = gst_element_get_factory(element);
  return factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : "";
}
bool legacy_execution(const char* factory) {
  constexpr std::array legacy{"neatprocesscvu", "neatpreparedrunner"};
  return std::any_of(legacy.begin(), legacy.end(),
                     [factory](const char* name) { return g_str_equal(factory, name); });
}
rt::Status check_pads(GstElement* element, std::size_t expected) {
  std::vector<bool> seen(expected, false);
  auto* iterator = gst_element_iterate_sink_pads(element);
  GValue value = G_VALUE_INIT;
  rt::Status failure;
  for (;;) {
    const auto result = gst_iterator_next(iterator, &value);
    if (result == GST_ITERATOR_DONE)
      break;
    if (result != GST_ITERATOR_OK) {
      failure = invalid("kernel-runner.pads", "candidate pad topology changed during validation");
      break;
    }
    auto* pad = GST_PAD(g_value_get_object(&value));
    bool matched = false;
    for (std::size_t index = 0; index < expected; ++index) {
      const auto name = "sink_" + std::to_string(index);
      if (g_str_equal(GST_PAD_NAME(pad), name.c_str()) && !seen[index]) {
        seen[index] = true;
        matched = true;
        break;
      }
    }
    g_value_reset(&value);
    if (!matched) {
      failure = invalid("kernel-runner.pads", "sink pads do not match the prepared stage");
      break;
    }
  }
  if (G_IS_VALUE(&value))
    g_value_unset(&value);
  gst_iterator_free(iterator);
  if (!failure && std::find(seen.begin(), seen.end(), false) != seen.end())
    failure = invalid("kernel-runner.pads", "a required prepared sink pad is missing");
  return failure;
}
rt::Status check_candidate(GstElement* pipeline, std::span<GstElement* const> elements) {
  if (!pipeline || !GST_IS_BIN(pipeline) || !stopped(pipeline))
    return invalid("kernel-runner.pipeline",
                   "an explicitly built, stopped candidate bin is required");
  for (std::size_t i = 0; i < elements.size(); ++i) {
    auto* element = elements[i];
    if (!element || !gst_object_has_as_ancestor(GST_OBJECT(element), GST_OBJECT(pipeline)) ||
        !g_str_equal(factory_name(element), "neatkernelrunner") || !stopped(element))
      return invalid("kernel-runner.element",
                     "attachment must name a stopped candidate kernel element");
    if (std::find(elements.begin(), elements.begin() + i, element) != elements.begin() + i)
      return invalid("kernel-runner.element", "duplicate candidate element attachment");
  }
  // Validate the whole bin before attaching even the first context. A partial
  // switch is unsafe when native and legacy EV firmware epochs differ.
  auto* iterator = gst_bin_iterate_recurse(GST_BIN(pipeline));
  GValue value = G_VALUE_INIT;
  rt::Status failure;
  for (;;) {
    const auto result = gst_iterator_next(iterator, &value);
    if (result == GST_ITERATOR_DONE)
      break;
    if (result != GST_ITERATOR_OK) {
      failure = invalid("kernel-runner.pipeline", "candidate topology changed during validation");
      break;
    }
    auto* element = GST_ELEMENT(g_value_get_object(&value));
    const auto* factory = factory_name(element);
    if (legacy_execution(factory))
      failure =
          invalid("kernel-runner.cohort", "candidate bin contains a legacy execution factory");
    else if (g_str_equal(factory, "neatkernelrunner") &&
             std::none_of(elements.begin(), elements.end(),
                          [element](auto* candidate) { return candidate == element; }))
      failure = invalid("kernel-runner.element", "candidate kernel element has no prepared stage");
    g_value_reset(&value);
    if (failure)
      break;
  }
  if (G_IS_VALUE(&value))
    g_value_unset(&value);
  gst_iterator_free(iterator);
  if (failure)
    return failure;
  return {};
}
// This is cold, whole-bin admission, before any prepared member opens a device.
// Fan-out is allowed: readers retain their current loans and writers invalidate
// the canonical generation. We do not retain allocation-wide READ epochs.
rt::Result<gs::MemoryCoherencyMode> select_memory_coherency(GstElement* pipeline, std::vector<GstElement*>& mla) {
  if (!demand_coherency_available()) return gs::MemoryCoherencyMode::Legacy;
  if (!pipeline || !GST_IS_BIN(pipeline) || !stopped(pipeline))
    return invalid("memory-coherency.pipeline", "a stopped candidate bin is required");
  if (!graph_participates_in_demand_coherency(pipeline))
    return gs::MemoryCoherencyMode::Legacy;
  auto* iterator = gst_bin_iterate_recurse(GST_BIN(pipeline));
  GValue value = G_VALUE_INIT;
  for (;;) {
    const auto step = gst_iterator_next(iterator, &value);
    if (step == GST_ITERATOR_DONE) break;
    if (step != GST_ITERATOR_OK) {
      if (G_IS_VALUE(&value)) g_value_unset(&value);
      gst_iterator_free(iterator);
      return invalid("memory-coherency.pipeline", "candidate changed during cold admission");
    }
    auto* element = GST_ELEMENT(g_value_get_object(&value));
    if (g_str_equal(factory_name(element), "neatprocessmla")) mla.push_back(element);
    g_value_reset(&value);
  }
  if (G_IS_VALUE(&value)) g_value_unset(&value);
  gst_iterator_free(iterator);
  if (gs::memory_coherency_api_revision() != 1)
    return invalid("memory-coherency.revision", "Scheduler/Runner memory coherency V1 is required");
  for (auto* element : mla) {
    auto* property = g_object_class_find_property(G_OBJECT_GET_CLASS(element), "memory-coherency-mode");
    if (!property || G_PARAM_SPEC_VALUE_TYPE(property) != G_TYPE_STRING)
      return invalid("memory-coherency.mla", "MLA plugin does not expose coherency V1");
  }
  return gs::MemoryCoherencyMode::DemandLocalV1;
}
} // namespace

rt::Status attach_kernel_runner_stages(GstElement* pipeline,
                                       std::span<const KernelRunnerStageBinding> bindings) {
  if (!pipeline || !GST_IS_BIN(pipeline) || !stopped(pipeline))
    return invalid("kernel-runner.pipeline",
                   "an explicitly built, stopped candidate bin is required");
  std::vector<GstElement*> elements;
  elements.reserve(bindings.size());
  for (const auto& binding : bindings)
    elements.push_back(binding.element);
  if (auto failure = check_candidate(pipeline, elements))
    return failure;
  for (const auto& binding : bindings) {
    if (!binding.stage)
      return invalid("kernel-runner.stage", "a prepared stage is required");
    const auto& stage = *binding.stage;
    if (!stage.bridge || !stage.output_caps || !gst_caps_is_fixed(stage.output_caps.get()) ||
        stage.capacity == 0 || stage.capacity > 4096 || stage.input_pads == 0 ||
        stage.input_pads > 1024 || stage.max_in_flight_members == 0)
      return invalid("kernel-runner.stage", "invalid prepared stage capacity, caps or provider");
    for (std::size_t member = 0; member < stage.members.size(); ++member) {
      const auto& entry = stage.members[member];
      if (!entry.executor || !entry.operation ||
          std::any_of(entry.dependencies.begin(), entry.dependencies.end(),
                      [member](auto dependency) { return dependency >= member; }))
        return invalid("kernel-runner.stage", "invalid prepared member or dependency ordering");
    }
    if (binding.metrics) {
      if (auto failure = gs::validate_metrics(stage, *binding.metrics))
        return failure;
    }
    if (auto failure = check_pads(binding.element, stage.input_pads))
      return failure;
  }
  std::vector<std::shared_ptr<GstContext>> contexts;
  contexts.reserve(bindings.size());
  for (const auto& binding : bindings)
    contexts.emplace_back(gs::make_context(binding.stage, binding.metrics), gst_context_unref);
  for (std::size_t i = 0; i < bindings.size(); ++i)
    gst_element_set_context(bindings[i].element, contexts[i].get());
  return {};
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

rt::Status instantiate_kernel_stage_state(const KernelStageRecipe& recipe,
                                          gs::TensorStageRequest& request) {
  if (request.frame_adapter || request.members.size() != recipe.members.size())
    return invalid("kernel-runner.instance",
                   "fresh request with matching checked members required");
  for (std::size_t i = 0; i < recipe.members.size(); ++i)
    if (recipe.members[i].resolve || request.members[i].resolve ||
        request.members[i].executable != recipe.members[i].executable)
      return invalid("kernel-runner.instance",
                     "recipe must contain immutable checked descriptors only");
  std::vector<KernelStageRecipe::Resolver> resolvers(recipe.members.size());
  std::vector<bool> seen(recipe.members.size(), false);
  for (const auto& factory : recipe.resolver_factories) {
    if (factory.member >= seen.size() || seen[factory.member] || !factory.instantiate)
      return invalid("kernel-runner.instance", "invalid or duplicate resolver factory member");
    seen[factory.member] = true;
  }
  if (recipe.frame_adapter_factory &&
      (recipe.members.size() != 1 || !recipe.resolver_factories.empty()))
    return invalid("kernel-runner.instance", "image policy requires one member without resolvers");
  // Do not commit partly constructed state on a factory error. No factory may
  // open a session; Scheduler validates and admits the complete cohort later.
  std::shared_ptr<gs::TensorStageFrameAdapter> adapter;
  for (const auto& factory : recipe.resolver_factories) {
    resolvers[factory.member] = factory.instantiate();
    if (!resolvers[factory.member])
      return invalid("kernel-runner.instance", "resolver factory returned no instance");
  }
  if (recipe.frame_adapter_factory) {
    auto result = recipe.frame_adapter_factory();
    if (const auto* failure = std::get_if<rt::Error>(&result))
      return *failure;
    adapter = std::get<std::shared_ptr<gs::TensorStageFrameAdapter>>(std::move(result));
    if (!adapter || adapter->session().size())
      return invalid("kernel-runner.instance",
                     "adapter factory must return fresh unadmitted state");
  }
  for (std::size_t i = 0; i < resolvers.size(); ++i)
    request.members[i].resolve = std::move(resolvers[i]);
  request.frame_adapter = std::move(adapter);
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
                                             std::span<const KernelRunnerStageRequest> requests,
                                             std::size_t a65_workers) {
  std::vector<GstElement*> mla_elements;
  const auto selected_memory = select_memory_coherency(pipeline, mla_elements);
  if (const auto* failure = std::get_if<rt::Error>(&selected_memory)) return *failure;
  const auto memory_mode = std::get<gs::MemoryCoherencyMode>(selected_memory);
  std::vector<gs::TensorStageRequest> preparations;
  preparations.reserve(requests.size());
  std::vector<std::shared_ptr<GstElement>> elements;
  std::vector<KernelRunnerStageBinding> bindings;
  std::vector<std::shared_ptr<GstPad>> compact_peers;
  std::vector<GstElement*> candidates;
  elements.reserve(requests.size());
  bindings.reserve(requests.size());
  candidates.reserve(requests.size());
  for (const auto& request : requests) {
    const auto& [element_name, stage_key, capacity, output_pool_capacity, compact_sink,
                 native_stage] = request;
    if (!pipeline || !GST_IS_BIN(pipeline) || !stopped(pipeline) || capacity == 0 || capacity > 64)
      return invalid("kernel-runner.graph", "a stopped graph and bounded capacity are required");
    const std::string name(element_name);
    const std::string key(stage_key);
    auto element = std::shared_ptr<GstElement>(gst_bin_get_by_name(GST_BIN(pipeline), name.c_str()),
                                               [](auto* value) {
                                                 if (value)
                                                   gst_object_unref(value);
                                               });
    if (!element)
      return invalid("kernel-runner.graph", "selected semantic stage was not rendered");
    auto manifest = std::shared_ptr<GstContext>(
        gst_element_get_context(pipeline, SIMA_PLUGIN_STATIC_MANIFEST_CONTEXT_TYPE),
        [](auto* value) {
          if (value)
            gst_context_unref(value);
        });
    SimaPluginManifestLookupStatus status{};
    const auto* placement = sima_plugin_manifest_context_stage_lookup_typed_checked(
        manifest.get(), key.c_str(), name.c_str(), SIMA_PLUGIN_STAGE_PAYLOAD_NONE, &status);
    if (!placement)
      return invalid("kernel-runner.placement", sima_plugin_manifest_lookup_status_name(status));
    gs::TensorStageOptions options;
    options.storage = gs::make_storage_with_memory_v1(memory_mode);
    options.capacity = capacity;
    options.output_pool_capacity = output_pool_capacity;
    if (!compact_sink.empty()) {
      const std::string sink_name(compact_sink);
      auto sink = std::shared_ptr<GstElement>(
          gst_bin_get_by_name(GST_BIN(pipeline), sink_name.c_str()), [](auto* value) {
            if (value)
              gst_object_unref(value);
          });
      auto src = std::shared_ptr<GstPad>(gst_element_get_static_pad(element.get(), "src"),
                                         [](auto* value) {
                                           if (value)
                                             gst_object_unref(value);
                                         });
      auto peer =
          std::shared_ptr<GstPad>(src ? gst_pad_get_peer(src.get()) : nullptr, [](auto* value) {
            if (value)
              gst_object_unref(value);
          });
      auto peer_element = std::shared_ptr<GstElement>(
          peer ? gst_pad_get_parent_element(peer.get()) : nullptr, [](auto* value) {
            if (value)
              gst_object_unref(value);
          });
      // Only this graph-owned direct edge is admitted. A queue, tee, custom
      // consumer or inserted observer boundary cannot inherit this promise.
      if (sink && peer_element == sink && g_str_equal(factory_name(sink.get()), "appsink")) {
        compact_peers.push_back(std::move(peer));
        options.compact_publication = true;
      }
    }
    if (!native_stage || !native_stage->native_kernel)
      return invalid("kernel-runner.recipe", "stage has no Core-authored canonical recipe");
    gs::TensorStageRequest prepared;
    prepared.target = native_stage->native_kernel->target;
    prepared.members = native_stage->native_kernel->members;
    prepared.constant_data = native_stage->native_kernel->constant_data;
    prepared.placement = std::shared_ptr<const SimaPluginStageSpec>(manifest, placement);
    prepared.options = std::move(options);
    std::string error;
    if (!build_kernel_stage_publication(*native_stage, &prepared.publication, &error))
      return invalid("kernel-runner.publication", std::move(error));
    std::size_t input_pads = native_stage->native_kernel->frame_adapter_factory ? 1 : 0;
    preparations.push_back(std::move(prepared));
    for (guint i = 0; placement->input_bindings && i < placement->input_bindings_len; ++i) {
      const auto& binding = placement->input_bindings[i];
      if (binding.required) {
        if (binding.sink_pad_index < 0 || binding.sink_pad_index >= 64)
          return invalid("kernel-runner.pads", "required input pad is outside the supported bound");
        input_pads = std::max(input_pads, std::size_t(binding.sink_pad_index) + 1);
      }
    }
    if (auto failure = check_pads(element.get(), input_pads))
      return failure;
    candidates.push_back(element.get());
    elements.push_back(std::move(element));
  }
  // Topology is known before numerical preparation opens a device. Runtime
  // separately checks operation, placement and budget contracts before connect.
  if (auto failure = check_candidate(pipeline, candidates))
    return failure;
  for (std::size_t i = 0; i < preparations.size(); ++i)
    if (auto failure = instantiate_kernel_stage_state(*requests[i].native_stage->native_kernel,
                                                      preparations[i]))
      return failure;
  gs::TensorStageCohortOptions options;
  options.a65_workers = a65_workers;
  // One shared execution domain per target; two EV endpoints only when graph
  // concurrency requires them. No operation dispatch or transport per member.
  options.ev74.ev74.device_lanes = kernel_runner_ev_device_lanes(preparations);
  auto result = gs::prepare_tensor_stages_with_memory_v1(preparations, std::move(options), memory_mode);
  if (auto* failure = std::get_if<rt::Error>(&result))
    return *failure;
  auto stages = std::get<std::vector<std::shared_ptr<const gs::Stage>>>(std::move(result));
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto* native = requests[i].native_stage;
    bindings.push_back({elements[i].get(), std::move(stages[i]),
                        native ? native->native_kernel->metrics : nullptr});
  }
  if (auto failure = attach_kernel_runner_stages(pipeline, bindings))
    return failure;
  // Commit the stopped MLA producers and ingress policy only after the entire
  // prepared cohort attached. Invalid recipes leave all legacy flags untouched.
  if (memory_mode == gs::MemoryCoherencyMode::DemandLocalV1)
    for (auto* element : mla_elements)
      g_object_set(element, "memory-coherency-mode", "demand-local-v1",
          "defer-output-invalidate", TRUE, "output-read-handoff", FALSE, nullptr);
  g_object_set_qdata(G_OBJECT(pipeline), memory_coherency_mode_quark(),
      memory_mode == gs::MemoryCoherencyMode::DemandLocalV1 ? GINT_TO_POINTER(1) : nullptr);
  // Install negotiation only after the cohort has committed. Failed preparation
  // leaves a retryable candidate without duplicate allocation-query probes.
  for (const auto& peer : compact_peers) {
    gst_pad_add_probe(
        peer.get(), GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM,
        +[](GstPad*, GstPadProbeInfo* info, gpointer) -> GstPadProbeReturn {
          auto* query = GST_PAD_PROBE_INFO_QUERY(info);
          if (query && GST_QUERY_TYPE(query) == GST_QUERY_ALLOCATION &&
              simaai::gst::tensor_buffer_advertise_compact_frame_meta(query))
            return GST_PAD_PROBE_HANDLED;
          return GST_PAD_PROBE_OK;
        },
        nullptr, nullptr);
  }
  return {};
}

} // namespace simaai::neat::pipeline_internal::sima
