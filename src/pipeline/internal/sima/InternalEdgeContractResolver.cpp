#include "pipeline/internal/sima/InternalEdgeContractResolver.h"

#include <algorithm>
#include <sstream>
#include <string_view>
#include <utility>

namespace simaai::neat::pipeline_internal::sima::edgecontract {
namespace {

constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);

void set_error(std::string* error_message, std::string message) {
  if (error_message) {
    *error_message = std::move(message);
  }
}

bool matches_index_or_position(int requested, int declared, std::size_t position) {
  if (requested < 0) {
    return false;
  }
  return declared == requested ||
         (declared < 0 && static_cast<std::size_t>(requested) == position);
}

bool same_nonempty(std::string_view lhs, std::string_view rhs) {
  return !lhs.empty() && !rhs.empty() && lhs == rhs;
}

bool stage_identity_matches(const StageStaticSpec& stage, const std::string& key) {
  return !key.empty() &&
         (stage.logical_stage_id == key || stage.element_name == key || stage.plugin_kind == key ||
          stage.kernel_kind == key);
}

bool output_route_matches_binding(const StageOutputRoute& route,
                                  const InputBindingStaticSpec& binding) {
  if (binding.src_output_slot >= 0 && route.output_slot == binding.src_output_slot) {
    return true;
  }
  if (binding.src_logical_output_index >= 0 &&
      route.logical_output_index == binding.src_logical_output_index) {
    return true;
  }
  if (same_nonempty(route.cm_output_name, binding.source_segment_name) ||
      same_nonempty(route.segment_name, binding.source_segment_name)) {
    return true;
  }
  return false;
}

bool logical_output_matches_binding(const LogicalTensorStaticSpec& logical, std::size_t position,
                                    const InputBindingStaticSpec& binding) {
  if (matches_index_or_position(binding.src_logical_output_index, logical.logical_index,
                                position)) {
    return true;
  }
  if (binding.src_output_slot >= 0 && logical.output_slot == binding.src_output_slot) {
    return true;
  }
  if (binding.src_physical_output_index >= 0 &&
      logical.physical_index == binding.src_physical_output_index) {
    return true;
  }
  if (same_nonempty(logical.logical_name, binding.source_segment_name) ||
      same_nonempty(logical.backend_name, binding.source_segment_name) ||
      same_nonempty(logical.segment_name, binding.source_segment_name)) {
    return true;
  }
  return false;
}

bool physical_output_matches_binding(const PhysicalBufferStaticSpec& physical,
                                     std::size_t position,
                                     const InputBindingStaticSpec& binding) {
  if (matches_index_or_position(binding.src_physical_output_index, physical.physical_index,
                                position)) {
    return true;
  }
  if (same_nonempty(physical.segment_name, binding.source_segment_name)) {
    return true;
  }
  return false;
}

bool stage_outputs_match_binding(const StageStaticSpec& stage,
                                 const InputBindingStaticSpec& binding) {
  if (stage_identity_matches(stage, binding.src_stage_id)) {
    return true;
  }
  for (const auto& route : stage.output_order) {
    if (output_route_matches_binding(route, binding)) {
      return true;
    }
  }
  for (std::size_t i = 0; i < stage.logical_outputs.size(); ++i) {
    if (logical_output_matches_binding(stage.logical_outputs[i], i, binding)) {
      return true;
    }
  }
  for (std::size_t i = 0; i < stage.physical_outputs.size(); ++i) {
    if (physical_output_matches_binding(stage.physical_outputs[i], i, binding)) {
      return true;
    }
  }
  return false;
}

std::pair<const StageStaticSpec*, std::size_t>
find_producer_stage(const SimaPluginStaticManifest& manifest, std::size_t consumer_stage_index,
                    const InputBindingStaticSpec& binding) {
  const bool has_explicit_stage_selector = binding.src_stage_index >= 0 ||
                                           !binding.src_stage_id.empty();
  if (binding.src_stage_index >= 0) {
    const auto index = static_cast<std::size_t>(binding.src_stage_index);
    if (index < manifest.stages.size()) {
      const auto& candidate = manifest.stages[index];
      if (binding.src_stage_id.empty() || stage_identity_matches(candidate, binding.src_stage_id)) {
        return {&candidate, index};
      }
    }
  }

  if (!binding.src_stage_id.empty()) {
    for (std::size_t i = 0; i < manifest.stages.size(); ++i) {
      if (i == consumer_stage_index) {
        continue;
      }
      if (stage_identity_matches(manifest.stages[i], binding.src_stage_id)) {
        return {&manifest.stages[i], i};
      }
    }
  }

  if (has_explicit_stage_selector) {
    return {nullptr, kNoIndex};
  }

  const std::size_t end = std::min(consumer_stage_index, manifest.stages.size());
  for (std::size_t i = end; i-- > 0U;) {
    if (stage_outputs_match_binding(manifest.stages[i], binding)) {
      return {&manifest.stages[i], i};
    }
    if (i == 0U) {
      break;
    }
  }

  // Legacy one-edge manifests sometimes omit producer identity.  Keep the fallback deliberately
  // narrow so multi-output/multi-stage manifests must describe their binding.
  if (consumer_stage_index > 0U && consumer_stage_index <= manifest.stages.size()) {
    const auto previous_index = consumer_stage_index - 1U;
    if (manifest.stages[previous_index].logical_outputs.size() <= 1U &&
        manifest.stages[previous_index].physical_outputs.size() <= 1U) {
      return {&manifest.stages[previous_index], previous_index};
    }
  }

  return {nullptr, kNoIndex};
}

const LogicalInputStaticSpec* find_consumer_logical_input(const StageStaticSpec& consumer,
                                                         const InputBindingStaticSpec& binding) {
  for (std::size_t i = 0; i < consumer.logical_inputs.size(); ++i) {
    const auto& logical = consumer.logical_inputs[i];
    if (matches_index_or_position(binding.local_logical_input_index, logical.logical_index, i)) {
      return &logical;
    }
  }
  if (binding.sink_pad_index >= 0) {
    for (std::size_t i = 0; i < consumer.logical_inputs.size(); ++i) {
      const auto& logical = consumer.logical_inputs[i];
      if (matches_index_or_position(binding.sink_pad_index, logical.backend_input_index, i)) {
        return &logical;
      }
    }
  }
  for (const auto& logical : consumer.logical_inputs) {
    if (same_nonempty(logical.logical_name, binding.cm_input_name) ||
        same_nonempty(logical.backend_name, binding.cm_input_name) ||
        same_nonempty(logical.segment_name, binding.cm_input_name)) {
      return &logical;
    }
  }
  if (consumer.logical_inputs.size() == 1U) {
    return &consumer.logical_inputs.front();
  }
  return nullptr;
}

const LogicalTensorStaticSpec*
find_producer_logical_output(const StageStaticSpec& producer,
                             const InputBindingStaticSpec& binding) {
  for (std::size_t i = 0; i < producer.logical_outputs.size(); ++i) {
    const auto& logical = producer.logical_outputs[i];
    if (matches_index_or_position(binding.src_logical_output_index, logical.logical_index, i)) {
      return &logical;
    }
  }
  for (const auto& logical : producer.logical_outputs) {
    if (binding.src_output_slot >= 0 && logical.output_slot == binding.src_output_slot) {
      return &logical;
    }
  }
  for (const auto& route : producer.output_order) {
    if (!output_route_matches_binding(route, binding)) {
      continue;
    }
    for (const auto& logical : producer.logical_outputs) {
      if ((route.logical_output_index >= 0 &&
           logical.logical_index == route.logical_output_index) ||
          (route.output_slot >= 0 && logical.output_slot == route.output_slot) ||
          (route.tensor_index >= 0 && logical.tensor_index == route.tensor_index) ||
          same_nonempty(route.cm_output_name, logical.logical_name) ||
          same_nonempty(route.cm_output_name, logical.backend_name) ||
          same_nonempty(route.segment_name, logical.segment_name)) {
        return &logical;
      }
    }
  }
  for (std::size_t i = 0; i < producer.logical_outputs.size(); ++i) {
    const auto& logical = producer.logical_outputs[i];
    if (logical_output_matches_binding(logical, i, binding)) {
      return &logical;
    }
  }
  if (producer.logical_outputs.size() == 1U) {
    return &producer.logical_outputs.front();
  }
  return nullptr;
}

const PhysicalBufferStaticSpec*
find_producer_physical_output(const StageStaticSpec& producer,
                              const LogicalTensorStaticSpec* producer_logical,
                              const InputBindingStaticSpec& binding) {
  for (std::size_t i = 0; i < producer.physical_outputs.size(); ++i) {
    const auto& physical = producer.physical_outputs[i];
    if (matches_index_or_position(binding.src_physical_output_index, physical.physical_index, i)) {
      return &physical;
    }
  }
  if (producer_logical && producer_logical->physical_index >= 0) {
    for (std::size_t i = 0; i < producer.physical_outputs.size(); ++i) {
      const auto& physical = producer.physical_outputs[i];
      if (matches_index_or_position(producer_logical->physical_index, physical.physical_index, i)) {
        return &physical;
      }
    }
  }
  for (const auto& physical : producer.physical_outputs) {
    if (same_nonempty(physical.segment_name, binding.source_segment_name) ||
        (producer_logical &&
         same_nonempty(physical.segment_name, producer_logical->segment_name))) {
      return &physical;
    }
  }
  if (producer.physical_outputs.size() == 1U) {
    return &producer.physical_outputs.front();
  }
  return nullptr;
}

const PhysicalBufferStaticSpec*
find_consumer_physical_input(const StageStaticSpec& consumer,
                             const LogicalInputStaticSpec* consumer_logical,
                             const InputBindingStaticSpec& binding) {
  if (consumer_logical && consumer_logical->physical_index >= 0) {
    for (std::size_t i = 0; i < consumer.physical_inputs.size(); ++i) {
      const auto& physical = consumer.physical_inputs[i];
      if (matches_index_or_position(consumer_logical->physical_index, physical.physical_index, i)) {
        return &physical;
      }
    }
  }
  for (const auto& physical : consumer.physical_inputs) {
    if ((consumer_logical &&
         same_nonempty(physical.segment_name, consumer_logical->segment_name)) ||
        same_nonempty(physical.segment_name, binding.cm_input_name) ||
        same_nonempty(physical.segment_name, binding.source_segment_name)) {
      return &physical;
    }
  }
  if (consumer.physical_inputs.size() == 1U) {
    return &consumer.physical_inputs.front();
  }
  return nullptr;
}

bool materialization_is_view(TensorMaterializationKind kind) {
  return kind != TensorMaterializationKind::Unknown && kind != TensorMaterializationKind::Direct;
}

bool vectors_differ_when_known(const std::vector<std::int64_t>& lhs,
                               const std::vector<std::int64_t>& rhs) {
  return !lhs.empty() && !rhs.empty() && lhs != rhs;
}

bool strings_differ_when_known(const std::string& lhs, const std::string& rhs) {
  return !lhs.empty() && !rhs.empty() && lhs != rhs;
}

bool consumer_requires_view_contract(const InputBindingStaticSpec& binding,
                                     const LogicalInputStaticSpec* consumer_logical,
                                     const LogicalTensorStaticSpec* producer_logical,
                                     const PhysicalBufferStaticSpec* consumer_physical) {
  if (binding.src_physical_byte_offset != 0) {
    return true;
  }
  if (consumer_physical && consumer_physical->source_byte_offset != 0) {
    return true;
  }
  if (!consumer_logical) {
    return false;
  }
  if (materialization_is_view(consumer_logical->materialization_kind)) {
    return true;
  }
  if (!producer_logical) {
    return consumer_logical->byte_offset != 0;
  }
  if (consumer_logical->byte_offset != producer_logical->byte_offset) {
    return true;
  }
  if (consumer_logical->size_bytes != 0U && producer_logical->size_bytes != 0U &&
      consumer_logical->size_bytes != producer_logical->size_bytes) {
    return true;
  }
  return vectors_differ_when_known(consumer_logical->shape, producer_logical->shape) ||
         vectors_differ_when_known(consumer_logical->stride_bytes,
                                   producer_logical->stride_bytes) ||
         strings_differ_when_known(consumer_logical->dtype, producer_logical->dtype) ||
         strings_differ_when_known(consumer_logical->layout, producer_logical->layout);
}

std::string binding_label(std::size_t consumer_stage_index, std::size_t binding_index) {
  std::ostringstream oss;
  oss << "consumer_stage_index=" << consumer_stage_index;
  if (binding_index != kNoIndex) {
    oss << " binding_index=" << binding_index;
  }
  return oss.str();
}

} // namespace

std::optional<ResolvedEdgeContract> resolve_edge_contract_for_binding(
    const SimaPluginStaticManifest& manifest, std::size_t consumer_stage_index,
    std::size_t binding_index, std::string* error_message) {
  if (consumer_stage_index >= manifest.stages.size()) {
    set_error(error_message, "consumer stage index is out of range");
    return std::nullopt;
  }
  const auto& consumer = manifest.stages[consumer_stage_index];
  if (binding_index >= consumer.input_bindings.size()) {
    set_error(error_message, "input binding index is out of range");
    return std::nullopt;
  }
  auto out = resolve_edge_contract_for_binding(manifest, consumer_stage_index,
                                               consumer.input_bindings[binding_index],
                                               error_message);
  if (out) {
    out->binding_index = binding_index;
  }
  return out;
}

std::optional<ResolvedEdgeContract> resolve_edge_contract_for_binding(
    const SimaPluginStaticManifest& manifest, std::size_t consumer_stage_index,
    const InputBindingStaticSpec& binding, std::string* error_message) {
  if (consumer_stage_index >= manifest.stages.size()) {
    set_error(error_message, "consumer stage index is out of range");
    return std::nullopt;
  }
  const auto& consumer = manifest.stages[consumer_stage_index];
  const auto [producer, producer_index] = find_producer_stage(manifest, consumer_stage_index,
                                                             binding);
  if (!producer) {
    set_error(error_message, "could not resolve producer stage for " +
                                 binding_label(consumer_stage_index, kNoIndex));
    return std::nullopt;
  }

  ResolvedEdgeContract resolved;
  resolved.consumer_stage_index = consumer_stage_index;
  resolved.producer_stage_index = producer_index;
  resolved.binding_index = kNoIndex;
  resolved.consumer_stage = &consumer;
  resolved.producer_stage = producer;
  resolved.binding = &binding;
  resolved.consumer_logical_input = find_consumer_logical_input(consumer, binding);
  resolved.producer_logical_output = find_producer_logical_output(*producer, binding);
  resolved.consumer_physical_input =
      find_consumer_physical_input(consumer, resolved.consumer_logical_input, binding);
  resolved.producer_physical_output =
      find_producer_physical_output(*producer, resolved.producer_logical_output, binding);
  resolved.consumer_requires_view_contract =
      consumer_requires_view_contract(binding, resolved.consumer_logical_input,
                                      resolved.producer_logical_output,
                                      resolved.consumer_physical_input);

  if (!resolved.consumer_logical_input) {
    set_error(error_message, "could not resolve consumer logical input for " +
                                 binding_label(consumer_stage_index, kNoIndex));
    return std::nullopt;
  }
  if (!resolved.producer_logical_output) {
    set_error(error_message, "could not resolve producer logical output for " +
                                 binding_label(consumer_stage_index, kNoIndex));
    return std::nullopt;
  }
  if (!resolved.producer_physical_output) {
    set_error(error_message, "could not resolve producer physical output for " +
                                 binding_label(consumer_stage_index, kNoIndex));
    return std::nullopt;
  }

  set_error(error_message, {});
  return resolved;
}

std::vector<ResolvedEdgeContract> resolve_consumer_edge_contracts(
    const SimaPluginStaticManifest& manifest, std::size_t consumer_stage_index,
    std::string* error_message) {
  std::vector<ResolvedEdgeContract> out;
  if (consumer_stage_index >= manifest.stages.size()) {
    set_error(error_message, "consumer stage index is out of range");
    return out;
  }

  const auto& consumer = manifest.stages[consumer_stage_index];
  out.reserve(consumer.input_bindings.size());
  for (std::size_t i = 0; i < consumer.input_bindings.size(); ++i) {
    std::string edge_error;
    auto resolved =
        resolve_edge_contract_for_binding(manifest, consumer_stage_index, i, &edge_error);
    if (!resolved) {
      set_error(error_message, edge_error);
      out.clear();
      return out;
    }
    out.push_back(*resolved);
  }
  set_error(error_message, {});
  return out;
}

} // namespace simaai::neat::pipeline_internal::sima::edgecontract
