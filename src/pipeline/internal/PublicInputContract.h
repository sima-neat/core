#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "nodes/io/Input.h"
#include "model/internal/ModelRouteRetarget.h"
#include "builder/NodeContractProvider.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/InputRouteProcessor.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <stdexcept>

namespace simaai::neat::pipeline_internal {

// Preserve an alias already established by the checked model consumer, rather
// than guessing that every leading singleton dimension is a batch axis.
inline std::vector<std::int64_t>
model_input_wire_shape(const sima::MpkTensorContract& authored,
                       const sima::LogicalInputStaticSpec& consumer) {
  if (authored.shape_semantics != sima::MpkShapeSemantics::Geometry ||
      authored.mpk_shape.size() != 4U || authored.mpk_shape.front() != 1 ||
      authored.logical_shape.size() != 3U || consumer.shape != authored.logical_shape ||
      !std::equal(authored.logical_shape.begin(), authored.logical_shape.end(),
                  authored.mpk_shape.begin() + 1) ||
      std::any_of(consumer.shape.begin(), consumer.shape.end(), [](auto dim) { return dim <= 0; }))
    return {};
  return consumer.shape;
}

// Interpret an unspecified application Input from its directly bound model ingress.
// Source allocation, scheduling and geometry remain application-owned. In particular,
// a downstream fragment that merely overlaps this pipeline is not an ingress owner.
template <typename View> InputOptions public_input_options(const View& view, std::size_t vertex) {
  const auto* input = dynamic_cast<const Input*>(view.vertices.at(vertex).get());
  if (!input) {
    throw std::logic_error("public_input_options requires an Input vertex");
  }
  InputOptions options = input->options();
  if (!options.caps_override.empty() ||
      (!options.format.empty() && !is_raw_video_format(options.format.tag)) ||
      options.payload_type == PayloadType::Tensor || options.payload_type == PayloadType::Encoded ||
      (options.payload_type == PayloadType::Image && !options.format.empty())) {
    return options;
  }

  const InputOptions* ingress = nullptr;
  for (const auto& edge : view.edges) {
    if (edge.to == vertex) {
      // This is an internal boundary declaration, not an application-owned source.
      return options;
    }
    if (edge.from != vertex) {
      continue;
    }
    bool matched = false;
    for (const auto& fragment : view.fragments) {
      if (fragment.graph_start != edge.to || !fragment.boundary_hints.has_value()) {
        continue;
      }
      const auto& hints = *fragment.boundary_hints;
      if (hints.ingress_inputs.size() != 1U || hints.tensor_mode) {
        continue;
      }
      const auto& candidate = hints.ingress_inputs.front();
      if (candidate.payload_type != PayloadType::Image || candidate.format.empty()) {
        continue;
      }
      if (ingress && options.format.empty() && candidate.format.str() != ingress->format.str()) {
        // Inference is optional. An explicit Sample/Tensor can still resolve
        // this boundary, so overlapping declarations must not forbid the graph.
        return options;
      }
      ingress = &candidate;
      matched = true;
    }
    if (!matched) {
      return options;
    }
  }
  if (!ingress) {
    return options;
  }
  if (options.payload_type == PayloadType::Auto) {
    options.payload_type = ingress->payload_type;
  }
  if (options.format.empty()) {
    options.format = ingress->format;
  }
  return options;
}

// Internal boundaries and fanouts have no single application-ingress owner.
template <typename View>
std::optional<std::size_t> direct_public_input_consumer(const View& view, std::size_t vertex) {
  if (!dynamic_cast<const Input*>(view.vertices.at(vertex).get()))
    return std::nullopt;
  std::optional<std::size_t> consumer;
  for (const auto& edge : view.edges) {
    if (edge.to == vertex)
      return std::nullopt;
    if (edge.from == vertex) {
      if (consumer)
        return std::nullopt;
      consumer = edge.to;
    }
  }
  return consumer;
}

// A Model-owned transform can authorize its batch-one wire alias. A custom
// stage before the Model cannot assign meaning to the application's axes.
template <typename View>
std::vector<std::int64_t> public_model_input_shape(const View& view, std::size_t vertex,
                                                   const InputRouteProcessorPtr& route = {}) {
  const auto consumer = direct_public_input_consumer(view, vertex);
  if (!consumer)
    return {};
  const auto& node = view.vertices.at(*consumer);
  // Separate public endpoints are still individual tensors. Their Model's
  // representative wire policy belongs after the existing fan-in join, not on
  // each endpoint before that join.
  const bool joined_ingress =
      std::any_of(view.edges.begin(), view.edges.end(),
                  [&](const auto& edge) { return edge.to == *consumer && edge.from != vertex; });
  if (route && !joined_ingress) {
    auto shape = route->model_input_shape(node);
    if (!shape.empty())
      return shape;
  }
  for (const auto& fragment : view.fragments) {
    const auto start = fragment.graph_start;
    if (joined_ingress || start < vertex || start > *consumer || fragment.graph_end <= *consumer ||
        !fragment.boundary_hints || !fragment.boundary_hints->input_route_processor)
      continue;
    auto shape = fragment.boundary_hints->input_route_processor->model_input_shape(node);
    if (!shape.empty())
      return shape;
  }
  const auto* lineage = internal::node_model_lineage_binding(node);
  const auto* contract = internal::node_model_processcvu_contract(node);
  if (!lineage || lineage->stage_role != internal::ModelLineageStageRole::Preprocess ||
      node->kind() == "Preproc" || !contract ||
      contract->runtime_contract.logical_inputs.size() != 1U)
    return {};
  const auto& input = contract->runtime_contract.logical_inputs.front();
  // Some MPKs omit the layout token; their checked logical shape still owns the wire contract.
  if (!input.layout.empty() && input.layout != "HWC" && input.layout != "NHWC" &&
      input.layout != "CHW" && input.layout != "NCHW")
    return {};
  auto shape = input.shape;
  if (shape.size() == 4U && shape.front() == 1)
    shape.erase(shape.begin());
  return shape.size() == 3U ? shape : std::vector<std::int64_t>{};
}

// Only the direct, unambiguous consumer of an application Input can author
// this policy. Compilation here is cold metadata work only.
template <typename View>
std::optional<OrderedPublicInputContract> ordered_public_input_contract(const View& view,
                                                                        std::size_t vertex) {
  const auto consumer = direct_public_input_consumer(view, vertex);
  if (!consumer)
    return std::nullopt;
  for (const auto& fragment : view.fragments) {
    if (fragment.graph_start <= *consumer && *consumer < fragment.graph_end &&
        fragment.boundary_hints && fragment.boundary_hints->input_route_processor)
      return std::nullopt;
  }
  const auto* provider =
      dynamic_cast<const NodeContractProvider*>(view.vertices.at(*consumer).get());
  if (!provider)
    return std::nullopt;
  const auto definition = provider->contract_definition();
  if (definition.inputs.size() < 2U ||
      std::any_of(definition.fields.begin(), definition.fields.end(), [](const auto& field) {
        return field.source != ContractFieldSource::Fixed &&
               field.source != ContractFieldSource::BuilderOption;
      }))
    return std::nullopt;
  // The normal contextual compile happens after seed specialization, and
  // connected segments may be lazy. Query only context-free authors here;
  // never supply a seed that could redefine the expected application geometry.
  ContractCompileInput input;
  input.pipeline_label = "public_input";
  input.node_index = static_cast<int>(*consumer);
  input.processcvu_requested_run_target = view.options.processcvu_requested_run_target;
  input.processcvu = view.options.processcvu;
  CompiledNodeContract compiled;
  std::string error;
  if (!provider->compile_node_contract(input, &compiled, &error))
    return std::nullopt;
  return std::move(compiled.public_input_contract);
}

} // namespace simaai::neat::pipeline_internal
