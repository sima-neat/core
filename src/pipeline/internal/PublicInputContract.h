#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "nodes/io/Input.h"

#include <cstddef>
#include <stdexcept>

namespace simaai::neat::pipeline_internal {

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

} // namespace simaai::neat::pipeline_internal
