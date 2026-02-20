/**
 * @file
 * @ingroup graph
 * @brief Hybrid graph node interface (backend + ports + output spec).
 */
#pragma once

#include "graph/GraphTypes.h"

#include <string>
#include <vector>

namespace simaai::neat::graph {

/**
 * @brief Base class for hybrid graph nodes.
 *
 * This interface is STL-only and does not own GStreamer runtime objects.
 */
class Node {
public:
  virtual ~Node() = default;

  // Backend type (pipeline-backed vs stage-backed)
  virtual Backend backend() const = 0;

  // Deterministic type label
  virtual std::string kind() const = 0;

  // Optional user label
  virtual std::string user_label() const {
    return "";
  }

  // Declared input/output ports
  virtual std::vector<PortDesc> input_ports() const = 0;
  virtual std::vector<PortDesc> output_ports() const = 0;

  // Output spec inference for compiler (may return unknown spec).
  virtual OutputSpec output_spec(const std::vector<OutputSpec>& inputs, PortId out_port) const {
    (void)inputs;
    (void)out_port;
    return OutputSpec{};
  }
};

} // namespace simaai::neat::graph
