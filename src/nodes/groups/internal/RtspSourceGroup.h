#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "nodes/io/RTSPInput.h"

#include <algorithm>
#include <iosfwd>
#include <span>

namespace simaai::neat {

struct BuildResult;
struct NodeReport;

// The group factory owns the boundary. A standalone or sliced RTSPInput
// resets only the transport when its encoded endpoint is absent.
struct RtspSourceGroup {
  static void bind(const std::vector<std::shared_ptr<Node>>& nodes) {
    static_cast<RTSPInput&>(*nodes.front()).encoded_group_end_ = nodes.back();
  }

  explicit RtspSourceGroup(std::span<const std::shared_ptr<Node>> nodes = {}) {
    if (!nodes.empty())
      source = dynamic_cast<const RTSPInput*>(nodes.front().get());
    if (source) {
      auto end = source->encoded_group_end_.lock();
      last = end && std::find(nodes.begin(), nodes.end(), end) != nodes.end() ? end.get() : source;
    }
  }

  void render(std::ostream& stream, BuildResult& build, const Node& node,
              const NodeReport& report) const;

  const RTSPInput* source = nullptr;
  const Node* last = nullptr;
};
} // namespace simaai::neat
