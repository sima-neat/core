#pragma once

#include "PcieModelFactsReader.h"

#include <gst/gst.h>

namespace simaai::neat::pcie::internal {

inline bool is_bbox_output_caps(const GstCaps* caps) {
  const auto* structure =
      caps && gst_caps_get_size(caps) ? gst_caps_get_structure(caps, 0) : nullptr;
  return structure && gst_structure_has_name(structure, "application/vnd.simaai.tensor") &&
         g_strcmp0(gst_structure_get_string(structure, "format"), "BBOX") == 0;
}

struct HostPcieOutputLayout {
  PcieModelFacts facts;
  // Zero for a variable byte stream; otherwise the exact raw received extent.
  std::size_t raw_bytes = 0;
};

// Bind MPK logical tensors to negotiated regions once per session. Reserved
// allocation capacity is independent of each tensor's accessed stride footprint.
HostPcieOutputLayout resolve_output_layout(const PcieModelFacts& facts, const GstCaps* caps);

} // namespace simaai::neat::pcie::internal
