#pragma once

#include "PcieModelFactsReader.h"

#include <gst/gst.h>

namespace simaai::neat::pcie::internal {

struct HostPcieOutputLayout {
  PcieModelFacts facts;
  // Zero for a variable byte stream; otherwise the exact raw received extent.
  std::size_t raw_bytes = 0;
  // Canonical tag for a decoded byte stream without fixed regions.
  const char* byte_stream_format = nullptr;
};

// Bind MPK logical tensors to negotiated regions once per session. Reserved
// allocation capacity is independent of each tensor's accessed stride footprint.
HostPcieOutputLayout resolve_output_layout(const PcieModelFacts& facts, const GstCaps* caps,
                                           bool expects_bbox_output = false);

} // namespace simaai::neat::pcie::internal
