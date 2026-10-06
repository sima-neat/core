#pragma once

#include "PcieModelFactsReader.h"

#include <gst/gst.h>

namespace simaai::neat::pcie::internal {

struct HostPcieOutputLayout {
  PcieModelFacts facts;
  // Zero for legacy packed CAPS; otherwise the exact raw received extent.
  std::size_t raw_bytes = 0;
};

// Translate MPK packed offsets into runtime arena offsets once per session.
// The CAPS describe physical carriers in MPK order, not logical tensor order.
HostPcieOutputLayout resolve_output_layout(const PcieModelFacts& facts, const GstCaps* caps);

} // namespace simaai::neat::pcie::internal
