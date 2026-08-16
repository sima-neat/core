#pragma once

#include "HostPcieTensorPayload.h"
#include "PcieModelFactsReader.h"

#include <memory>

#include <gst/gst.h>

namespace simaai::neat::pcie::internal {

struct MappedSample {
  GstSample* sample = nullptr;
  GstBuffer* buffer = nullptr;
  GstMapInfo map{};
  bool mapped = false;

  ~MappedSample();
};

void attach_tensor_set_meta(GstBuffer* buffer, const std::vector<TensorMetaSpan>& spans,
                            const std::vector<PcieTensorFact>& input_facts);

} // namespace simaai::neat::pcie::internal
