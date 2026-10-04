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

TensorList tensors_from_output_payload(const std::shared_ptr<MappedSample>& owner,
                                       const PcieModelFacts& facts);

void attach_tensor_set_meta(GstBuffer* buffer, const std::vector<TensorMetaSpan>& spans,
                            const std::vector<PcieTensorFact>& input_facts,
                            const PcieTensorFact* packed_input = nullptr);

} // namespace simaai::neat::pcie::internal
