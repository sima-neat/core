#include "HostPcieOutputLayout.h"
#include <neat_pcie_return_layout.h>

#include <limits>
#include <stdexcept>
#include <utility>

static_assert(NEAT_PCIE_RETURN_CAPS_VERSION == 3u,
              "Core PCIe requires logical return-layout version 3");

namespace simaai::neat::pcie::internal {
namespace {

[[noreturn]] void invalid(const char* reason) {
  throw std::runtime_error(std::string("Invalid PCIe return layout: ") + reason);
}

std::size_t tensor_span(const PcieTensorFact& tensor) {
  if (tensor.transport_strides_bytes.empty())
    return tensor.size_bytes;
  if (tensor.shape.size() != tensor.transport_strides_bytes.size())
    invalid("tensor stride rank mismatch");
  std::size_t elements = 1;
  for (auto dim : tensor.shape) {
    if (dim <= 0 ||
        static_cast<std::uint64_t>(dim) > std::numeric_limits<std::size_t>::max() / elements)
      invalid("tensor shape overflows");
    elements *= static_cast<std::size_t>(dim);
  }
  if (!tensor.size_bytes || tensor.size_bytes % elements != 0)
    invalid("invalid tensor element size");
  std::size_t span = tensor.size_bytes / elements;
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    const auto stride = tensor.transport_strides_bytes[i];
    const auto count = static_cast<std::size_t>(tensor.shape[i] - 1);
    if (stride < 0 || (count && static_cast<std::uint64_t>(stride) >
                                    (std::numeric_limits<std::size_t>::max() - span) / count))
      invalid("tensor stride span overflows");
    span += count * static_cast<std::size_t>(stride);
  }
  return span;
}

} // namespace

HostPcieOutputLayout resolve_output_layout(const PcieModelFacts& facts, const GstCaps* caps) {
  HostPcieOutputLayout out{facts};
  NeatPcieReturnLayout layout{};
  GError* error = nullptr;
  if (!neat_pcie_return_layout_read_caps(caps, &layout, &error)) {
    const std::string message = error ? error->message : "invalid Scheduler return description";
    g_clear_error(&error);
    throw std::runtime_error("Invalid PCIe return layout: " + message);
  }
  if (!layout.count)
    return out;
  if (is_bbox_output_caps(caps)) {
    if (layout.count != 1)
      invalid("BBOX output requires one logical region");
    const auto& region = layout.regions[0];
    PcieTensorFact tensor;
    tensor.name = "BBOX";
    tensor.dtype = "UINT8";
    tensor.shape = {static_cast<std::int64_t>(region.size)};
    tensor.size_bytes = region.size;
    tensor.tensor_index = tensor.physical_index = 0;
    tensor.payload_offset = region.offset;
    out.facts.outputs = {std::move(tensor)};
    out.facts.packed_output_bytes = region.size;
    out.facts.dense_output_bytes = 0;
  } else {
    if (layout.count != facts.outputs.size())
      invalid("logical output count differs from MPK contract");
    const auto overlaps = [](std::size_t left, std::size_t left_size, std::size_t right,
                             std::size_t right_size) {
      return left <= right ? right - left < left_size : left - right < right_size;
    };
    for (std::size_t i = 0; i < facts.outputs.size(); ++i) {
      const auto& tensor = facts.outputs[i];
      const auto span = tensor_span(tensor);
      const auto& region = layout.regions[i];
      if (tensor.payload_offset > facts.packed_output_bytes ||
          span > facts.packed_output_bytes - tensor.payload_offset || region.size != span)
        invalid("logical output extent differs from MPK contract");
      for (std::size_t j = 0; j < i; ++j) {
        const auto& previous = facts.outputs[j];
        const auto& previous_region = layout.regions[j];
        const bool aliased =
            overlaps(tensor.payload_offset, span, previous.payload_offset, previous_region.size);
        if (aliased !=
            overlaps(region.offset, region.size, previous_region.offset, previous_region.size))
          invalid("logical output alias differs from MPK contract");
        if (aliased) {
          const bool same_relative_offset =
              tensor.payload_offset <= previous.payload_offset
                  ? region.offset <= previous_region.offset &&
                        previous_region.offset - region.offset ==
                            previous.payload_offset - tensor.payload_offset
                  : previous_region.offset <= region.offset &&
                        region.offset - previous_region.offset ==
                            tensor.payload_offset - previous.payload_offset;
          if (!same_relative_offset)
            invalid("logical output alias offset differs from MPK contract");
        }
      }
      out.facts.outputs[i].payload_offset = region.offset;
    }
  }
  out.raw_bytes = static_cast<std::size_t>(layout.raw_bytes);
  return out;
}

} // namespace simaai::neat::pcie::internal
