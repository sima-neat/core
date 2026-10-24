#include "HostPcieOutputLayout.h"
#include <neat_pcie_return_layout.h>

#include <limits>
#include <stdexcept>
#include <vector>

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
  struct Region {
    std::size_t offset, size, packed;
  };
  std::vector<Region> carriers;
  carriers.reserve(layout.count);
  std::size_t packed = 0;
  for (guint i = 0; i < layout.count; ++i) {
    const auto& region = layout.regions[i];
    carriers.push_back(
        {static_cast<std::size_t>(region.offset), static_cast<std::size_t>(region.size), packed});
    packed += static_cast<std::size_t>(region.size);
  }
  if (packed != facts.packed_output_bytes)
    invalid("carrier extent differs from MPK contract");
  for (auto& tensor : out.facts.outputs) {
    const auto span = tensor_span(tensor);
    bool found = false;
    for (const auto& carrier : carriers) {
      if (tensor.payload_offset < carrier.packed)
        continue;
      const auto local = tensor.payload_offset - carrier.packed;
      if (local >= carrier.size || span > carrier.size - local)
        continue;
      tensor.payload_offset = carrier.offset + local;
      found = true;
      break;
    }
    if (!found)
      invalid("tensor span does not fit its physical carrier");
  }
  out.raw_bytes = static_cast<std::size_t>(layout.raw_bytes);
  return out;
}

} // namespace simaai::neat::pcie::internal
