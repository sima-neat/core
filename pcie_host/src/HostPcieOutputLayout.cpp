#include "HostPcieOutputLayout.h"

#include <limits>
#include <stdexcept>
#include <vector>

namespace simaai::neat::pcie::internal {
namespace {

constexpr std::size_t kMaxRegions = 256;
constexpr std::uint64_t kMaxBytes = 128U * 1024U * 1024U;

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
  if (!caps || gst_caps_get_size(caps) != 1)
    invalid("missing single output CAPS");
  const auto* s = gst_caps_get_structure(caps, 0);
  if (!gst_structure_has_field(s, "pcie-return-layout-version") &&
      !gst_structure_has_field(s, "pcie-return-raw-bytes") &&
      !gst_structure_has_field(s, "pcie-return-regions"))
    return out;

  guint version = 0;
  guint64 raw_bytes = 0;
  const GValue* regions = gst_structure_get_value(s, "pcie-return-regions");
  if (!gst_structure_get_uint(s, "pcie-return-layout-version", &version) || version != 2 ||
      !gst_structure_get_uint64(s, "pcie-return-raw-bytes", &raw_bytes) || !raw_bytes ||
      raw_bytes > kMaxBytes || !regions || !GST_VALUE_HOLDS_ARRAY(regions))
    invalid("missing or incompatible raw tensor layout");
  const guint count = gst_value_array_get_size(regions);
  if (!count || count % 2 || count / 2 > kMaxRegions)
    invalid("invalid carrier count");

  struct Region {
    std::size_t offset, size, packed;
  };
  std::vector<Region> carriers;
  std::size_t packed = 0;
  for (guint i = 0; i < count; i += 2) {
    const auto* offset_value = gst_value_array_get_value(regions, i);
    const auto* size_value = gst_value_array_get_value(regions, i + 1);
    if (!G_VALUE_HOLDS_UINT64(offset_value) || !G_VALUE_HOLDS_UINT64(size_value))
      invalid("carrier offsets and sizes must be uint64");
    const auto offset = g_value_get_uint64(offset_value);
    const auto size = g_value_get_uint64(size_value);
    if (!size || offset > raw_bytes || size > raw_bytes - offset || size > kMaxBytes - packed)
      invalid("carrier exceeds raw allocation");
    for (const auto& other : carriers)
      if (offset < other.offset + other.size && other.offset < offset + size)
        invalid("overlapping carriers");
    carriers.push_back({static_cast<std::size_t>(offset), static_cast<std::size_t>(size), packed});
    packed += static_cast<std::size_t>(size);
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
  out.raw_bytes = static_cast<std::size_t>(raw_bytes);
  return out;
}

} // namespace simaai::neat::pcie::internal
